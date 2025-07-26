import json
from typing import Callable
from pathlib import Path

import pandas as pd

from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistant_texts import AssistantTexts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions


from utils import get_questions_by_assistant, get_texts_by_assistant, get_assistant_language, is_normalized_text_matching, filter_matchings_by_language, get_default_embedding_model, cosine_distance, get_user_prompt_id_from_source_node

import vocal.common.static  # noqa: F401
from vocal.common.embedding_utils import EmbeddingFunction
from vocal.common.embedding_utils import compute_embeddings_gpu


CALL_TRANSCRIPTS_PATH = "/www/files/call_transcripts"  # Path to saved call transcripts
MATCHING_PATH = "/www/files/matching.xlsx"  # Path to save matchings extracted from call transcripts
MATCHING_ES_PATH = "/www/files/matching_es.xlsx"  # Spanigh matchings
MATCHING_DISTANCE_ES_PATH = "/www/files/matching_es.xlsx"  # Spanigh matchings
MATCHING_DISTANCE_PATH = "/www/files/matching_distance.xlsx"  # Matchings with distance between query & match embeddings
LLM_MATCHING_UP_PATH = "/www/files/llm_matching_up.xlsx"  # Path to save LLM user prompt matchings
LLM_MATCHING_AO_PATH = "/www/files/llm_matching_ao.xlsx"  # Path to save LLM assistant output matchings


def extract_embedding_matchings(
    call_transcripts_path: str = CALL_TRANSCRIPTS_PATH,
) -> pd.DataFrame:
    """
    Extract matchings from call transcripts that involve embedding matching (with distance > 0.0).
    """
    df = pd.DataFrame(
        columns=[
            "assistant",
            "call_id",
            "query",
            "match",
            "language",
            "source",
            "distance",
        ]
    )
    seen_conv_path_ids = set()

    for assistant_dir in Path(CALL_TRANSCRIPTS_PATH).iterdir():
        assistant = assistant_dir.name
        language = get_assistant_language(assistant)

        for file in assistant_dir.iterdir():
            call_id = Path(file.name).stem
            message_list = json.loads(file.read_text(encoding="utf-8"))
            ut_query = None

            for message in message_list:
                if message["role"] == "USER":
                    # User prompt query
                    ut_query = message["text"]  # live transcription
                    if message["matching"]:
                        ut_query = message["matching"]["original"]  # offline transcription

                if (
                    message["matching"] is None  # USER message, there's no matching
                    or "distance" not in message["matching"]  # ASSISTANT message with no matching found
                    or "distance" in message["matching"]  # ASSISTANT message with distance = 0.0
                    and message["matching"]["distance"] == 0.0
                ):
                    continue

                if "conv_path_id" in message["matching"]:
                    # It was the user prompt that was matched
                    if message["matching"]["conv_path_id"] not in seen_conv_path_ids:
                        # make sure that each conv_path_id is processed only once
                        seen_conv_path_ids.add(message["matching"]["conv_path_id"])

                        user_prompt_id = message["matching"]["user_prompt_id"]
                        up_example_match = UserPrompts.get_by_ids([user_prompt_id])[0].text
                        df.loc[len(df)] = {
                            "assistant": assistant,
                            "call_id": call_id,
                            "query": ut_query,
                            "match": up_example_match,
                            "language": language,
                            "source": "DB_USER_PROMPT",
                            "distance": message["matching"]["distance"],
                        }
                else:
                    # It was the assistant output that was matched
                    ao_match = None
                    if message["source"] == "DB_QUESTION":
                        ao_match = AssistantQuestions.get_by_ids([message["matching"]["id"]])
                    elif message["source"] == "DB_TEXT":
                        ao_match = AssistantTexts.get_by_ids([message["matching"]["id"]])
                    if ao_match:
                        ao_query = message["matching"]["original"]  # Assistant output query
                        ao_match = ao_match[0].text
                        df.loc[len(df)] = {
                            "assistant": assistant,
                            "call_id": call_id,
                            "query": ao_query,
                            "match": ao_match,
                            "language": language,
                            "source": message["source"],
                            "distance": message["matching"]["distance"],
                        }

    return df


def compute_matching_distance_by_language(
    df: pd.DataFrame,
    get_embedding_model_per_language: Callable[[str], str] = get_default_embedding_model,
):
    """
    Compute the distance between query and match embeddings using one embedding model per language.
    """
    distances = []
    embedding_models = []
    for _idx, row in df.iterrows():
        embedding_model = get_embedding_model_per_language(row["language"])
        embed_fn = EmbeddingFunction(embedding_model)

        query_embedding = embed_fn([row["query"]])[0]
        match_embedding = embed_fn([row["match"]])[0]

        distances.append(cosine_distance(query_embedding, match_embedding))
        embedding_models.append(embedding_model)

    df["distance using default embedding model"] = distances
    df["embedding model"] = embedding_models


def compute_matching_distance_with_multiple_embedding_models(df: pd.DataFrame, embedding_models: list[str]):
    """
    Compute the distance between query and match embeddings using the specified embedding models.
    """
    for model_name in embedding_models:
        query_embeddings = compute_embeddings_gpu(model_name, df["query"].tolist())
        matching_embeddings = compute_embeddings_gpu(model_name, df["match"].tolist())

        distances = cosine_distance(query_embeddings, matching_embeddings)

        df[model_name + " distance"] = distances


def extract_LLM_matchings(
    call_transcripts_path: str = CALL_TRANSCRIPTS_PATH,
) -> pd.DataFrame:
    """
    Extract matchings from call transcripts that involve LLM matchings (with distance = 0.0 and that is neither the initial message nor exact matching).
    """
    up_matchings = pd.DataFrame(
        columns=[
            "conversation",
            "user_prompt",
            "list_of_possible_intents",
            "list_of_possible_answers",
            "match",
            "assistant",
            "call_id",
            "source",
            "language",
        ]
    )
    ao_matchings = pd.DataFrame(
        columns=[
            "conversation",
            "source_question",
            "list_of_possible_questions",
            "match",
            "assistant",
            "call_id",
            "source",
            "language",
        ]
    )

    seen_conv_path_ids = set()

    for assistant_dir in Path(CALL_TRANSCRIPTS_PATH).iterdir():
        assistant = assistant_dir.name
        language = get_assistant_language(assistant)
        assistant_questions = get_questions_by_assistant(assistant)
        assistant_texts = get_texts_by_assistant(assistant)

        for file in assistant_dir.iterdir():
            call_id = Path(file.name).stem
            message_list = json.loads(file.read_text(encoding="utf-8"))
            ut_query = None
            conversation = ""

            for message in message_list[1:]:
                list_of_possible_answers = []
                list_of_possible_intents = []
                list_of_possible_questions = []

                conversation += f"{message['role']}: {message['text']}\n"

                if message["role"] == "USER":
                    # User prompt query
                    ut_query = message["text"]  # live transcription
                    if message["matching"]:
                        ut_query = message["matching"]["original"]  # offline transcription

                if message["matching"] is not None and "distance" in message["matching"] and message["matching"]["distance"] == 0.0:
                    # A matching with distance = 0.0 is either an exact match or initial message matching or LLM matching

                    if "conv_path_id" in message["matching"]:
                        # It was the user prompt that was matched

                        # ignore initial message matching
                        if ut_query is None or "init" in ut_query.lower():
                            continue

                        if message["matching"]["conv_path_id"] not in seen_conv_path_ids:
                            # make sure that each conv_path_id is processed only once
                            seen_conv_path_ids.add(message["matching"]["conv_path_id"])

                            if message["matching"]["user_prompt_id"] is not None:
                                user_prompt_id = message["matching"]["user_prompt_id"]

                                # handle error case where user_prompt_id = False or conv_path_id
                                if user_prompt_id is False or UserPrompts.get_by_ids([user_prompt_id]) == []:
                                    continue

                                # ignore exact matching
                                up_example_match = UserPrompts.get_by_ids([user_prompt_id])[0].text
                                if is_normalized_text_matching(ut_query, up_example_match):
                                    continue
                            else:
                                up_example_match = None

                            # for LLM matching, extract conversation, user_prompt, list_of_possible_intents, and list_of_possible_answers
                            source_node_id = message["matching"]["conv_path_id"].split("_")[0]
                            primary_ids = get_user_prompt_id_from_source_node(source_node_id)
                            primary_ups = UserPrompts.get_by_ids(primary_ids)

                            for primary_up in primary_ups:
                                list_of_possible_intents.append(primary_up.text)

                                attached_up_ids = primary_up.attached_user_prompt_ids
                                if attached_up_ids:
                                    list_of_possible_answers.extend([up.text for up in UserPrompts.get_by_ids(attached_up_ids)])

                            up_matchings.loc[len(up_matchings)] = {
                                "conversation": conversation,
                                "user_prompt": ut_query,
                                "list_of_possible_intents": list_of_possible_intents,
                                "list_of_possible_answers": list_of_possible_answers,
                                "match": up_example_match,
                                "assistant": assistant,
                                "call_id": call_id,
                                "source": "DB_USER_PROMPT",
                                "language": language,
                            }

                    else:
                        # It was the assistant output that was matched

                        # ignore initial message matching
                        if message["id"] == "init":
                            continue

                        ao_query = message["matching"]["original"]  # Assistant output query

                        ao_match = None
                        if message["source"] == "DB_QUESTION":
                            ao_match = AssistantQuestions.get_by_ids([message["matching"]["id"]])
                        elif message["source"] == "DB_TEXT":
                            ao_match = AssistantTexts.get_by_ids([message["matching"]["id"]])
                        if ao_match:
                            ao_match = ao_match[0].text

                            # ignore exact matching
                            if is_normalized_text_matching(ao_query, ao_match):
                                continue

                            # for LLM matching, extract conversation, source_question, and list_of_possible_questions
                            if message["source"] == "DB_QUESTION":
                                list_of_possible_questions.extend(assistant_questions)
                            elif message["source"] == "DB_TEXT":
                                list_of_possible_questions.extend(assistant_texts)

                            ao_matchings.loc[len(ao_matchings)] = {
                                "conversation": conversation,
                                "source_question": ao_query,
                                "list_of_possible_questions": list_of_possible_questions,
                                "match": ao_match,
                                "assistant": assistant,
                                "call_id": call_id,
                                "source": message["source"],
                                "language": language,
                            }

    return up_matchings, ao_matchings


def user_text_matching(user_prompt: str, list_of_possible_intents: list[str], list_of_possible_answers: list[str]) -> dict:
    """Matches """


if __name__ == "__main__":
    up_matchings, ao_matchings = extract_LLM_matchings(CALL_TRANSCRIPTS_PATH)
    ao_matchings.to_excel(LLM_MATCHING_AO_PATH, index=False)
    up_matchings.to_excel(LLM_MATCHING_UP_PATH, index=False)
    

    df = extract_embedding_matchings(CALL_TRANSCRIPTS_PATH)
    df.to_excel(MATCHING_PATH, index=False)

    df = pd.read_excel(MATCHING_PATH)

    compute_matching_distance_by_language(df)
    df.to_excel(MATCHING_DISTANCE_PATH, index=False)

    df_es = filter_matchings_by_language(df, "es")
    df_es.to_excel(MATCHING_ES_PATH, index=False)
    df_es = pd.read_excel(MATCHING_ES_PATH)

    # embedding_models = [MONOINGUAL_EMBEDDING_MODELS_PER_LANGUAGE["es"][0]]
    embedding_models = ["UAE-Large-V1"]
    compute_matching_distance_with_multiple_embedding_models(df_es, embedding_models)
    df_es.to_excel(MATCHING_DISTANCE_ES_PATH, index=False)
