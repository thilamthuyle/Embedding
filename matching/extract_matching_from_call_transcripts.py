import json
import logging
import ast
from typing import Callable
from pathlib import Path
import asyncio
import pandas as pd
import sqlmodel as sm

from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistant_texts import AssistantTexts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions
from getvocal.datamodel.sql.assistant_answers import AssistantAnswers
from getvocal.datamodel.sql.conversational_paths import ConversationalPaths

from utils import (
    get_assistant_language,
    check_normalized_text_matching,
    get_default_embedding_model,
    cosine_distance,
    remove_last_assistant_messages,
    user_text_matching,
)

import vocal.common.static  # noqa: F401
from vocal.common.embedding_utils import EmbeddingFunction
from vocal.common.embedding_utils import compute_embeddings_gpu

# Configure logging to show DEBUG level messages
# Force reconfiguration even if logging was already configured
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


CALL_TRANSCRIPTS_PATH = "/www/files/call_transcripts"  # Path to saved call transcripts
MATCHING_PATH = "/www/files/matching.xlsx"  # Path to save matchings extracted from call transcripts
MATCHING_ES_PATH = "/www/files/matching_es.xlsx"  # Spanigh matchings
MATCHING_DISTANCE_ES_PATH = "/www/files/matching_es.xlsx"  # Spanigh matchings
MATCHING_DISTANCE_PATH = (
    "/www/files/matching_distance.xlsx"  # Matchings with distance between query & match embeddings
)
LLM_MATCHING_UP_PATH = "/www/files/llm_matching_up.xlsx"  # Path to save LLM user prompt matchings
LLM_MATCHING_AO_PATH = (
    "/www/files/llm_matching_ao.xlsx"  # Path to save LLM assistant output matchings
)


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
                    or "distance"
                    not in message["matching"]  # ASSISTANT message with no matching found
                    or "distance" in message["matching"]  # ASSISTANT message with distance = 0.0
                    and message["matching"]["distance"] == 0.0
                ):
                    continue

                if "conv_path_id" in message["matching"]:
                    # It was the user prompt that was matched
                    if message["matching"]["conv_path_id"] not in seen_conv_path_ids:
                        # make sure that each conv_path_id is processed only once
                        seen_conv_path_ids.add(message["matching"]["conv_path_id"])

                        user_prompt_id = message["matching"]["user_prompt_id"]  # NON PRIMARY
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


def compute_matching_distance_with_multiple_embedding_models(
    df: pd.DataFrame, embedding_models: list[str]
):
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
            "user_text",
            "possible_conv_paths",
            "match_pred",
            "assistant",
            "call_id",
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
        # assistant_questions = get_questions_by_assistant(assistant)
        # assistant_texts = get_texts_by_assistant(assistant)

        for file in assistant_dir.iterdir():
            call_id = Path(file.name).stem

            # file = Path('/www/files/call_transcripts/J2gyMMqPFjLccAyocqTj/c064e4ed-9fc8-42f2-a3e1-5d104717101f.json')

            message_list = json.loads(file.read_text(encoding="utf-8"))
            ut_query = None
            conversation = ""

            logging.info(f"Processing call_id: {call_id} for assistant: {assistant}")

            for message in message_list[1:]:
                # list_of_possible_answers = []
                # list_of_possible_intents = []
                # list_of_possible_questions = []
                possible_conv_paths = []

                conversation += f"{message['role']}: {message['text']}\n"

                if message["role"] == "USER":
                    # User prompt query
                    ut_query = message["text"]  # live transcription
                    if message["matching"]:
                        ut_query = message["matching"]["original"]  # offline transcription

                if (
                    message["matching"] is not None
                    and "distance" in message["matching"]
                    and message["matching"]["distance"] == 0.0
                ):
                    # A matching with distance = 0.0 is either an exact match or initial message matching or LLM matching

                    if "conv_path_id" in message["matching"]:
                        # It was the user prompt that was matched

                        # ignore initial message matching
                        if ut_query is None or "init" in ut_query.lower():
                            continue

                        # make sure that each conv_path_id is processed only once
                        conv_path_id = message["matching"]["conv_path_id"]
                        if conv_path_id in seen_conv_path_ids:
                            continue
                        seen_conv_path_ids.add(conv_path_id)

                        # ignore conv_path that is not in the DB
                        conv_path = ConversationalPaths.get_by_ids([conv_path_id])
                        if not conv_path:
                            continue

                        conv_path = conv_path[0]
                        source_node_id = conv_path.source_node_id
                        user_prompt_id = conv_path.user_prompt_id  # PRIMARY user prompt
                        assistant_answer_id = conv_path.assistant_answer_id
                        target_node_id = conv_path.target_node_id

                        # depth 1 matching
                        if not source_node_id:
                            continue

                        # depth 2 matching
                        up_pred = UserPrompts.get_by_ids([user_prompt_id])[0].text
                        aa_pred = (
                            AssistantAnswers.get_by_ids([assistant_answer_id])[0].text
                            if assistant_answer_id
                            else None
                        )
                        aq_pred = (
                            AssistantQuestions.get_by_ids([target_node_id])[0].text
                            if AssistantQuestions.get_by_ids([target_node_id])
                            else None
                        )  # follow up question is optional

                        # ignore exact matching
                        if check_normalized_text_matching(ut_query, user_prompt_id) is True:
                            continue

                        # Extract possible (UP, AA, AQ) from source_node
                        conv_paths_from_source_node = ConversationalPaths.query(
                            sm.select(ConversationalPaths).where(
                                ConversationalPaths.source_node_id == source_node_id
                            )
                        )
                        for conv_path in conv_paths_from_source_node:
                            up = UserPrompts.get_by_ids([conv_path.user_prompt_id])[0].text
                            aa = AssistantAnswers.get_by_ids([conv_path.assistant_answer_id])[
                                0
                            ].text
                            aq = (
                                AssistantQuestions.get_by_ids([conv_path.target_node_id])[0].text
                                if AssistantQuestions.get_by_ids([conv_path.target_node_id])
                                else None
                            )  # follow up question is optional
                            possible_conv_paths.append((up, aa, aq))

                        up_matchings.loc[len(up_matchings)] = {
                            "conversation": remove_last_assistant_messages(conversation),
                            "user_text": ut_query,
                            "possible_conv_paths": possible_conv_paths,
                            "match_pred": (up_pred, aa_pred, aq_pred),
                            "assistant": assistant,
                            "call_id": call_id,
                            "language": language,
                        }

                    # else:
                    #     # It was the assistant output that was matched

                    #     # ignore initial message matching
                    #     if message["id"] == "init":
                    #         continue

                    #     ao_query = message["matching"]["original"]  # Assistant output query

                    #     ao_match = None
                    #     if message["source"] == "DB_QUESTION":
                    #         ao_match = AssistantQuestions.get_by_ids([message["matching"]["id"]])
                    #     elif message["source"] == "DB_TEXT":
                    #         ao_match = AssistantTexts.get_by_ids([message["matching"]["id"]])
                    #     if ao_match:
                    #         ao_match = ao_match[0].text

                    #         # ignore exact matching
                    #         if is_normalized_text_matching(ao_query, ao_match):
                    #             continue

                    #         # for LLM matching, extract conversation, source_question, and list_of_possible_questions
                    #         if message["source"] == "DB_QUESTION":
                    #             list_of_possible_questions.extend(assistant_questions)
                    #         elif message["source"] == "DB_TEXT":
                    #             list_of_possible_questions.extend(assistant_texts)

                    #         ao_matchings.loc[len(ao_matchings)] = {
                    #             "conversation": conversation,
                    #             "source_question": ao_query,
                    #             "list_of_possible_questions": list_of_possible_questions,
                    #             "match": ao_match,
                    #             "assistant": assistant,
                    #             "call_id": call_id,
                    #             "source": message["source"],
                    #             "language": language,
                    #         }
                # conversation += f"{message['role']}: {message['text']}\n"

    return up_matchings, ao_matchings


def add_user_text_matching_to_df(df: pd.DataFrame, model: str = "gpt-4o"):
    """Add user text matching results as a new column to the DataFrame."""
    # convert string back to list because saving DataFrame to Excel converts lists to strings
    df["possible_conv_paths"] = df["possible_conv_paths"].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    match_true = []
    reasonings = []

    for _, row in df.iterrows():
        matched_user_prompt, reasoning = asyncio.run(
            user_text_matching(
                row["user_text"],
                row["possible_conv_paths"],
                row["conversation"],
                row["language"],
                model=model,
            )
        )
        match_true.append(matched_user_prompt)
        reasonings.append(reasoning)
    df["match_true"] = match_true
    df["reasoning"] = reasonings


if __name__ == "__main__":
    up_matchings, ao_matchings = extract_LLM_matchings(CALL_TRANSCRIPTS_PATH)
    # ao_matchings.to_excel(LLM_MATCHING_AO_PATH, index=False)
    # up_matchings.to_excel(LLM_MATCHING_UP_PATH, index=False)

    df = pd.read_excel(LLM_MATCHING_UP_PATH)
    # df = df.rename(columns={"user_prompt": "user_text"})  # Rename column

    df = df[df["language"] == "en"][:5]
    add_user_text_matching_to_df(df, model="gpt-4.1")
    print(df.head())

    # df = extract_embedding_matchings(CALL_TRANSCRIPTS_PATH)
    # df.to_excel(MATCHING_PATH, index=False)

    # df = pd.read_excel(MATCHING_PATH)

    # compute_matching_distance_by_language(df)
    # df.to_excel(MATCHING_DISTANCE_PATH, index=False)

    # df_es = filter_by_language(df, "es")
    # df_es.to_excel(MATCHING_ES_PATH, index=False)
    # df_es = pd.read_excel(MATCHING_ES_PATH)

    # embedding_models = [MONOINGUAL_EMBEDDING_MODELS_PER_LANGUAGE["es"][0]]
    # embedding_models = ["UAE-Large-V1"]
    # compute_matching_distance_with_multiple_embedding_models(df_es, embedding_models)
    # df_es.to_excel(MATCHING_DISTANCE_ES_PATH, index=False)
