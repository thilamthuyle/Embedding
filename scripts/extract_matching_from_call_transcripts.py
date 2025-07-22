import json
from typing import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from getvocal.datamodel.sql.assistants import DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE
from getvocal.datamodel.sql.assistants import Assistants
from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistant_texts import AssistantTexts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions

import vocal.common.static  # noqa: F401
from vocal.common.embedding_utils import EmbeddingFunction
from vocal.common.embedding_utils import compute_embeddings_gpu
from vocal.common.utils import normalize_text

CALL_TRANSCRIPTS_PATH = "/www/files/call_transcripts"  # Path to saved call transcripts
MATCHING_PATH = "/www/files/matching.xlsx"  # Path to save matchings extracted from call transcripts
MATCHING_ES_PATH = "/www/files/matching_es.xlsx"  # Spanigh matchings
MATCHING_DISTANCE_ES_PATH = "/www/files/matching_es.xlsx"  # Spanigh matchings
MATCHING_DISTANCE_PATH = "/www/files/matching_distance.xlsx"  # Matchings with distance between query & match embeddings


def get_assistant_language(assistant_id: str) -> str:
    assistant = Assistants.get_by_ids([assistant_id])
    return assistant[0].language


def is_message_matched(message: dict, ut_query: str) -> bool:  
    """
    If message is from USER, there's no matching, so return False.
    If message is from ASSISTANT:
        - If a matching was performed: 
            - if distance > 0.0: a (not exact) match is found. Return True.
            - if distance == 0.0: it can be either the initial message, or exact matching, or LLM matching.
                - if it's the initial message, return False.
                - otherwise, return True.
    """
    if message["matching"] is None:                 # USER message, there's no matching
        return False
    elif "distance" not in message["matching"]:     # ASSISTANT with no matching found 
        return False
    elif message["matching"]["distance"] == 0.0:    # ASSISTANT message with distance=0.0
        if message["id"] == "init":                 # the call starts with ASSISTANT message(s)
            return False
        elif "init" in ut_query.lower():            # the call starts with USER message(s) and message is the first ASSISTANT message in the call that performs matching on the initial user prompt
            return False
    return True


def is_normalized_text_matching(query: str, match: str) -> bool:
    """
    Check if the query and match are the same using normalized text comparison.
    """
    normalized_query = normalize_text(query)
    normalized_match = normalize_text(match)
    return normalized_query == normalized_match


def extract_matchings(call_transcripts_path: str = CALL_TRANSCRIPTS_PATH) -> pd.DataFrame:
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
                    
                if is_message_matched(message, ut_query) is False:
                    continue
                if message["matching"]["distance"] == 0.0:
                    continue

                if "conv_path_id" in message["matching"]:
                    # It was the user prompt that was matched
                    if message["matching"]["conv_path_id"] not in seen_conv_path_ids:
                        # make sure that each conv_path_id is processed only once
                        seen_conv_path_ids.add(message["matching"]["conv_path_id"])

                        up_id = message["matching"]["user_prompt_id"]
                        up_match = UserPrompts.get_by_ids([up_id])[0].text
                        if not is_normalized_text_matching(ut_query, up_match):
                            df.loc[len(df)] = {
                                "assistant": assistant,
                                "call_id": call_id,
                                "query": ut_query,
                                "match": up_match,
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
                        if not is_normalized_text_matching(ao_query, ao_match):
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


def filter_matchings_by_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Filter the DataFrame to only include rows where the assistant's language matches the specified language.
    """
    return df[df["language"] == language]


def get_default_embedding_model(language: str) -> str:
    if language in DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE.keys():
        return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE[language]
    return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE["others"]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - cos_sim


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


if __name__ == "__main__":
    df = extract_matchings(CALL_TRANSCRIPTS_PATH)
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


