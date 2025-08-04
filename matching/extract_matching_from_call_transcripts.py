import json
import logging
import ast
from typing import Callable
from pathlib import Path
import asyncio
import pandas as pd
import sqlmodel as sm

import vocal.common.static  # noqa: F401
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
    LANGUAGES,
)


import vocal.common.static  # noqa: F401
from vocal.common.embedding_utils import EmbeddingFunction
# from vocal.common.embedding_utils import compute_embeddings_gpu

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
UT_MATCHING_LLM_PATH = "/www/files/ut_matching_llm.xlsx"


def process_conversation_file(
    conversation_file: Path,
    assistant_id: str,
    language: str,
    call_id: str
):
    call_id = Path(conversation_file.name).stem
    output_dir = Path(f"{save_to_path}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{call_id}") 
    conversation = ""
    matching_id = 0
    seen_conv_path_ids = set()
    if Path(f"{output_dir}/{matching_id}.json").exists():
        logging.debug(f"File {output_dir}/{matching_id}.json already exists, skipping...")
        continue


async def extract_ut_to_conv_path_matching(
    save_to_path: str, call_transcripts_path: str = CALL_TRANSCRIPTS_PATH
):
    """
    Extract user text to conversational path matchings from call transcripts.
    Saves the results to a specified directory.
    """
    all_arguments = []
    for assistant_dir in Path(call_transcripts_path).iterdir():
        assistant_id = assistant_dir.name
        language = get_assistant_language(assistant_id)

        for conversation_file in assistant_dir.iterdir():
            conversation_id = Path(conversation_file.name).stem
            output_dir = Path(f"{save_to_path}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{conversation_id}") 
            conversation = ""
            matching_id = 0
            seen_conv_path_ids = set()
            if Path(f"{output_dir}/{matching_id}.json").exists():
                logging.debug(f"File {output_dir}/{matching_id}.json already exists, skipping...")
                continue
            all_arguments.append((conversation_file, assistant_id, language, conversation_id))
    # Process all files concurrently using asyncio
    tasks = [process_conversation_file(*args) for args in all_arguments]
    await asyncio.gather(*tasks)


async def extract_ut_to_conv_path_matching(
    save_to_path: str, call_transcripts_path: str = CALL_TRANSCRIPTS_PATH
):
    """
    Extract user text to conversational path matchings from call transcripts.
    Saves the results to a specified directory.
    """
    for assistant_dir in Path(call_transcripts_path).iterdir():
        assistant_id = assistant_dir.name
        language = get_assistant_language(assistant_id)

        for conversation_file in assistant_dir.iterdir():
            call_id = Path(conversation_file.name).stem
            output_dir = Path(f"{save_to_path}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{call_id}") 
            conversation = ""
            matching_id = 0
            seen_conv_path_ids = set()
            if Path(f"{output_dir}/{matching_id}.json").exists():
                logging.debug(f"File {output_dir}/{matching_id}.json already exists, skipping...")
                continue

            message_list = json.loads(conversation_file.read_text(encoding="utf-8"))
            for message in message_list[1:]:  # Skip the first message (usually the opener)
                conversation += f"{message['role']}: {message['text']}\n"

                # Get user_text query
                if message["role"] == "USER":
                    user_text = message["text"]  # live transcription
                    if message["matching"]:
                        user_text = message["matching"]["original"]  # offline transcription
                    continue

                # Make sure that the message has an unseen user prompt matching
                if (
                    not message.get("matching", None)
                    or "distance" not in message["matching"]
                    or "conv_path_id" not in message["matching"]
                    or message["matching"]["conv_path_id"] in seen_conv_path_ids
                ):
                    continue

                conv_path_id = message["matching"]["conv_path_id"]
                seen_conv_path_ids.add(conv_path_id)
                conv_path = await ConversationalPaths.aget(conv_path_id)
                if not conv_path:
                    continue
                conv_path = conv_path[0]

                # Skip depth 1 matching
                if not conv_path.source_node_id:
                    continue

                # Skip exact matching
                if message["matching"]["distance"] == 0.0 and check_normalized_text_matching(
                    user_text, conv_path.user_prompt_id
                ):
                    continue

                # Extract conv_paths from source_node as matching candidates
                conv_path_candidates = await ConversationalPaths.aquery(
                    sm.select(ConversationalPaths).where(
                        ConversationalPaths.source_node_id == conv_path.source_node_id
                    )
                )
                candidates = {"up": [], "aa": [], "aq": [], "conv_path_id": []}
                for conv_path in conv_path_candidates:
                    candidates["up"].append(await UserPrompts.aget_by_ids([conv_path.user_prompt_id])[0].text)
                    candidates["aa"].append(await AssistantAnswers.aget_by_ids([conv_path.assistant_answer_id])[0].text)
                    candidates["aq"].append(
                        await AssistantQuestions.get_by_ids([conv_path.target_node_id])[0].text
                        if AssistantQuestions.get_by_ids([conv_path.target_node_id])
                        else None
                    )  # follow up question is optional
                    candidates["conv_path_id"].append(conv_path.id)

                # Save matching into .json file
                matching = {
                    "assistant_id": assistant_id,
                    "call_id": call_id,
                    "language": language,
                    "conversation": remove_last_assistant_messages(conversation), # make sure that the last message is from USER
                    "user_text": user_text,
                    "candidates": candidates,
                }
                output_dir.mkdir(parents=True, exist_ok=True)
                file_path = Path(f"{output_dir}/{matching_id}.json")
                file_path.write_text(json.dumps(matching), encoding='utf-8')
                logging.info(f"Saved matching to {file_path}")
                
                matching_id += 1


def extract_up_to_examples_matching():
    pass


if __name__ == "__main__":
    extract_ut_to_conv_path_matching(save_to_path="/www/files/")

