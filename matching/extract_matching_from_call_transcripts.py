from collections import defaultdict
import json
import logging

# import ast
# from typing import Callable
from pathlib import Path

# import asyncio
# import pandas as pd
import sqlmodel as sm
import concurrent.futures
from pydantic import BaseModel

import vocal.common.static  # noqa: F401
from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions
from getvocal.datamodel.sql.assistant_answers import AssistantAnswers
from getvocal.datamodel.sql.conversational_paths import ConversationalPaths

from utils import CALL_TRANSCRIPTS_DIR

from utils import (
    get_assistant_language,
    check_normalized_text_matching,
    get_default_embedding_model,
    cosine_distance,
    remove_last_assistant_messages,
    user_text_matching,
    LANGUAGES,
)


# from vocal.common.embedding_utils import EmbeddingFunction
# from vocal.common.embedding_utils import compute_embeddings_gpu

# Configure logging to show DEBUG level messages
# Force reconfiguration even if logging was already configured
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


# def process_conversation_file(
#     conversation_file: Path,
#     assistant_id: str,
#     language: str,
#     call_id: str
# ):
#     call_id = Path(conversation_file.name).stem
#     output_dir = Path(f"{save_to_path}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{call_id}")
#     conversation = ""
#     matching_id = 0
#     seen_conv_path_ids = set()
#     if Path(f"{output_dir}/{matching_id}.json").exists():
#         logging.debug(f"File {output_dir}/{matching_id}.json already exists, skipping...")
#         continue


# async def extract_ut_to_conv_path_matching(
#     save_to_path: str, call_transcripts_path: str = CALL_TRANSCRIPTS_PATH
# ):
#     """
#     Extract user text to conversational path matchings from call transcripts.
#     Saves the results to a specified directory.
#     """
#     all_arguments = []
#     for assistant_dir in Path(call_transcripts_path).iterdir():
#         assistant_id = assistant_dir.name
#         language = get_assistant_language(assistant_id)

#         for conversation_file in assistant_dir.iterdir():
#             conversation_id = Path(conversation_file.name).stem
#             output_dir = Path(f"{save_to_path}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{conversation_id}")
#             conversation = ""
#             matching_id = 0
#             seen_conv_path_ids = set()
#             if Path(f"{output_dir}/{matching_id}.json").exists():
#                 logging.debug(f"File {output_dir}/{matching_id}.json already exists, skipping...")
#                 continue
#             all_arguments.append((conversation_file, assistant_id, language, conversation_id))
#     # Process all files concurrently using asyncio
#     tasks = [process_conversation_file(*args) for args in all_arguments]
#     await asyncio.gather(*tasks)


class Matching(BaseModel):
    distance: float | None = None
    conv_path_id: str | None = None
    original: str | None = None


class Message(BaseModel):
    role: str
    text: str
    source: str
    matching: Matching | None = None


def get_conversational_paths(message_list: list) -> dict[str, ConversationalPaths]:
    """
    Extracts the matched conversational paths from the message list.
    - Matched conv_paths are identified by their conv_path_id in the message matching field.
    - Remove depth 1 matchings (conv_paths without source node).
    - Return:
        - cp_id_to_cp: conv_path_id -> ConversationalPaths dictionary
    """
    # Get from DB all_conv_paths with conv_path_id in message list 
    conv_path_ids = set()               # Make sure that each conv_path_id is processed only once
    for message in message_list[1:]:    # Skip the first message (usually the opener)
        message = Message(**message)
        if (
            message.role == "ASSISTANT"
            and message.matching
            and message.matching.distance
            and message.matching.conv_path_id
        ):
            conv_path_ids.add(message.matching.conv_path_id)
    all_conv_paths = ConversationalPaths.get_by_ids(conv_path_ids)
    
    # Remove depth 1 matching - conv_path_id without source_node
    for conv_path in all_conv_paths[:]: # iterate over a shallow copy
        if not conv_path.source_node_id:
            all_conv_paths.remove(conv_path)

    cp_id_to_cp = {cp.id: cp for cp in all_conv_paths}

    return cp_id_to_cp


def get_conversational_paths_by_source_node(
    all_conv_paths: list[ConversationalPaths],
) -> dict[str, list[ConversationalPaths]]:
    source_node_ids = {cp.source_node_id for cp in all_conv_paths if cp.source_node_id}
    conv_path_candidates = ConversationalPaths.query(
        sm.select(ConversationalPaths).where(
            ConversationalPaths.source_node_id.in_(source_node_ids)
        )
    )
    conv_paths_by_source_node = defaultdict(list)
    for cp in conv_path_candidates:
        conv_paths_by_source_node[cp.source_node_id].append(cp)

    return conv_paths_by_source_node


def process_call_transcript(
    call_transcript_path: Path,
    output_dir: Path,
    language: str,
    assistant_id: str,
    call_id: str,
    matching_id: int,
) -> None:
    """
    TODO: Replace conversation by index of last user message
    One possibility is cp_id_to_ut_idx: conv_path_id -> index of the last user message dictionary
    """
    message_list = json.loads(call_transcript_path.read_text(encoding="utf-8"))

    conversation = ""
    seen_conv_path_ids = set()
    conv_path_by_id = get_conversational_paths(message_list)
    conv_paths_by_source_node = get_conversational_paths_by_source_node(
        list(conv_path_by_id.values())
    )

    for message in message_list[1:]:  # Skip the first message (usually the opener)
        message = Message(**message)  # Convert dict to Message model
        conversation += f"{message.role}: {message.text}\n"

        # Get user_text query
        if message.role == "USER":
            user_text = message.text  # live transcription
            if message.matching:
                user_text = message.matching.original  # offline transcription
            ### TODO: add last_user_message_index here
            continue

        # Make sure that the message has an unseen user prompt matching
        if (
            not message.matching
            or not message.matching.distance
            or not message.matching.conv_path_id
            or message.matching.conv_path_id in seen_conv_path_ids
        ):
            continue

        conv_path_id = message.matching.conv_path_id
        seen_conv_path_ids.add(conv_path_id)
        conv_path = conv_path_by_id.get(conv_path_id)
        if not conv_path:
            continue

        ### TODO: Pre-filter conv_path with source_node_id=None to reduced memory + number of conv_path to consider
        ### TODO: Limit conv_path query to return only needed columns
        # # Skip depth 1 matching
        # if not conv_path.source_node_id:
        #     continue

        # Skip exact matching
        if message.matching.distance == 0.0 and check_normalized_text_matching(
            user_text, conv_path.user_prompt_id
        ):
            continue

        # Extract conv_paths from source_node as matching candidates
        conv_path_candidates = conv_paths_by_source_node.get(conv_path.source_node_id, [])
        candidates = {"up": [], "aa": [], "aq": [], "conv_path_id": []}
        ### TODO: optimize DB calls by building ID list then make request to DB only once
        for conv_path in conv_path_candidates:
            ### TODO: search how to limit return columns in SQLModel. For example only get 'text' column
            assistant_question = AssistantQuestions.get_by_ids([conv_path.target_node_id])
            assistant_question_text = assistant_question[0].text if assistant_question else None

            candidates["up"].append(UserPrompts.get_by_ids([conv_path.user_prompt_id])[0].text)
            candidates["aa"].append(
                AssistantAnswers.get_by_ids([conv_path.assistant_answer_id])[0].text
            )
            candidates["aq"].append(assistant_question_text)  # follow up question is optional
            candidates["conv_path_id"].append(conv_path.id)

        # Save matching into .json file
        matching = {
            "assistant_id": assistant_id,
            "call_id": call_id,
            "language": language,
            "conversation": remove_last_assistant_messages(
                conversation
            ),  # make sure that the last message is from USER
            "user_text": user_text,
            "candidates": candidates,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = Path(f"{output_dir}/{matching_id}.json")
        file_path.write_text(json.dumps(matching), encoding="utf-8")
        logging.info(f"Saved matching to {file_path}")

        matching_id += 1


def extract_ut_to_conv_path_matching(
    save_to_dir: str, call_transcripts_dir: str = CALL_TRANSCRIPTS_DIR
):
    """
    Extract user text to conversational path matchings from call transcripts.
    Saves the results to a specified directory using multiprocessing.
    """
    arguments_list = []
    for assistant_dir in Path(call_transcripts_dir).iterdir():
        assistant_id = assistant_dir.name
        language = get_assistant_language(assistant_id)

        for call_transcript_path in assistant_dir.iterdir():
            call_id = Path(call_transcript_path.name).stem
            output_dir = Path(
                f"{save_to_dir}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{call_id}"
            )
            matching_id = 0

            if Path(f"{output_dir}/{matching_id}.json").exists():
                logging.debug(f"File {output_dir}/{matching_id}.json already exists, skipping...")
                continue

            arguments_list.append(
                (call_transcript_path, output_dir, language, assistant_id, call_id, matching_id)
            )

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_call_transcript, *args) for args in arguments_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Error in process_call_transcript: {exc}")


def extract_up_to_examples_matching():
    pass


if __name__ == "__main__":
    extract_ut_to_conv_path_matching(save_to_dir="/www/files/")
