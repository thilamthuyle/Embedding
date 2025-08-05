import json
import logging

# import ast
# from typing import Callable
from pathlib import Path

# import asyncio
# import pandas as pd
import sqlmodel as sm
import concurrent.futures
import time


import vocal.common.static  # noqa: F401
from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions
from getvocal.datamodel.sql.assistant_answers import AssistantAnswers


from utils import (
    filter_message_list,
    get_assistant_language,
    check_normalized_text_matching,
    CALL_TRANSCRIPTS_DIR,
    get_conv_paths_by_ids,
    get_conv_paths_by_source_node,
    Message,
    save_matching_to_json,
)
# get_default_embedding_model,
# cosine_distance,
# user_text_matching,


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


def process_call_transcript(
    call_transcript_path: Path,
    output_dir: Path,
    language: str,
    assistant_id: str,
    call_id: str,
) -> None:
    # logging.info(f"Processing call transcript: {call_transcript_path}")

    message_list = json.loads(call_transcript_path.read_text(encoding="utf-8"))
    valid_messages = filter_message_list(message_list)
    cp_id_to_cp = get_conv_paths_by_ids(valid_messages)
    source_node_to_cp = get_conv_paths_by_source_node(list(cp_id_to_cp.values()))

    conversation = []
    seen_conv_path_ids = set()
    user_text_idx = -1
    matching_id = 0

    for idx, message in enumerate(valid_messages):
        conversation.append({"role": message.role, "text": message.text})
        # Get user_text query
        if message.role == "USER":
            user_text = message.text  # live transcription
            if message.matching:
                user_text = message.matching.original  # offline transcription
                user_text_idx = idx
            continue

        # Make sure that message contains an unseen depth 2 matching 
        if (
            message.matching.conv_path_id not in cp_id_to_cp
            or message.matching.conv_path_id in seen_conv_path_ids
        ):
            continue

        conv_path_id = message.matching.conv_path_id
        seen_conv_path_ids.add(conv_path_id)
        conv_path = cp_id_to_cp.get(conv_path_id)
        if not conv_path:
            continue

        # Skip exact matching
        if message.matching.distance == 0.0 and check_normalized_text_matching(
            user_text, conv_path.user_prompt_id
        ):
            continue

        # Extract conv_paths from source_node as matching candidates
        conv_path_candidates = source_node_to_cp[conv_path.source_node_id]
        candidates = {"up": [], "aa": [], "aq": [], "conv_path_id": []}

        all_up_ids = [], all_aa_ids = [], all_aq_ids = []
        for conv_path in conv_path_candidates:
            all_up_ids.append(conv_path.user_prompt_id)
            all_aa_ids.append(conv_path.assistant_answer_id)
            all_aq_ids.append(conv_path.target_node_id)
            candidates["conv_path_id"].append(conv_path.id)

        ### TODO: search how to limit return columns in SQLModel. For example only get 'text' column
        all_ups = UserPrompts.get_by_ids(all_up_ids)
        all_aas = AssistantAnswers.get_by_ids(all_aa_ids)
        all_aqs = AssistantQuestions.get_by_ids(all_aq_ids)

        candidates["up"] = [up.text for up in all_ups]
        candidates["aa"] = [aa.text for aa in all_aas]
        candidates["aq"] = [
            aq.text if aq else None for aq in all_aqs
        ]  # follow up question is optional

        # Save matching into .json file
        save_matching_to_json(
            output_dir,
            language,
            assistant_id,
            call_id,
            matching_id,
            user_text,
            user_text_idx,
            candidates,
        )

        matching_id += 1

    # Save conversation into .json file
    if matching_id > 0:
        Path(f"{output_dir}/transcript.json").write_text(json.dumps(conversation), encoding="utf-8")

    logging.info(
        f"Processed call transcript: {assistant_id} | {call_id} with {len(cp_id_to_cp)} matchings."
    )


def extract_ut_to_conv_path_matching(
    save_to_dir: str, call_transcripts_dir: str = CALL_TRANSCRIPTS_DIR
):
    """
    Extract user text to conversational path matchings from call transcripts.
    Saves the results to a specified directory using multiprocessing.
    """
    start = time.time()
    arguments_list = []

    logging.info("Preparing arguments list for processing call transcripts...")
    for assistant_dir in Path(call_transcripts_dir).iterdir():
        assistant_id = assistant_dir.name
        language = get_assistant_language(assistant_id)

        for call_transcript_path in assistant_dir.iterdir():
            call_id = Path(call_transcript_path.name).stem
            output_dir = Path(
                f"{save_to_dir}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{call_id}"
            )

            if Path(f"{output_dir}/transcript.json").exists():
                logging.info(f"File {call_transcript_path} has already been processed, skipping...")
                continue

            arguments_list.append(
                (call_transcript_path, output_dir, language, assistant_id, call_id)
            )
    logging.info(
        f"Total call transcripts to process: {len(arguments_list)}. Arguments list prepared in {time.time() - start:.2f} seconds."
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
    # extract_ut_to_conv_path_matching(save_to_dir="/www/files/")

    # call_transcript_path = Path(
    #     "/www/files/call_transcripts/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6.json"
    # )
    # output_dir = Path(
    #     "/www/files/matching_dataset/inputs/ut_to_conv_path/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
    # )
    # language = "en"
    # assistant_id = "d37mNMstUaZwSPqtXUIJ"
    # call_id = "ef7501dd-e530-4e70-82bd-6795b17cedc6"
    # process_call_transcript(call_transcript_path, output_dir, language, assistant_id, call_id)

    call_transcript_path = Path(
        "/www/files/call_transcripts/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6.json"
    )
    message_list = json.loads(call_transcript_path.read_text(encoding="utf-8"))
    valid_messages = filter_message_list(message_list)
    cp_id_to_cp = get_conv_paths_by_ids(valid_messages)
    source_node_to_cp = get_conv_paths_by_source_node(list(cp_id_to_cp.values()))
    print(source_node_to_cp)
