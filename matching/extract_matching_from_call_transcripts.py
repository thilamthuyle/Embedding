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


from utils import (
    filter_messages_with_up_matching,
    get_assistant_language,
    check_normalized_text_matching,
    CALL_TRANSCRIPTS_DIR,
    get_depth2_conv_paths_by_ids_dict,
    get_conv_paths_from_source_nodes_dict,
    save_matching_to_json,
    extract_matching_candidates_from_source_node,
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

    all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))
    messages_with_up_matching, messages_with_up_matching_idx = filter_messages_with_up_matching(
        all_messages
    )
    depth2_conv_paths_by_ids_dict = get_depth2_conv_paths_by_ids_dict(messages_with_up_matching)
    conv_paths_from_source_nodes_dict = get_conv_paths_from_source_nodes_dict(
        list(depth2_conv_paths_by_ids_dict.values())
    )

    conversation = []
    seen_conv_path_ids = set()
    user_text_idx = -1
    matching_id = 0

    for idx, message in enumerate(all_messages):
        conversation.append({"role": message["role"], "text": message["text"]})

        # Make sure that
        conv_path_id = messages_with_up_matching[matching_id].matching.conv_path_id
        if (
            idx != messages_with_up_matching_idx[matching_id]     # the current message contains matching
            or conv_path_id not in depth2_conv_paths_by_ids_dict  # the matching is depth 2
            or conv_path_id in seen_conv_path_ids                 # the conv_path was not processed yet
        ):
            continue

        # Get user_text from the message before the matching message
        user_text_idx = idx - 1
        user_text = all_messages[user_text_idx].text  # live transcription
        try:
            user_text = all_messages[user_text_idx].matching.original  # offline transcription
        except AttributeError:
            pass

        # Skip exact matching
        conv_path = depth2_conv_paths_by_ids_dict.get(conv_path_id)
        if message.matching.distance == 0.0 and check_normalized_text_matching(
            user_text, conv_path.user_prompt_id
        ):
            continue

        # Extract possible conv_paths from source_node as matching candidates
        candidates = extract_matching_candidates_from_source_node(
            conv_path.source_node_id, conv_paths_from_source_nodes_dict
        )

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

        seen_conv_path_ids.add(conv_path_id)
        matching_id += 1

    # Save conversation into .json file only if there is at least one matching
    if matching_id > 0:
        Path(f"{output_dir}/transcript.json").write_text(json.dumps(conversation), encoding="utf-8")

    logging.info(
        f"Processed call transcript: {assistant_id} | {call_id} with {len(depth2_conv_paths_by_ids_dict)} matchings."
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
    all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))
    messages_with_up_matching, messages_with_up_matching_idx = filter_messages_with_up_matching(
        all_messages
    )
    depth2_conv_paths_by_ids_dict = get_depth2_conv_paths_by_ids_dict(messages_with_up_matching)
    conv_paths_from_source_nodes = get_conv_paths_from_source_nodes_dict(
        list(depth2_conv_paths_by_ids_dict.values())
    )
    process_call_transcript(
        call_transcript_path,
        output_dir=Path(
            "/www/files/matching_dataset/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
        ),
        language="en",
        assistant_id="d37mNMstUaZwSPqtXUIJ",
        call_id="ef7501dd-e530-4e70-82bd-6795b17cedc6",
    )
    print(source_node_to_cp)
