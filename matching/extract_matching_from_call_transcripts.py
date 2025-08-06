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
    Message,
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
    logging.debug(f"Start processing call transcript: {assistant_id} / {call_id}") 

    all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))
    messages_with_up_matching, messages_with_up_matching_idx = filter_messages_with_up_matching(
        all_messages
    )
    depth2_conv_paths_by_ids_dict = get_depth2_conv_paths_by_ids_dict(messages_with_up_matching)
    depth2_conv_path_ids = depth2_conv_paths_by_ids_dict.keys()
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
        try:
            message = Message(**message)
            conv_path_id = message.matching.conv_path_id
            if (
                idx not in messages_with_up_matching_idx  # the current message contains matching
                or conv_path_id not in depth2_conv_path_ids  # matching is depth 2
                or conv_path_id in seen_conv_path_ids  # and conv_path_id was not processed yet
            ):
                continue
        except Exception:
            continue

        # Get user_text from the message before the matching message
        user_text_idx = idx - 1
        user_text = all_messages[user_text_idx]["text"]  # live transcription
        try:
            user_text = all_messages[user_text_idx]["matching"]["original"]  # offline transcription
        except KeyError:
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

        # Remove matchings with §NO_NEED§ in user prompt candidates
        if not candidates:
            logging.debug("Remove matching with §NO_NEED§ in user prompt candidates")
            continue
        
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

    # Save conversation only if there is at least one matching
    if matching_id > 0:
        Path(f"{output_dir}/conversation.json").write_text(json.dumps(conversation), encoding="utf-8")

    logging.info(
        f"Processed call transcript: {assistant_id} / {call_id} with {matching_id} matchings."
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

            if Path(f"{output_dir}/conversation.json").exists():
                logging.debug(f"File {call_transcript_path} has already been processed, skipping...")
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
    extract_ut_to_conv_path_matching(save_to_dir="/www/files/")

    # call_transcript_path = Path("/www/files/call_transcripts/bzzd3IuANPMYYsitiBlo/12c8a927-94be-48fe-9fe0-c626f38584a5.json")
    # output_dir = Path("/www/files/matching_dataset/inputs/ut_to_conv_path/test/bzzd3IuANPMYYsitiBlo/12c8a927-94be-48fe-9fe0-c626f38584a5")
    # process_call_transcript(call_transcript_path, output_dir, "test", "bzzd3IuANPMYYsitiBlo", "12c8a927-94be-48fe-9fe0-c626f38584a5")

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

    ######
    # from getvocal.datamodel.sql.conversational_paths import ConversationalPaths
    # from utils import ConvPath

    # source_node_id = "6f9f88344d8d9ed41aa94da2644e121c"
    # conv_paths_from_source_nodes_dict = get_conv_paths_from_source_nodes_dict(
    #     [ConversationalPaths.get("6f9f88344d8d9ed41aa94da2644e121c_a9226fc4c9693896251ff045797b9d27_76fb41c902d6f8cb30f7a6bbcb7fc20c_bc05ab88b87e67c06cadf44895c07c1f")]
    # )
    # candidates = extract_matching_candidates_from_source_node(source_node_id, conv_paths_from_source_nodes_dict)
    # save_matching_to_json(output_dir=Path(
    #         "/www/files/matching_dataset/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
    #     ),
    #     language="en",
    #     assistant_id="d37mNMstUaZwSPqtXUIJ",
    #     call_id="ef7501dd-e530-4e70-82bd-6795b17cedc6",
    #     matching_id=0, 
    #     user_text="What is the weather like today?",
    #     user_text_idx=0,  # Assuming the first message is the user text
    #     candidates=candidates
    # )


    ######
    # call_transcript_path = Path(
    #     "/www/files/call_transcripts/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6.json"
    # )
    # all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))
    # messages_with_up_matching, messages_with_up_matching_idx = filter_messages_with_up_matching(
    #     all_messages
    # )
    # depth2_conv_paths_by_ids_dict = get_depth2_conv_paths_by_ids_dict(messages_with_up_matching)
    # conv_paths_from_source_nodes = get_conv_paths_from_source_nodes_dict(
    #     list(depth2_conv_paths_by_ids_dict.values())
    # )
    # process_call_transcript(
    #     call_transcript_path,
    #     output_dir=Path(
    #         "/www/files/matching_dataset/en/d37mNMstUaZwSPqtXUIJ/ef7501dd-e530-4e70-82bd-6795b17cedc6"
    #     ),
    #     language="en",
    #     assistant_id="d37mNMstUaZwSPqtXUIJ",
    #     call_id="ef7501dd-e530-4e70-82bd-6795b17cedc6",
    # )
    # print(conv_paths_from_source_nodes)
