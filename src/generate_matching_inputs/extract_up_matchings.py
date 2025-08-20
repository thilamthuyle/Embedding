"""
User Prompt Matching Extraction from Call Transcripts

This module processes call transcript files to extract user text and their corresponding conversational path candidates for matching.

Features:
- Parses call transcripts to identify user text and candidate conversational paths for matching.
- Saves extracted matchings in a structured JSON format for downstream analysis and modeling.

Usage:
    Run this script to generate matching datasets from raw call transcripts.
"""

import json
import logging
from pathlib import Path
import concurrent.futures
import time
import sys

import sqlmodel as sm
from getvocal.datamodel.sql.conversational_paths import ConversationalPaths
from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions
from getvocal.datamodel.sql.assistant_answers import AssistantAnswers

sys.path.insert(0, "/www/Embedding")
from src.generate_matching_inputs.utils import (
    get_assistant_language,
    CALL_TRANSCRIPTS_DIR,
    process_call_transcript,
    process_matching_json_file,
)


def extract_ut_to_conv_path(
    ut_to_conv_path_dir: str, call_transcripts_dir: str = CALL_TRANSCRIPTS_DIR
):
    """
    Walk through all call transcripts and extract user prompt matchings using multiprocessing.
    Saves the results into JSON files.
    """
    start = time.time()
    arguments_list = []

    logging.info("Preparing arguments list for processing call transcripts...")
    for assistant_dir in Path(call_transcripts_dir).iterdir():
        assistant_id = assistant_dir.name
        language = get_assistant_language(assistant_id)

        for call_transcript_path in assistant_dir.iterdir():
            call_id = Path(call_transcript_path.name).stem

            # Since the conversation.json is only created after processing all matchings in call transcript,
            # we check if it exists to know whether the transcript has been completely processed.
            # Note that a call transcript without conversation.json can also mean that it has no matchings.
            # We still reprocess it in this code (even if it was processed before).
            if Path(
                f"{ut_to_conv_path_dir}/{language}/{assistant_id}/{call_id}/conversation.json"
            ).exists():
                logging.debug(
                    f"File {call_transcript_path} has already been processed, skipping..."
                )
                continue

            arguments_list.append(
                (call_transcript_path, ut_to_conv_path_dir, language, assistant_id, call_id)
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
    
    logging.info(
        f"Done extracting matchings from call transcripts. Took {time.time() - start:.2f} seconds."
    )


def etract_up_to_examples(up_to_examples_dir: str, ut_to_conv_path_dir: str):
    start = time.time()
    logging.info("Preparing arguments list for generating up_to_examples...")
    arguments_list = []
    matching_json_files = [
        file for file in Path(ut_to_conv_path_dir).rglob("*.json") 
        if file.name != "conversation.json"
    ]
    for file in matching_json_files:
        candidates = json.loads(file.read_text(encoding="utf-8"))["candidates"]
        arguments_list.append((up_to_examples_dir, candidates))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_matching_json_file, *args) for args in arguments_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logging.error(f"Error in process_matching_json_file: {exc}")
    logging.info(f"Done processing {len(matching_json_files)} matching JSON files in {time.time() - start:.2f} seconds.")



def find_call_transcripts_with_different_consecutive_conv_path_ids(call_transcripts_dir: str = CALL_TRANSCRIPTS_DIR):
    seen_call_id = set()
    for assistant_dir in Path(call_transcripts_dir).iterdir():
        assistant_id = assistant_dir.name

        for call_transcript_path in assistant_dir.iterdir():
            call_id = Path(call_transcript_path.name).stem
            all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))

            # Find transcripts with 3 consecutive messages with conv_path_id
            flag = 0
            for message in all_messages:
                if message["matching"] is None or "conv_path_id" not in message["matching"]:
                    flag = 0
                    continue
                flag += 1
                if flag >= 3:
                    if call_id not in seen_call_id:
                        logging.info(f"Found call transcript {call_id} for assistant {assistant_id} with >=3 consecutive messages with conv_path_id {message['matching']['conv_path_id']}")

                        seen_call_id.add(call_id)
                    

def extract_prod_outputs(prod_outputs_dir: str, ut_to_conv_path_dir: str, call_transcripts_dir: str = CALL_TRANSCRIPTS_DIR):
    """
    Walk through all matching JSON files in ut_to_conv_path_dir and extract baseline matching output.
    The output is extracted from the first conv_path_id that appears after user_text_idx in the conversation 
    and whose source node matches the source node of all candidates in the matching JSON file.    
    Args:
        prod_outputs_dir: directory where the baseline matching outputs will be saved.
        ut_to_conv_path_dir: directory containing the matching JSON files to process.
        call_transcripts_dir: directory containing the original call transcript files.
    """
    logging.info("Extracting baseline matching outputs from call transcripts...")

    seen_ut_matchings = set()

    start = time.time()
    matching_json_files = [
        file for file in Path(ut_to_conv_path_dir).rglob("*.json") 
        if file.name != "conversation.json"
    ]
    
    for file in matching_json_files:
        language, assistant_id, call_id, matching_id = file.parts[-4], file.parts[-3], file.parts[-2], Path(file.name).stem

        file_text = json.loads(file.read_text(encoding="utf-8"))
        user_text_idx = file_text.get("user_text_idx")
        source_node = file_text.get("candidates")["conv_path_id"][0].split("_")[0]

        # find the first conv_path_id that appears after user_text_idx in the call transcript whose source node matches the source node of all candidates
        call_transcript_path = Path(f"{call_transcripts_dir}/{assistant_id}/{call_id}.json")
        all_messages = json.loads(call_transcript_path.read_text(encoding="utf-8"))
        for message in all_messages[user_text_idx+1:]:
            if message["matching"] is None or "conv_path_id" not in message["matching"]:
                continue
            if message["matching"]["conv_path_id"].startswith(source_node):
                output = {}
                output["conv_path_id"] = message["matching"]["conv_path_id"]
                conv_path = ConversationalPaths.query(sm.select(ConversationalPaths.user_prompt_id, ConversationalPaths.assistant_answer_id, ConversationalPaths.target_node_id).where(ConversationalPaths.id == message["matching"]["conv_path_id"])).first()
                output["up"] = UserPrompts.query(sm.select(UserPrompts.text).where(UserPrompts.id == conv_path.user_prompt_id)).first().text
                output["aa"] = AssistantAnswers.query(sm.select(AssistantAnswers.text).where(AssistantAnswers.id == conv_path.assistant_answer_id)).first().text
                output["aq"] = AssistantQuestions.query(sm.select(AssistantQuestions.text).where(AssistantQuestions.id == conv_path.target_node_id)).first().text
                break
            
        # check if a same user_text_idx is matched multiple times in different conv_path_ids
        ut_matching = (language, assistant_id, call_id, matching_id, user_text_idx)
        if ut_matching not in seen_ut_matchings:
            seen_ut_matchings.add(ut_matching)
            continue
        logging.debug(f"User text index: {ut_matching} has appeared in more than one matching.")

                
                
                
                # conversation_json_path = Path(
                #     f"{ut_to_conv_path_dir}/{language_dir}/{assistant_id}/{call_id}/conversation.json"
                # )
                # if not conversation_json_path.exists():
                #     logging.warning(f"Conversation JSON for {call_id} does not exist, skipping...")
                #     continue

    logging.info(f"Total matching JSON files to process: {len(matching_json_files)}. Took {time.time() - start:.2f} seconds.")    



def extract_user_text_idx_from_matchings(ut_to_conv_path_dir: str):
    """
    Skim through all .json files in ut_to_conv_path folder (excluding conversation.json)
    and extract the "user_text_idx" field from each file.
    """
    logging.info("Extracting user_text_idx from all matching JSON files...")
    
    user_text_indices = []
    matching_json_files = [
        file for file in Path(ut_to_conv_path_dir).rglob("*.json") 
        if file.name != "conversation.json"
    ]
    
    for file in matching_json_files:
        try:
            matching_data = json.loads(file.read_text(encoding="utf-8"))
            user_text_idx = matching_data.get("user_text_idx")
            if user_text_idx is not None:
                user_text_indices.append({
                    "file": str(file),
                    "user_text_idx": user_text_idx
                })
                logging.debug(f"Found user_text_idx: {user_text_idx} in {file}")
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")
    
    logging.info(f"Extracted user_text_idx from {len(user_text_indices)} files")
    return user_text_indices


if __name__ == "__main__":
    # extract_ut_to_conv_path(ut_to_conv_path_dir="/www/files/up_matching_dataset/inputs/ut_to_conv_path")


    # etract_up_to_examples(up_to_examples_dir="/www/files/up_matching_dataset/inputs/up_to_examples",
    #                          ut_to_conv_path_dir="/www/files/up_matching_dataset/inputs/ut_to_conv_path")


    # find_call_transcripts_with_different_consecutive_conv_path_ids(call_transcripts_dir=CALL_TRANSCRIPTS_DIR)
    
    # Extract user_text_idx from all matching files
    # user_text_indices = extract_user_text_idx_from_matchings(ut_to_conv_path_dir="/www/files/up_matching_dataset/inputs/ut_to_conv_path")
    # print(f"Found {len(user_text_indices)} files with user_text_idx")
    # extract_prod_outputs()

    process_call_transcript(call_transcript_path=Path("/www/files/call_transcripts/eyMe6oXKrytagjMyDkaC/300011c5-307a-4e2e-a98f-620ce3ef8567.json"),
                            ut_to_conv_path_dir=Path("/www/files/up_matching_dataset/inputs/ut_to_conv_path"),
                            language="nl",
                            assistant_id="eyMe6oXKrytagjMyDkaC",
                            call_id="300011c5-307a-4e2e-a98f-620ce3ef8567")


    # extract_prod_outputs(prod_outputs_dir="/www/files/up_matching_dataset/outputs/prod",
    #                      ut_to_conv_path_dir="/www/files/up_matching_dataset/inputs/ut_to_conv_path",
    #                      call_transcripts_dir=CALL_TRANSCRIPTS_DIR)
