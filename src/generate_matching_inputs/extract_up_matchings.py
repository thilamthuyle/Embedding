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




if __name__ == "__main__":
    extract_ut_to_conv_path(ut_to_conv_path_dir="/www/files/up_matching_dataset/inputs/ut_to_conv_path")


    etract_up_to_examples(up_to_examples_dir="/www/files/up_matching_dataset/inputs/up_to_examples",
                             ut_to_conv_path_dir="/www/files/up_matching_dataset/inputs/ut_to_conv_path")


