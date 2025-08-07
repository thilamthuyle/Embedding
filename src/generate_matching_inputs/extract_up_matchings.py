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
from utils import *


def process_call_transcript(
    call_transcript_path: Path,
    output_dir: Path,
    language: str,
    assistant_id: str,
    call_id: str,
) -> None:
    """
    Process a single call transcript to extract all user prompt matchings, then save them to JSON files.
    """
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
        if message["role"] == "USER":
            user_text_idx = idx

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

        # Get user_text from the last USER message before the current ASSISTANT message with matching
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
            logging.debug(
                f"Remove matching with §NO_NEED§ user prompt candidate in {call_transcript_path}"
            )
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
        Path(f"{output_dir}/conversation.json").write_text(
            json.dumps(conversation), encoding="utf-8"
        )

    logging.info(
        f"Processed call transcript: {assistant_id} / {call_id} with {matching_id} matchings."
    )


def extract_up_matchings_from_call_transcripts(
    save_to_dir: str, call_transcripts_dir: str = CALL_TRANSCRIPTS_DIR
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
            output_dir = Path(
                f"{save_to_dir}/matching_dataset/inputs/ut_to_conv_path/{language}/{assistant_id}/{call_id}"
            )

            # Since the conversation.json is only created after processing all matchings in call transcript,
            # we check if it exists to know whether the transcript has already been processed.
            # Note that a call transcript without conversation.json can also mean that it has no matchings.
            # We still reprocess it in this code (even if it was processed before).
            if Path(f"{output_dir}/conversation.json").exists():
                logging.debug(
                    f"File {call_transcript_path} has already been processed, skipping..."
                )
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


if __name__ == "__main__":
    # start = time.time()
    # extract_up_matchings_from_call_transcripts(save_to_dir="/www/files/")
    # logging.info(f"Done extracting matchings from call transcripts. Took {time.time() - start:.2f} seconds.")

    call_transcript_path = Path("/www/files/call_transcripts/nRvPgRrKr2MuO7bjYCRY/255edbe5-cd84-4969-89a0-269a31437e75.json")
    output_dir = Path("/www/files/testttttt")
    language  = "en"
    assistant_id = "nRvPgRrKr2MuO7bjYCRY"
    call_id = "255edbe5-cd84-4969-89a0-269a31437e75"
    process_call_transcript(call_transcript_path, output_dir, language, assistant_id, call_id)
