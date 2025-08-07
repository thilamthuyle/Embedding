import json
import logging
from pathlib import Path
import concurrent.futures
import sqlmodel as sm
from sqlalchemy import select

import vocal.common.static  # noqa: E402,F401
from getvocal.datamodel.sql.calls import Calls
from getvocal.datamodel.sql.assistants import Assistants

from utils import CALL_TRANSCRIPTS_DIR


def process_call(
    assistant_id: str, num_calls: int | None = None, save_to_dir: str = CALL_TRANSCRIPTS_DIR
):
    """
    Save the last `num_calls` completed call transcripts for a given assistant to JSON files.
    If num_calls is None, save all completed calls.
    A completed call means the lead did take the call.
    Args:
        assistant: The assistant for which to save call transcripts.
        num_calls: The number of most recent completed calls to save.
        save_dir: Directory where the call transcripts will be saved.
    """
    logging.debug(f"Saving call transcripts for assistant {assistant_id}...")

    if num_calls is None:
        calls = Calls.query(select(Calls).where((Calls.assistant_id == assistant_id) & (Calls.status == "completed")))
    else:
        calls = Calls.query(
            select(Calls)
            .where((Calls.assistant_id == assistant_id) & (Calls.status == "completed"))
            .order_by(Calls.dt_started.desc())
            .limit(num_calls)
        )

    if calls == []:
        logging.debug(f"Found no completed calls for assistant {assistant_id}.")
        return

    output_dir = Path(f"{save_to_dir}/{assistant_id}")

    for call in calls:
        file_path = Path(f"{output_dir}/{call.id}.json")
        if file_path.exists():
            logging.debug(f"Skipping call transcript {call.id}: already processed.")
            continue
        if not call.conversation_transcript:
            logging.debug(f"Found no transcript for call {call.id}.")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(call.conversation_transcript, indent=2))
        logging.info(f"Saved transcript to {file_path}")


def save_call_transcripts_to_json(
    num_calls: int | None = None, save_to_dir: str = CALL_TRANSCRIPTS_DIR
):
    # Filter assistants to only those in production
    logging.info("Fetching all assistants in production...")
    all_assistants = Assistants.query(
        sm.select(Assistants.id).where(Assistants.settings["version"] == "prod")
    )

    arguments_list = []
    for assistant in all_assistants:
        arguments_list.append((assistant["id"], num_calls, save_to_dir))
    logging.info(f"Total call transcripts to save: {len(arguments_list)}.")

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_call, *args) for args in arguments_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing call: {e}")
                continue


if __name__ == "__main__":
    save_call_transcripts_to_json(2, CALL_TRANSCRIPTS_DIR)
    logging.info(f"Successfully saved call transcripts to directory: {CALL_TRANSCRIPTS_DIR}")
