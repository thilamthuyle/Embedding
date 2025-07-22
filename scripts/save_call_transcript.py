import json
from pathlib import Path

import sqlmodel as sm
from sqlmodel import create_engine
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from getvocal.datamodel.sql import _BaseModel

# Adjust the import path as needed based on your project structure
from getvocal.datamodel.sql.calls import Calls
from getvocal.datamodel.sql.assistants import Assistants

import vocal.common.static  # noqa: E402,F401
from vocal.common.settings import DatabaseSettings

CALL_TRANSCRIPTS_PATH = "/www/files/call_transcripts"  # Path to save call transcripts

print("Loading environment variables...")
print(f"Database settings: {DatabaseSettings()}")
url = (
    f"postgresql://{DatabaseSettings().GETVOCAL_SQL_DATABASE_USER}:"
    f"{DatabaseSettings().GETVOCAL_SQL_DATABASE_PASSWORD}@"
    f"{DatabaseSettings().GETVOCAL_SQL_DATABASE_HOST}:5432/"
    f"{DatabaseSettings().GETVOCAL_SQL_DATABASE_NAME}"
)

engine = create_engine(url, echo=False)
print(f"Engine created with URL: {url}")

session_maker = sessionmaker(bind=engine)
_BaseModel.initialize_session_makers(sync_session_maker=session_maker, async_session_maker=None)


def save_call_transcript_json(assistant: Assistants, num_calls: int, save_dir: str) -> None:
    calls = Calls.query(
        select(Calls)
        .where((Calls.assistant_id == assistant.id) & (Calls.status == "completed"))
        .order_by(Calls.dt_started.desc())
        .limit(num_calls)
    )

    print(f"Saving call transcripts for assistant {assistant.id}...")
    save_path = Path(save_dir) / f"{assistant.id}"
    save_path.mkdir(parents=True, exist_ok=True)
    for call in calls:
        if not call.conversation_transcript:
            print(f"Call {call.id} has no transcript.")
            continue
        file_path = save_path / f"{call.id}.json"
        file_path.write_text(json.dumps(call.conversation_transcript, indent=2))
        print(f"Saved transcript for call {call.id} to {file_path}")


if __name__ == "__main__":
    list_assistants = Assistants.query(sm.select(Assistants))

    for assistant in list_assistants:
        if assistant.settings.version != "prod":
            continue
        save_call_transcript_json(assistant, num_calls=2, save_dir=CALL_TRANSCRIPTS_PATH)
