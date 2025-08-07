import vocal.common.static  # noqa: F401

import sqlmodel as sm
import numpy as np
import json
import logging
from pydantic import BaseModel
from collections import defaultdict
from pathlib import Path

# Matching extraction
from vocal.common.utils import normalize_text
from vocal.chat_engine.utils import remove_guards
from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistants import Assistants
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions
from getvocal.datamodel.sql.assistant_answers import AssistantAnswers
from getvocal.datamodel.sql.conversational_paths import ConversationalPaths

# Embedding
from getvocal.datamodel.sql.assistants import DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE
from getvocal.multimodal.llms import chat_response

CALL_TRANSCRIPTS_DIR = "/www/files/call_transcripts"


LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
}

SELECT_USER_DB_CONTEXT = """
You are a helpful conversational assistant talking with a user over a phone call. 
The conversation is in {language}. 
Here's an ongoing conversation between an assistant and a user:

{conversation}

Here are some noise sources in transcripts:
- **Misrecognized words**: The system predicts the wrong word.
- **Homophones**: The system predicts a word that sounds like what the speaker actually said.
- **Background noise**: External sounds interfere with transcription.
- **Speaker overlap**: Another speaker talks alongside the main speaker.
- **Truncated words and phonetic spelling**: Words may be cut off or misspelled based on their sounds.
- **Hallucinations**: The transcription system may fill in inaccurate or non-existent information.

This is what the user/speaker has said: "{user_prompt}".
Your task is to determine the exact intent from the list below. **Only select an intent if the user’s message explicitly matches one of the described intents. Do NOT guess or choose the closest option if there is no clear match. If the user’s message does not directly fit any of the listed intents, return "none".**
To help you choose the best match, each intent is followed by an example assistant response. Remember, these examples are just for context — you must strictly match the meaning of the user’s message to the intent description.
Now here's the list of user/speaker intents:
{list_of_possible_answers}

In case there is no match, you must output `'none'`.

**Output format**

You should format your answer as a json,

```
{
  'reasoning': 'why you picked that particular answer',
  'output': 'index_of_selected_scenario'
}
```

If there is no match with what the speaker has said, then

```
{
  'reasoning': 'why you didn't choose any scenario',
  'output': 'none'
}
```
"""

# -----------


def get_user_prompt_id_from_source_node(source_node_id: str) -> list[str]:
    user_prompt_ids = []
    conv_paths = ConversationalPaths.query(
        sm.select(ConversationalPaths).where(ConversationalPaths.source_node_id == source_node_id)
    )
    for conv_path in conv_paths:
        user_prompt_ids.append(conv_path.user_prompt_id)

    return user_prompt_ids





def get_default_embedding_model(language: str) -> str:
    if language in DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE.keys():
        return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE[language]
    return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE["others"]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - cos_sim


async def user_text_matching(
    user_text: str,
    possible_conv_paths: list[(str, str, str)],
    conversation: str,
    language: str,
    model: str,
) -> tuple[tuple[str, str, str] | None, str]:
    """
    Match a user text to a set of possible conversational paths using an LLM.

    Returns:
        matched_conv_path: The selected conversational path tuple (up, aa, aq) or None if no match
        reasoning (str): The LLM's explanation for why it picked that particular answer or why no match was found
    """
    list_of_possible_answers = ""
    for i, (up, aa, aq) in enumerate(possible_conv_paths):
        up = remove_guards(up)  # remove guards from user prompt
        if aq is None:
            list_of_possible_answers += f"{i}. '{up}' and assistant responds with: '{aa}'\n"
        else:
            list_of_possible_answers += f"{i}. '{up}' and assistant responds with: '{aa} {aq}'\n"

    prompt = (
        SELECT_USER_DB_CONTEXT.replace("{language}", language)
        .replace("{conversation}", conversation)
        .replace("{user_prompt}", user_text)
        .replace("{list_of_possible_answers}", list_of_possible_answers)
    )
    logging.debug(f'user_text_matching system prompt: "{prompt}"')
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": "Following the previous guidelines, provide your output following the desired format.",
        },
    ]

    response = await chat_response(
        messages=messages, model=model, response_format={"type": "json_object"}, stream=False
    )

    try:
        response = response.output_text
        result = json.loads(response)
        matched_conv_path = (
            None if result["output"] == "none" else possible_conv_paths[int(result["output"])]
        )
        reasoning = result["reasoning"]
        return matched_conv_path, reasoning
    except Exception as e:
        logging.debug(f"Failed to decode the response: {response}. Error: {e}")
        return None, None
