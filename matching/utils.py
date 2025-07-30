from getvocal.datamodel.sql.conversational_paths import ConversationalPaths
import sqlmodel as sm
import numpy as np
import pandas as pd
import logging
from enum import Enum
from vocal.common.utils import normalize_text
from getvocal.datamodel.sql.assistants import Assistants
from getvocal.datamodel.sql.assistant_texts import AssistantTexts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions
from getvocal.datamodel.sql.user_prompts import UserPrompts
from vocal.chat_engine.utils import remove_guards

from getvocal.datamodel.sql.assistants import DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE


class Language(Enum):
    """Enum for supported languages."""

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"


SELECT_USER_DB_CONTEXT = """
You are a helpful conversational assistant talking with a user over a phone call. Here's an ongoing conversation between an assistant and a user:

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


def get_user_prompt_id_from_source_node(source_node_id: str) -> list[str]:
    user_prompt_ids = []
    conv_paths = ConversationalPaths.query(
        sm.select(ConversationalPaths).where(
            ConversationalPaths.source_node_id == source_node_id
        )
    )
    for conv_path in conv_paths:
        user_prompt_ids.append(conv_path.user_prompt_id)

    return user_prompt_ids


def get_assistant_language(assistant_id: str) -> str:
    assistant = Assistants.get_by_ids([assistant_id])
    return assistant[0].language


def get_default_embedding_model(language: str) -> str:
    if language in DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE.keys():
        return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE[language]
    return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE["others"]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - cos_sim


def get_texts_by_assistant(assistant_id: str) -> list[str]:
    assistant_texts = AssistantTexts.query(
        sm.select(AssistantTexts).where(AssistantTexts.assistant_id == assistant_id)
    )
    return [text.text for text in assistant_texts]


def get_questions_by_assistant(assistant_id: str) -> list[str]:
    assistant_questions = AssistantQuestions.query(
        sm.select(AssistantQuestions).where(
            AssistantQuestions.assistant_id == assistant_id
        )
    )
    return [question.text for question in assistant_questions]


def remove_last_assistant_message(conversation: str) -> str:
    """
    Remove the last assistant message from the conversation string.
    """
    if not conversation:
        return conversation

    # Split the conversation into lines
    lines = conversation.strip().split("\n")

    # If the last line is an assistant message, remove it
    if lines and lines[-1].startswith("ASSISTANT:"):
        return "\n".join(lines[:-1])

    return conversation


def check_normalized_text_matching(ut_query: str, user_prompt_id: str) -> bool:
    """
    Check exact matching between a user text and a user prompt by ID.
        - if user prompt is secondary, compare user prompt's text with user text
        - if user prompt is primary, check attached (secondary) user prompts and compare their texts with user text
    """
    try:
        user_prompt = UserPrompts.get_by_ids([user_prompt_id])[0]
        if user_prompt.primary_id:  # If the user prompt is secondary, get its text
            ut_match = user_prompt.text
            return normalize_text(ut_query) == normalize_text(ut_match)
        else:  # If the user prompt is primary, check for attached (secondary) user prompts and get their texts
            attached_user_prompt_ids = user_prompt.attached_user_prompt_ids
            if attached_user_prompt_ids:  # primary user prompt is likely to have empty list of attached user prompts
                for attached_user_prompt_id in attached_user_prompt_ids:
                    try:
                        attached_user_text = UserPrompts.get_by_ids(
                            [attached_user_prompt_id]
                        )[0].text
                        if normalize_text(ut_query) == normalize_text(
                            attached_user_text
                        ):
                            logging.info(
                                f"Skipping exact match for user prompt: {ut_query} with attached user prompt: {attached_user_text}"
                            )
                            return True
                    except IndexError as e:
                        logging.debug(
                            f"Error retrieving attached user prompt: {attached_user_prompt_id} for user prompt: {user_prompt_id}."
                        )
            return False
    except IndexError as e:
        logging.debug(f"Error retrieving user prompt: {user_prompt_id}.")
        return False


# def user_text_matching(
#     user_text: str,
#     possible_conv_paths: list[(str, str, str)],
#     conversation: str,
#     language: str,
# ) -> list[(str, str, str)]:
#     """
#     Match a user text to a set of possible conversational paths using an LLM.
#     """
#     list_of_possible_answers = ""
#     for i, (up, aa, aq) in enumerate(possible_conv_paths):
#         up = remove_guards(up)  # remove guards from user prompt
#         if aq is None:
#             list_of_possible_answers += (
#                 f"{i}. '{up}' and assistant responds with: '{aa}'\n"
#             )
#         else:
#             list_of_possible_answers += (
#                 f"{i}. '{up}' and assistant responds with: '{aa} {aq}'\n"
#             )

#     prompt = (
#         SELECT_USER_DB_CONTEXT.replace("{conversation}", conversation)
#         .replace("{user_prompt}", user_text)
#         .replace("{list_of_possible_answers}", list_of_possible_answers)
#     )
