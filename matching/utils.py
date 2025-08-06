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


class ConvPath(BaseModel):
    id: str
    source_node_id: str
    user_prompt_id: str
    assistant_answer_id: str
    target_node_id: str | None = None

    @classmethod
    def columns(cls):
        return [sm.column(c) for c in cls.model_fields.keys()]


class Matching(BaseModel):
    distance: float
    conv_path_id: str
    original: str | None = None


class Message(BaseModel):
    role: str
    text: str
    matching: Matching


def filter_messages_with_up_matching(message_list: list[dict]) -> tuple[list[Message], list[int]]:
    """
    Filter messages from the message list that have a matching field containing both conv_path_id
    and distance.
    NB: Note that one conv_path_id can appear in multiple messages.
    Return:
        messages_with_up_matching: list of valid Message objects
        message_idx: list of indices of valid Message objects in the original conversation before filtering
    """
    messages_with_up_matching = []
    messages_with_up_matching_idx = []  # Stores the original indices of messages in the conversation before filtering
    for i, message in enumerate(message_list):
        try:
            messages_with_up_matching.append(Message(**message))
            messages_with_up_matching_idx.append(i)
        except Exception:
            pass  # Ignore messages that cannot be parsed

    return messages_with_up_matching, messages_with_up_matching_idx


def get_depth2_conv_paths_by_ids_dict(
    messages_with_up_matching: list[Message],
) -> dict[str, ConvPath]:
    """
    Filter messages from the message list that have depth 2 matching conv_path.
    Extract the matched conversational paths from conv_path_id in filtered messages.
    Return:
        depth2_conv_paths_by_ids_dict: conv_path_id -> conv_path dictionary
    """
    # Get all conv_path_ids in message list
    conv_path_ids = set()  # Make sure that each conv_path_id is processed only once
    for message in messages_with_up_matching:
        conv_path_ids.add(message.matching.conv_path_id)

    # Query DB and select only depth 2 conv_paths, meaning those with source node.
    # Init conv_path and depth 1 conv_path are then excluded.
    all_conv_paths = ConversationalPaths.query(
        sm.select(*ConvPath.columns()).where(
            ConversationalPaths.id.in_(conv_path_ids),
            ConversationalPaths.source_node_id.is_not(None),
        )
    )
    conv_paths = [ConvPath(**cp) for cp in all_conv_paths]
    depth2_conv_paths_by_ids_dict = {cp.id: cp for cp in conv_paths}

    return depth2_conv_paths_by_ids_dict


def get_conv_paths_from_source_nodes_dict(
    conv_paths: list[ConvPath],
) -> dict[str, list[ConvPath]]:
    """
    Extract all possibile conv_paths from the source nodes of given conv_paths.
    Given that all_conv_paths have source node.
    Return:
        conv_paths_from_source_nodes_dict: source_node_id -> list of conv paths from source node dictionary
    """
    source_node_ids = {
        cp.source_node_id for cp in conv_paths
    }  # Make sure that each source_node_id is processed only once
    all_conv_paths_from_source_nodes = ConversationalPaths.query(
        sm.select(*ConvPath.columns()).where(
            ConversationalPaths.source_node_id.in_(source_node_ids)
        )
    )
    conv_paths_from_source_nodes_dict = defaultdict(list)
    for cp in all_conv_paths_from_source_nodes:
        cp = ConvPath(**cp)
        conv_paths_from_source_nodes_dict[cp.source_node_id].append(cp)

    return conv_paths_from_source_nodes_dict


def extract_matching_candidates_from_source_node(
    source_node_id: str, conv_paths_from_source_nodes_dict: dict
) -> dict[str, list[str]]:
    """
    Extract all possibile conv_paths from the source node.
    If any of the user prompts contains §NO_NEED*§, return None.
    Return:
        candidates: dictionary with keys "up", "aa", "aq", "conv_path_id" and values as lists of texts or IDs
    """
    conv_paths_from_source_node = conv_paths_from_source_nodes_dict[source_node_id]
    num_conv_paths = len(conv_paths_from_source_node)

    candidates = {"up": [], "aa": [], "aq": [], "conv_path_id": []}
    up_ids, aa_ids, aq_ids = [], [], []

    for conv_path in conv_paths_from_source_node:
        up_ids.append(conv_path.user_prompt_id)
        aa_ids.append(conv_path.assistant_answer_id)
        aq_ids.append(conv_path.target_node_id)

    # Note that not all ids may be present in the DB, so the number of retrieved texts may be less than num_conv_paths
    existing_ups = UserPrompts.query(
        sm.select(UserPrompts.id, UserPrompts.text).where(UserPrompts.id.in_(up_ids))
    )
    existing_aas = AssistantAnswers.query(
        sm.select(AssistantAnswers.id, AssistantAnswers.text).where(AssistantAnswers.id.in_(aa_ids))
    )
    existing_aqs = AssistantQuestions.query(
        sm.select(AssistantQuestions.id, AssistantQuestions.text).where(
            AssistantQuestions.id.in_(aq_ids)
        )
    )

    existing_ups_dict = {up["id"]: up["text"] for up in existing_ups}
    existing_aas_dict = {aa["id"]: aa["text"] for aa in existing_aas}
    existing_aqs_dict = {aq["id"]: aq["text"] for aq in existing_aqs}

    # Remove all matchings with §NO_NEED*§ in up candidates
    if any("NO_NEED" in up for up in existing_ups_dict.values()):
        return None

    for i in range(num_conv_paths):
        if (
            up_ids[i] in existing_ups_dict.keys()
            and aa_ids[i] in existing_aas_dict.keys()
            and (not aq_ids[i] or aq_ids[i] in existing_aqs_dict.keys()) # follow up question is optional
        ):
            candidates["up"].append(existing_ups_dict[up_ids[i]])
            candidates["aa"].append(existing_aas_dict[aa_ids[i]])
            candidates["aq"].append(existing_aqs_dict[aq_ids[i]] if aq_ids[i] else None) 
            candidates["conv_path_id"].append(conv_paths_from_source_node[i].id)

    return candidates


def save_matching_to_json(
    output_dir: Path,
    language: str,
    assistant_id: str,
    call_id: str,
    matching_id: int,
    user_text: str,
    user_text_idx: int,
    candidates: dict[str, list[str]],
):
    matching = {
        "assistant_id": assistant_id,
        "call_id": call_id,
        "language": language,
        "user_text": user_text,
        "user_text_idx": user_text_idx,
        "candidates": candidates,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = Path(f"{output_dir}/{matching_id}.json")
    file_path.write_text(json.dumps(matching), encoding="utf-8")
    logging.debug(f"Saved matching to {file_path}")


def check_normalized_text_matching(ut_query: str, user_prompt_id: str) -> bool:
    """
    Check exact matching between a user text and a user prompt by ID.
        - if user prompt is secondary, compare user prompt's text with user text
        - if user prompt is primary, check attached (secondary) user prompts and compare their texts with user text
    """
    try:
        user_prompt = UserPrompts.get(user_prompt_id)
        if user_prompt.primary_id:  # If the user prompt is secondary, get its text
            ut_match = user_prompt.text
            return normalize_text(ut_query) == normalize_text(ut_match)
        else:  # If the user prompt is primary, check for attached (secondary) user prompts and get their texts
            attached_user_prompt_ids = user_prompt.attached_user_prompt_ids
            if (
                attached_user_prompt_ids
            ):  # primary user prompt is likely to have empty list of attached user prompts
                for attached_user_prompt_id in attached_user_prompt_ids:
                    try:
                        attached_user_text = UserPrompts.get_by_ids([attached_user_prompt_id])[
                            0
                        ].text
                        if normalize_text(ut_query) == normalize_text(attached_user_text):
                            logging.debug(
                                f'Skipping exact match for user prompt: "{ut_query}" with attached user prompt: "{attached_user_text}"'
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



# -----------


def get_user_prompt_id_from_source_node(source_node_id: str) -> list[str]:
    user_prompt_ids = []
    conv_paths = ConversationalPaths.query(
        sm.select(ConversationalPaths).where(ConversationalPaths.source_node_id == source_node_id)
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
