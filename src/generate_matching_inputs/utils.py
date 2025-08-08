import json
import logging
import sqlmodel as sm
from pathlib import Path
from collections import defaultdict
from pydantic import BaseModel

import vocal.common.static  # noqa: F401
from vocal.common.utils import normalize_text
from getvocal.datamodel.sql import Assistants
from getvocal.datamodel.sql.user_prompts import UserPrompts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions
from getvocal.datamodel.sql.assistant_answers import AssistantAnswers
from getvocal.datamodel.sql.conversational_paths import ConversationalPaths

CALL_TRANSCRIPTS_DIR = "/www/files/call_transcripts"

LANGUAGES = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
}


# Configure logging to show DEBUG level messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)


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


def get_assistant_language(assistant_id: str) -> str:
    assistant = Assistants.get_by_ids([assistant_id])
    return assistant[0].language


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
            and (
                not aq_ids[i] or aq_ids[i] in existing_aqs_dict.keys()
            )  # follow up question is optional
        ):
            candidates["up"].append(existing_ups_dict[up_ids[i]])
            candidates["aa"].append(existing_aas_dict[aa_ids[i]])
            candidates["aq"].append(existing_aqs_dict[aq_ids[i]] if aq_ids[i] else None)
            candidates["conv_path_id"].append(conv_paths_from_source_node[i].id)

    return candidates


def check_normalized_text_matching(ut_query: str, user_prompt_id: str) -> bool:
    """
    Check exact matching between user text and user prompt ID.
        - if user prompt is secondary, compare user prompt's text with user text
        - if user prompt is primary, check attached (secondary) user prompts and compare their texts with user text
    """
    try:
        user_prompt = UserPrompts.get(user_prompt_id)

        if user_prompt.primary_id:
            # If the user prompt is secondary, get its text
            ut_match = user_prompt.text
            return normalize_text(ut_query) == normalize_text(ut_match)
        else:
            # If the user prompt is primary, check for attached (secondary) user prompts and get their texts
            attached_up_ids = user_prompt.attached_user_prompt_ids
            if attached_up_ids:
                # primary user prompt is likely to have empty list of attached user prompts
                attached_ups = UserPrompts.query(
                    sm.select(UserPrompts.text).where(UserPrompts.id.in_(attached_up_ids))
                )
                for up in attached_ups:
                    if normalize_text(ut_query) == normalize_text(up["text"]):
                        logging.debug(f'Skipping exact match for user prompt: "{ut_query}".')
                        return True
            return False
    except IndexError:
        logging.debug(f"Error retrieving user prompt: {user_prompt_id}.")
        return False


def save_ut_to_conv_path_matching(
    save_to_dir: Path,
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
    save_to_dir.mkdir(parents=True, exist_ok=True)
    file_path = Path(f"{save_to_dir}/{matching_id}.json")
    file_path.write_text(json.dumps(matching), encoding="utf-8")
    logging.debug(f"Saved matching to {file_path}")


def save_up_to_examples_matching(save_to_dir: Path, candidates: dict[str, list[str]]):
    """
    Save new user prompt to examples matching from candidates into JSON files.
    """
    for conv_path_id in candidates["conv_path_id"]:
        # Check if a conv_path_id already exists, skip processing
        if Path(f"{save_to_dir}/{conv_path_id}.json").exists():
            logging.debug(f"File {save_to_dir}/{conv_path_id}.json already exists, skipping...")
            continue

        user_prompt_id = conv_path_id.split("_")[1]  # Assuming conv_path_id is depth 2 conv_path, whose format is source_up_aa_aq        
        user_prompt = UserPrompts.get(user_prompt_id)
        if user_prompt.primary_id:
            # If the user prompt is secondary, get its text
            matching = {
                "conv_path_id": conv_path_id,
                "primary_user_prompt": None,
                "attached_user_prompts": [user_prompt.text]
            }
        else:
            # If the user prompt is primary, check for attached (secondary) user prompts and get their texts
            attached_up_ids = user_prompt.attached_user_prompt_ids
            attached_ups = UserPrompts.query(sm.select(UserPrompts.text).where(UserPrompts.id.in_(attached_up_ids)))
            attached_user_prompts = [up["text"] for up in attached_ups]
            matching = {
                "conv_path_id": conv_path_id,
                "primary_user_prompt": user_prompt.text,
                "attached_user_prompts": attached_user_prompts
            }
        save_to_dir.mkdir(parents=True, exist_ok=True)
        file_path = Path(f"{save_to_dir}/{conv_path_id}.json")
        file_path.write_text(json.dumps(matching, indent=2), encoding="utf-8")
        logging.debug(f"Saved up_to_examples matching to {file_path}")


# if __name__ == "__main__":
#     ut_query = "Vale, pero hacerlo rápido."
#     user_prompt_id = "50e9d0390e2cbf580a1fb266532be978"
#     check_normalized_text_matching(ut_query, user_prompt_id)
