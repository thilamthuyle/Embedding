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
    "nl": "dutch",
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
        if up_ids[i] in existing_ups_dict.keys() and aa_ids[i] in existing_aas_dict.keys():
            candidates["up"].append(existing_ups_dict[up_ids[i]])
            candidates["aa"].append(existing_aas_dict[aa_ids[i]])
            if aq_ids[i] is not None and aq_ids[i] in existing_aqs_dict.keys():
                candidates["aq"].append(existing_aqs_dict[aq_ids[i]])
            else:
                candidates["aq"].append(None)  # follow up question is optional
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
    file_path.write_text(json.dumps(matching, ensure_ascii=False), encoding="utf-8")
    logging.debug(f"Saved matching to {file_path}")


def process_call_transcript(
    call_transcript_path: Path,
    ut_to_conv_path_dir: Path,
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

        save_ut_to_conv_path_matching(
            Path(f"{ut_to_conv_path_dir}/{language}/{assistant_id}/{call_id}"),
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
        Path(
            f"{ut_to_conv_path_dir}/{language}/{assistant_id}/{call_id}/conversation.json"
        ).write_text(json.dumps(conversation), encoding="utf-8")

    logging.info(
        f"Processed call transcript: {assistant_id} / {call_id} with {matching_id} matchings."
    )


def process_matching_json_file(up_to_examples_dir: Path, candidates: dict[str, list[str]]):
    """
    Save new user prompt to examples matching from candidates into JSON files.
    """
    all_conv_path_ids = []
    all_user_prompt_ids = set()
    for conv_path_id in candidates["conv_path_id"]:
        # Check if a conv_path_id already exists, skip processing
        if Path(f"{up_to_examples_dir}/{conv_path_id}.json").exists():
            logging.debug(
                f"File {up_to_examples_dir}/{conv_path_id}.json already exists, skipping..."
            )
            continue
        all_conv_path_ids.append(conv_path_id)
        all_user_prompt_ids.add(
            conv_path_id.split("_")[1]
        )  # Assuming conv_path_id is depth 2 conv_path, whose format is source_up_aa_aq
    user_prompts = UserPrompts.query(
        sm.select(UserPrompts).where(UserPrompts.id.in_(all_user_prompt_ids))
    )
    # user_prompts = UserPrompts.get_by_ids(all_user_prompt_ids)
    user_prompts_dict = {up.id: up for up in user_prompts}

    for conv_path_id in all_conv_path_ids:
        user_prompt_id = conv_path_id.split("_")[1]
        user_prompt = user_prompts_dict[user_prompt_id]
        if user_prompt.primary_id:
            # If the user prompt is secondary, get its text
            matching = {
                "conv_path_id": conv_path_id,
                "primary_user_prompt": None,
                "attached_user_prompts": [user_prompt.text],
            }
        else:
            # If the user prompt is primary, check for attached (secondary) user prompts and get their texts
            attached_up_ids = user_prompt.attached_user_prompt_ids
            attached_ups = UserPrompts.query(
                sm.select(UserPrompts.text).where(UserPrompts.id.in_(attached_up_ids))
            )
            attached_user_prompts = [up["text"] for up in attached_ups]
            matching = {
                "conv_path_id": conv_path_id,
                "primary_user_prompt": user_prompt.text,
                "attached_user_prompts": attached_user_prompts,
            }
        Path(up_to_examples_dir).mkdir(parents=True, exist_ok=True)
        file_path = Path(f"{up_to_examples_dir}/{conv_path_id}.json")
        file_path.write_text(
            json.dumps(matching, indent=2, ensure_ascii=False), encoding="utf-8"
        )  # ensure_ascii=False to preserve § instead of converting it to \u00a7
        logging.debug(f"Saved up_to_examples matching to {file_path}")


def get_outputs_from_conv_path_ids(
    conv_path_ids: list[str],
) -> list[dict]:
    """
    Get outputs from the database for given conv_path_ids.
    Return:
        outputs: list of dictionaries with conv_path_id, up, aa, aq
    """
    # Query DB and construct conv_path_id dictionary
    all_conv_paths = ConversationalPaths.query(
        sm.select(*ConvPath.columns()).where(ConversationalPaths.id.in_(conv_path_ids))
    )
    conv_paths = [ConvPath(**cp) for cp in all_conv_paths]
    conv_path_id_dict = {
        cp.id: {"up": cp.user_prompt_id, "aa": cp.assistant_answer_id, "aq": cp.target_node_id}
        for cp in conv_paths
    }

    # Get all up, aa, aq ids from all conv_paths then query the DB only once to construct up, aa, aq dictionaries
    up_ids = []
    aa_ids = []
    aq_ids = []
    for conv_path in conv_paths:
        up_ids.append(conv_path.user_prompt_id)
        aa_ids.append(conv_path.assistant_answer_id)
        aq_ids.append(conv_path.target_node_id)

    ups = UserPrompts.query(
        sm.select(UserPrompts.id, UserPrompts.text).where(UserPrompts.id.in_(up_ids))
    )
    aas = AssistantAnswers.query(
        sm.select(AssistantAnswers.id, AssistantAnswers.text).where(AssistantAnswers.id.in_(aa_ids))
    )
    aqs = AssistantQuestions.query(
        sm.select(AssistantQuestions.id, AssistantQuestions.text).where(
            AssistantQuestions.id.in_(aq_ids)
        )
    )

    up_dict = {up["id"]: up["text"] for up in ups}
    aa_dict = {aa["id"]: aa["text"] for aa in aas}
    aq_dict = {aq["id"]: aq["text"] for aq in aqs}

    # Construct outputs, ensuring that outputs have the same length as the initial conv_path_ids
    outputs = [
        {
            "conv_path_id": conv_path_ids[i],
            "up": up_dict[conv_path_id_dict[conv_path_ids[i]]["up"]],
            "aa": aa_dict[conv_path_id_dict[conv_path_ids[i]]["aa"]],
            "aq": aq_dict[conv_path_id_dict[conv_path_ids[i]]["aq"]]
            if conv_path_id_dict[conv_path_ids[i]]["aq"] in aq_dict
            else "",  # aq is optional (may not be present in the DB)
        }
        for i in range(len(conv_path_ids))
    ]

    return outputs
