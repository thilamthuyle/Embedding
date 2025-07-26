from getvocal.datamodel.sql.conversational_paths import ConversationalPaths
import sqlmodel as sm
import numpy as np
import pandas as pd
from vocal.common.utils import normalize_text
from getvocal.datamodel.sql.assistants import Assistants
from getvocal.datamodel.sql.assistant_texts import AssistantTexts
from getvocal.datamodel.sql.assistant_questions import AssistantQuestions

from getvocal.datamodel.sql.assistants import DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE


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

Given a list of scenarios containing what the speaker could mean and the assistant's response, identify the scenario where what the speaker says matches what the speaker has said in meaning. Pay less attention to the assistant's response and only focus on the speaker's intent. If there is no scenario that matches, you must output 'none'.

This is what the speaker has said: "{user_prompt}"

Here's the list of scenarios:

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
    conv_paths = ConversationalPaths.query(sm.select(ConversationalPaths).where(ConversationalPaths.source_node_id == source_node_id))
    for conv_path in conv_paths:
        user_prompt_ids.append(conv_path.user_prompt_id)

    return user_prompt_ids


def is_normalized_text_matching(query: str, match: str) -> bool:
    """
    Check if the query and match are the same using normalized text comparison.
    """
    return normalize_text(query) == normalize_text(match)


def get_assistant_language(assistant_id: str) -> str:
    assistant = Assistants.get_by_ids([assistant_id])
    return assistant[0].language


def filter_matchings_by_language(df: pd.DataFrame, language: str) -> pd.DataFrame:
    """
    Filter the DataFrame to only include rows where the assistant's language matches the specified language.
    """
    return df[df["language"] == language]


def get_default_embedding_model(language: str) -> str:
    if language in DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE.keys():
        return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE[language]
    return DEFAULT_EMBEDDING_MODEL_PER_LANGUAGE["others"]


def cosine_distance(v1: list[float], v2: list[float]) -> float:
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return 1 - cos_sim


def get_texts_by_assistant(assistant_id: str) -> list[str]:
    assistant_texts = AssistantTexts.query(sm.select(AssistantTexts).where(AssistantTexts.assistant_id == assistant_id))
    return [text.text for text in assistant_texts]


def get_questions_by_assistant(assistant_id: str) -> list[str]:
    assistant_questions = AssistantQuestions.query(sm.select(AssistantQuestions).where(AssistantQuestions.assistant_id == assistant_id))
    return [question.text for question in assistant_questions]