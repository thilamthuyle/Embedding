# Embedding Project

## Dataset Structure

```
up_matching_dataset/
├── inputs/
│   ├── ut_to_conv_path/
│   │   └── <language>/
│   │       └── <assistant_id>/
│   │           └── <call_id>/
│   │               ├── conversation.json  # message list
│   │               ├── 0.json
│   │               ├── 1.json
│   │               └── ...
│   └── up_to_examples/
│       └── <conv_path_id>.json
└── outputs/
    ├── prod/
    │   └── <language>/
    │       └── <assistant_id>/
    │           └── <call_id>/
    │               ├── 0.json
    │               ├── 1.json
    │               └── ...
    ├── <prompt_id>/
    │   └── <language>_<model>/
    │       └── <assistant_id>/
    │           └── <call_id>/
    │               ├── 0.json
    │               ├── 1.json
    │               └── ...
    └── prompts/
        └── <prompt_id>.txt
```


## Structure of .json files 
- `ut_to_conv_path` files:
    ```json
    {
      "language": "string",
      "assistant_id": "string",
      "call_id": "string",
      "user_text": "string",
      "user_text_idx": "int", # index of the user_text message in the conversation (in conversation.json)
      "candidates": {
        "up": ["string"], # this can be either primary or attached user_prompt
        "aa": ["string"],
        "aq": ["string"],
        "conv_path_id": ["string"]
      }
    }
    ```
- `up_to_examples` files:
    ```json
    {
      "conv_path_id": "string",
      "primary_user_prompt": "string",
      "attached_user_prompts": ["string"]
    }
    ```
- `outputs` files:
    ```json
    {
      "up": "string",
      "aa": "string",
      "aq": "string",
      "conv_path_id": "string"
    }
    ```

## Implementation notes
- Extract only depth 2 user prompt matching (for both distance>0 and distance=0 cases)
- Remove all matchings with `§NO_NEED§` conv path as candidate
- Use async functions whenever possible to speed up processing


## Notes/ Remarks
- The dataset contains some noisy entries. 

  - There are cases where the `target_node_id` given in the `conv_path_id` but can't be found in the DB, which results in the attribute `aq`being set to `None`. This issue is likely due to dataset refactoring or graph modifications.<br/>
  For example: <br/>
    ```json
    "assistant_id": "eyMe6oXKrytagjMyDkaC"
    "call_id": "300011c5-307a-4e2e-a98f-620ce3ef8567"
    "conv_path_id": "2a31094393f7502d5243bc09974c3290_afcf133de94e417b50385c37dc70b67f_a7c158dc173c1913faa9f41f8a597454_921fb9ae-df8b-435d-bcf1-c853c5e05bb1"
    ```
    The `target_node_id` = `921fb9ae-df8b-435d-bcf1-c853c5e05bb1` is not found.
  - The database contains noisy entries related to `primary_user_prompt` references. Specifically, the `primary_user_prompts` of all `attached_user_prompts` associated with a given `primary_user_prompt` are not always consistent with the original `primary_user_prompt`.
  
    For example: <br/>
    - Initial `primary_user_prompt` = `97a2a0d1fbc2d5f6994f815a6342e14b`
    - When querying the DB for all `attached_user_prompts` of this `primary_user_prompt`, then checking each attached prompt’s own primary_user_prompt, we find that:
      - Not all `attached_user_prompts` point back to the initial `primary_user_prompt`.
      - Some have different `primary_user_prompt` values, introducing inconsistency.

- In the current setting, all matchings are retained for every `call_id` across all assistants, including matchings with the same `conv_path_id`(i.e. those occuring on the same graph, at the same node, and along the same edge). Consequently, assistants with a higher frequency of calls contribute more matchings, leading to a more important presentation of their corresonding `conv_path_id`in the database. Furthermore, even with a single conversational graph, certain `conv_path_id`(or `conv_path`) occur more frequently than others, resulting in an additional imbalance. 

  This observation raises a data construction question: should all duplicate of `conv_path_id` matchings be preserved, thereby allowing frequently occuring paths to exert proportionally greater influence, or should the dataset be restricted to unique `conv_path_id` matchings, thereby normalizing the contribution of each path and assigning them equal importance?