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