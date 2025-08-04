# Embedding Project

## Dataset Structure

```
matching_dataset/
├── inputs/
│   ├── ut_to_conv_path/
│   │   └── <language>/
│   │       └── <assistant_id>/
│   │           └── <call_id>/
│   │               ├── transcript.json  # list of messages 
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
      "assistant_id": "string",
      "call_id": "string",
      "language": "string",
      "conversation": "int", # index of the last USER message (which contains user_text)
      "user_text": "string",
      "candidates": {
        "up": ["string"],
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