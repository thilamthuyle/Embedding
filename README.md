# Embedding Project

## Dataset Structure

```
├── inputs/
│   ├── ut_to_conv_path/
│   │   └── <language>/
│   │       └── <assistant_id>/
│   │           └── <conversation_id>/
│   │               ├── 0.json
│   │               ├── 1.json
│   │               └── ...
│   │
│   └── up_to_examples/
│       └── <conv_path_id>.json
│
└── outputs/
    ├── prod/
    │   └── <language>/
    │       └── <assistant_id>/
    │           └── <conversation_id>/
    │               ├── 0.json
    │               ├── 1.json
    │               └── ...
    │
    ├── <prompt_id>/
    │   └── <language>_<model>/
    │       └── <assistant_id>/
    │           └── <conversation_id>/
    │               ├── 0.json
    │               ├── 1.json
    │               └── ...
    │
    └── prompts/
        └── <prompt_id>.txt
```


## Structure of .json files 
- `ut_to_conv_path` files:
    ```.json
    {
    "assistant_id": str,
    "call_id": str,
    "language": str,
    "conversation": str,
    "user_text": str,
    "candidates": {
        "up": list[str],
        "aa": list[str],
        "aq": list[str],
        "conv_path_id": list[str],
        }
    }
    ```
- `up_to_examples` files:
    ```.json
    {
    "conv_path_id": str,
    "primary_user_prompt": str,
    "attached_user_prompts": list[str],
    }
    ```
- `outputs` files:
    ```.json
    {
    "up": str,
    "aa": str,
    "aq": str,
    "conv_path_id": str
    }
    ```

## Implementation notes
- Extract only depth 2 user prompt matching (for both distance>0 and distance=0 cases)
- Remove all matchings with `§NO_NEED§` conv path as candidate
- Use async functions whenever possible to speed up processing