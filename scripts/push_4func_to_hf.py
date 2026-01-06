#!/usr/bin/env python3
"""Upload 4-function model to HuggingFace."""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_PATH = "models/music-4func-20260104_103212/final"
REPO_NAME = "Jageen/music-4func"  # Change if needed
COMMIT_MESSAGE = "Upload 4-function music assistant model (play_song, playback_control, search_music, create_playlist)"

# Model card content
MODEL_CARD = """---
license: apache-2.0
base_model: google/functiongemma-270m-it
tags:
- function-calling
- music
- peft
- lora
- functiongemma
library_name: peft
---

# Music Assistant - 4 Functions (FunctionGemma Fine-tuned)

This model is a fine-tuned version of [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) for music control function calling.

## Model Description

- **Base Model:** FunctionGemma-270M-it
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Training Data:** 100 examples (80 train, 20 eval)
- **Training Accuracy:** 98.9%
- **Evaluation Accuracy:** 98.5%
- **Training Time:** ~2.5 minutes on Mac M-series

## Supported Functions

This model can call 4 music control functions:

1. **play_song** - Play a specific song by name or artist
2. **playback_control** - Control playback (play, pause, skip, next, previous, stop, resume)
3. **search_music** - Search for music by query, artist, album, or genre
4. **create_playlist** - Create a new playlist with a given name

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/functiongemma-270m-it",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Jageen/music-4func")

# Load fine-tuned model
model = PeftModel.from_pretrained(base_model, "Jageen/music-4func")

# Define functions
FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "play_song",
            "description": "Play a specific song by name or artist",
            "parameters": {
                "type": "object",
                "properties": {
                    "song_name": {"type": "string", "description": "Name of the song to play"},
                    "artist": {"type": "string", "description": "Artist name (optional)"},
                    "album": {"type": "string", "description": "Album name (optional)"}
                },
                "required": ["song_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "playback_control",
            "description": "Control music playback",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["play", "pause", "skip", "next", "previous", "stop", "resume"]
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_music",
            "description": "Search for music by query, artist, album, or genre",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "type": {"type": "string", "enum": ["song", "artist", "album", "playlist", "genre"]}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_playlist",
            "description": "Create a new playlist with a given name",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the playlist"}
                },
                "required": ["name"]
            }
        }
    }
]

# Test
user_input = "Play Bohemian Rhapsody"
messages = [{"role": "user", "content": user_input}]

prompt = tokenizer.apply_chat_template(
    messages,
    tools=FUNCTIONS,
    add_generation_prompt=True,
    tokenize=False
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=False)
print(response)
```

## Training Details

- **LoRA Configuration:**
  - r=16, alpha=32
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Trainable params: 3.8M (1.40% of base model)

- **Training Arguments:**
  - Epochs: 5
  - Batch size: 2
  - Gradient accumulation: 4
  - Learning rate: 2e-4
  - Optimizer: AdamW

## Example Outputs

**Input:** "Play Bohemian Rhapsody"
**Output:** `<start_function_call>play_song{"song_name": "Bohemian Rhapsody"}<end_function_call>`

**Input:** "Pause the music"
**Output:** `<start_function_call>playback_control{"action": "pause"}<end_function_call>`

**Input:** "Search for rock songs"
**Output:** `<start_function_call>search_music{"query": "rock songs"}<end_function_call>`

**Input:** "Create a playlist called Workout Mix"
**Output:** `<start_function_call>create_playlist{"name": "Workout Mix"}<end_function_call>`

## Limitations

- Optimized for music control use cases
- May not generalize well to other domains
- Requires proper function schema definition

## Citation

```bibtex
@misc{music-4func-2024,
  author = {Jageen},
  title = {Music Assistant 4-Function Model},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/Jageen/music-4func}}
}
```

## License

Apache 2.0
"""

def main():
    print("=" * 80)
    print("📤 Uploading 4-Function Model to HuggingFace")
    print("=" * 80)
    print()

    # Check if model exists
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        return

    print(f"📂 Model path: {model_path}")
    print(f"🎯 Repo: {REPO_NAME}")
    print()

    # Check for HF token
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        print("⚠️  No HuggingFace token found in environment")
        print("Please set HUGGINGFACE_TOKEN or HF_TOKEN environment variable")
        print()
        print("To get a token:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Create a new token with 'write' access")
        print("  3. Export it: export HUGGINGFACE_TOKEN='your_token_here'")
        print()
        return

    # Initialize API
    api = HfApi(token=token)

    # Create repo (or get existing)
    print("Creating repository...")
    try:
        create_repo(
            repo_id=REPO_NAME,
            repo_type="model",
            exist_ok=True,
            token=token
        )
        print(f"✅ Repository created/verified: {REPO_NAME}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return

    print()

    # Upload model card
    print("Uploading model card...")
    try:
        api.upload_file(
            path_or_fileobj=MODEL_CARD.encode(),
            path_in_repo="README.md",
            repo_id=REPO_NAME,
            repo_type="model",
            commit_message=f"{COMMIT_MESSAGE} - README",
        )
        print("✅ Model card uploaded")
    except Exception as e:
        print(f"❌ Error uploading model card: {e}")

    print()

    # Upload model files
    print("Uploading model files...")
    try:
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=REPO_NAME,
            repo_type="model",
            commit_message=COMMIT_MESSAGE,
            ignore_patterns=["*.bin"]  # Only upload safetensors
        )
        print("✅ Model files uploaded")
    except Exception as e:
        print(f"❌ Error uploading model files: {e}")
        return

    print()
    print("=" * 80)
    print("🎉 Upload Complete!")
    print("=" * 80)
    print()
    print(f"🔗 Model URL: https://huggingface.co/{REPO_NAME}")
    print()
    print("Next steps:")
    print("  1. Visit the model page to verify")
    print("  2. Use in Colab for testing")
    print("  3. Share in your Medium blog!")
    print()

if __name__ == "__main__":
    main()
