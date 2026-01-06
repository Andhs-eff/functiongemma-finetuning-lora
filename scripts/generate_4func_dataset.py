#!/usr/bin/env python3
"""Generate 4-function dataset for gradual scaling."""

import sys
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.four_func_examples import FOUR_FUNC_EXAMPLES

# 4 functions
FOUR_FUNCTIONS = [
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

def main():
    print("=" * 80)
    print("📊 Generating 4-Function Dataset (Gradual Scaling)")
    print("=" * 80)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")

    # Convert examples to formatted training data
    print(f"Processing {len(FOUR_FUNC_EXAMPLES)} training examples...")
    formatted_examples = []

    for user_input, function_name, arguments in FOUR_FUNC_EXAMPLES:
        # Create messages format
        messages = [
            {"role": "user", "content": user_input},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": arguments  # Pass dict directly!
                        }
                    }
                ]
            }
        ]

        # Apply chat template
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tools=FOUR_FUNCTIONS,
            tokenize=False,
            add_generation_prompt=False
        )

        formatted_examples.append({
            "text": formatted_text,
            "user_input": user_input,
            "function_name": function_name
        })

    print(f"✅ Created {len(formatted_examples)} formatted examples")

    # Split into train/eval (80/20)
    split_idx = int(len(formatted_examples) * 0.8)
    train_examples = formatted_examples[:split_idx]
    eval_examples = formatted_examples[split_idx:]

    print(f"📊 Train: {len(train_examples)} examples")
    print(f"📊 Eval: {len(eval_examples)} examples")

    # Create datasets
    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    # Save to disk
    output_dir = project_root / "data" / "four_func_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Saving dataset to {output_dir}")
    train_dataset.save_to_disk(str(output_dir / "train"))
    eval_dataset.save_to_disk(str(output_dir / "eval"))

    print("\n✅ Dataset generation complete!")
    print("=" * 80)
    print("\nDataset Statistics:")
    print(f"  Total examples: {len(formatted_examples)}")
    print(f"  Training examples: {len(train_examples)}")
    print(f"  Evaluation examples: {len(eval_examples)}")
    print(f"  Functions: 4 (play_song, playback_control, search_music, create_playlist)")
    print("\nNext step:")
    print("  python scripts/train_4func.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
