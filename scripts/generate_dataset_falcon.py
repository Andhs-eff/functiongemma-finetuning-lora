#!/usr/bin/env python3
"""Generate dataset."""

import sys
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.func_examples_full import FUNC_EXAMPLES

# 2 functions
TWO_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "check_pin",
            "description": "Check if the provided PIN is correct. The PIN usually is a 4-digit number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pin": {
                        "type": "string",
                        "description": "The PIN number (consisting of digits) to verify"
                    }
                },
                "required": ["pin"]
            },
            "return": {
                "type": "string",
                "description": "Returns 'Please choose the language' if PIN is correct (7979), otherwise 'PIN is incorrect. Try again'"
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_language",
            "description": "Check if the provided language is supported.",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "The name of the language to verify"
                    }
                },
                "required": ["language"]
            },
            "return": {
                "type": "string",
                "description": "Returns 'Now I will connect to the interpreter' if language is Spanish, otherwise 'Please choose another language'"
            }
        }
    }
]

def main():
    print("=" * 80)
    print("📊 Generating Dataset")
    print("=" * 80)
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("tiiuae/Falcon-H1-Tiny-Tool-Calling-90M")

    # Convert examples to formatted training data
    print(f"Processing {len(FUNC_EXAMPLES)} training examples...")
    formatted_examples = []

    for user_input, function_name, arguments in FUNC_EXAMPLES:
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
            tools=TWO_FUNCTIONS,
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
    output_dir = project_root / "data" / "func_dataset"
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
    print(f"  Functions: 2 (check_pin, check_language)")
    print("=" * 80)

if __name__ == "__main__":
    main()
