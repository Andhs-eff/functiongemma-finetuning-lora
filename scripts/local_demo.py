#!/usr/bin/env python3
"""Local demonstration comparing base model vs fine-tuned model.

This script provides the same comparison as the Colab notebook but runs locally.
Perfect for taking screenshots for blog posts.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Model path
MODEL_PATH = "models/music-4func-20260104_103212/final"

# Define our 4 custom music functions
MUSIC_FUNCTIONS = [
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

# Test cases
TEST_CASES = [
    ("Play Bohemian Rhapsody", "play_song"),
    ("Play Imagine by John Lennon", "play_song"),
    ("Pause the music", "playback_control"),
    ("Skip to next song", "playback_control"),
    ("Search for rock songs", "search_music"),
    ("Find songs by The Beatles", "search_music"),
    ("Create a playlist called Workout Mix", "create_playlist"),
    ("Make a new playlist named Chill Vibes", "create_playlist"),
]

def test_model(model, tokenizer, model_name="Model"):
    """Test a model with our test cases."""
    print(f"\n{'='*80}")
    print(f"🧪 Testing {model_name}")
    print(f"{'='*80}\n")

    results = []

    for i, (user_input, expected_function) in enumerate(TEST_CASES, 1):
        print(f"Test {i}/{len(TEST_CASES)}: \"{user_input}\"")

        # Prepare input
        messages = [{"role": "user", "content": user_input}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=MUSIC_FUNCTIONS,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=False
        )

        # Check result
        success = expected_function in response and "<start_function_call>" in response
        results.append(success)

        if success:
            print(f"  ✅ PASS")
            # Extract function call for display
            if "<start_function_call>" in response:
                start = response.find("<start_function_call>") + len("<start_function_call>")
                end = response.find("<end_function_call>")
                if end > start:
                    call = response[start:end]
                    # Truncate if too long
                    if len(call) > 100:
                        call = call[:100] + "..."
                    print(f"     {call}")
        else:
            print(f"  ❌ FAIL")
            print(f"     Expected: {expected_function}")
            # Show first 50 chars of response
            preview = response[:50].replace('\n', ' ')
            if len(response) > 50:
                preview += "..."
            print(f"     Got: {preview}")
        print()

    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = (passed / total) * 100

    print(f"{'='*80}")
    print(f"📊 Results for {model_name}")
    print(f"{'='*80}")
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    print(f"{'='*80}\n")

    return results, success_rate

def main():
    print("="*80)
    print("🎵 FunctionGemma: Base vs Fine-Tuned Comparison")
    print("="*80)
    print()

    # Part 1: Test Base Model
    print("📥 Loading base FunctionGemma model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/functiongemma-270m-it",
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    print("✅ Base model loaded!\n")

    base_results, base_accuracy = test_model(
        base_model,
        base_tokenizer,
        "Base FunctionGemma (No Fine-Tuning)"
    )

    # Part 2: Test Fine-Tuned Model
    print("\n📥 Loading fine-tuned model...")
    print(f"Model path: {MODEL_PATH}")

    finetuned_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✅ Tokenizer loaded")

    # Load PEFT adapter and merge
    peft_model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    print("✅ PEFT adapter loaded")

    print("🔧 Merging adapters...")
    finetuned_model = peft_model.merge_and_unload()
    print("✅ Adapters merged!\n")

    finetuned_results, finetuned_accuracy = test_model(
        finetuned_model,
        finetuned_tokenizer,
        "Fine-Tuned Model (4 Functions, 98.9% Training Accuracy)"
    )

    # Part 3: Comparison Summary
    print("\n" + "="*80)
    print("🎯 FINAL COMPARISON SUMMARY")
    print("="*80)
    print(f"\n📊 Base Model Performance:     {base_accuracy:.1f}% ({sum(base_results)}/{len(base_results)} tests passed)")
    print(f"📊 Fine-Tuned Model Performance: {finetuned_accuracy:.1f}% ({sum(finetuned_results)}/{len(finetuned_results)} tests passed)")
    print(f"\n📈 Improvement: +{finetuned_accuracy - base_accuracy:.1f} percentage points")

    if finetuned_accuracy == 100:
        print("\n🎉 Perfect score! The fine-tuned model passed all tests!")
    elif finetuned_accuracy > base_accuracy:
        print("\n✅ Fine-tuning successfully improved the model's performance!")
    else:
        print("\n⚠️ No improvement observed.")

    print("\n" + "="*80)
    print("📝 Training Details:")
    print("="*80)
    print("• Dataset: 100 examples (80 train, 20 eval)")
    print("• Method: LoRA fine-tuning (r=16, alpha=32)")
    print("• Epochs: 5")
    print("• Training Time: ~2.5 minutes")
    print("• Final Training Accuracy: 98.9%")
    print("• Trainable Parameters: 3.8M (1.40% of base model)")
    print("="*80 + "\n")

    print("✅ Demo complete! Take screenshots of the output above for your blog.\n")

if __name__ == "__main__":
    main()
