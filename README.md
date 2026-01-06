# 🎵 Fine-Tuning FunctionGemma for Music Control

Train Google's FunctionGemma-270M model to understand custom music functions in under 3 minutes on a CPU.

## ✨ Results

Improved model performance from **75% → 100%** accuracy using a gradual scaling approach:

| Model | Accuracy | Functions |
|-------|----------|-----------|
| Base FunctionGemma | 75% | 4 functions |
| Fine-Tuned | **100%** | 4 functions |

**Live Models**:
- [Jageen/music-2func](https://huggingface.co/Jageen/music-2func) - 2 functions, 100% accuracy
- [Jageen/music-4func](https://huggingface.co/Jageen/music-4func) - 4 functions, 98.9% accuracy

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- 8GB RAM minimum
- HuggingFace account with FunctionGemma access ([get access](https://huggingface.co/google/functiongemma-270m-it))

### Setup

```bash
# Clone the repository
cd music_app_training

# Run setup script
chmod +x setup.sh
./setup.sh

# Activate environment
source venv/bin/activate

# Login to HuggingFace
python -c "from huggingface_hub import login; login()"
```

### Train Your Model

**4-Function Model** (2-3 minutes)
```bash
# Generate dataset
python scripts/generate_4func_dataset.py

# Train
python scripts/train_4func.py

# Expected: 98.9% training accuracy
```

### Test Locally

Compare base model vs fine-tuned performance:

```bash
python scripts/local_demo.py
```

Output:
```
Base FunctionGemma: 75% (6/8 tests)
Fine-Tuned Model: 100% (8/8 tests)
Improvement: +25 percentage points
```

### Deploy to HuggingFace

```bash
export HUGGINGFACE_TOKEN='hf_your_token_here'
python scripts/push_4func_to_hf.py
```

## 🎯 What It Does

Converts natural language to structured function calls:

```python
Input:  "Play Bohemian Rhapsody"
Output: play_song(song_name="Bohemian Rhapsody")

Input:  "Pause the music"
Output: playback_control(action="pause")

Input:  "Search for rock songs"
Output: search_music(query="rock songs")
```

## 📊 Training Details

**Approach**: Gradual scaling (2→4→8→18 functions)
- Start small, validate, then scale
- Prevents cognitive overload
- Achieves 95-100% accuracy per stage

**Efficient Fine-Tuning**:
- Method: LoRA (Low-Rank Adaptation)
- Trainable params: 3.8M (1.4% of base model)
- Training time: ~2.5 minutes per stage (CPU)
- Model size: 15MB adapter

**Training Configuration**:
- Base model: google/functiongemma-270m-it
- Epochs: 5
- Batch size: 2
- Learning rate: 2e-4
- Examples per function: 25-30

## 📁 Project Structure

```
music_app_training/
├── scripts/
│   ├── generate_4func_dataset.py   # Generate training data
│   ├── train_4func.py              # Train the model
│   ├── local_demo.py               # Local comparison demo
│   └── push_4func_to_hf.py         # Deploy to HuggingFace
├── config/
│   └── music_functions.py          # Function definitions
├── data/
│   └── four_func_dataset/          # Training datasets
├── requirements.txt                # Dependencies
└── setup.sh                        # Setup script
```

## 🔧 Usage in Production

### Python

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "google/functiongemma-270m-it",
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, "Jageen/music-4func")
tokenizer = AutoTokenizer.from_pretrained("Jageen/music-4func")

# Merge for faster inference (recommended)
model = model.merge_and_unload()

# Use for inference
def predict(user_input):
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=MUSIC_FUNCTIONS,
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(outputs[0])

# Test
result = predict("Play Bohemian Rhapsody")
print(result)
# Output: <start_function_call>call:play_song{song_name:<escape>Bohemian Rhapsody<escape>}<end_function_call>
```

### iOS/Android

Models are compatible with mobile deployment. See [HuggingFace deployment guides](https://huggingface.co/docs) for platform-specific instructions.

## 🎯 Key Findings

**What Works**:
- Gradual scaling: 2→4→8→18 functions
- Complete LoRA configuration (all 7 target modules)
- Proper data formatting (pass dicts, not JSON strings)
- 25-30 examples per function minimum

**Results by Approach**:
| Approach | Functions | Accuracy | Status |
|----------|-----------|----------|--------|
| All 18 at once | 18 | 0% | ❌ Failed |
| Gradual (2 func) | 2 | 100% | ✅ Success |
| Gradual (4 func) | 4 | 98.9% | ✅ Success |

## 📚 Resources

**Models**:
- [Jageen/music-2func](https://huggingface.co/Jageen/music-2func)
- [Jageen/music-4func](https://huggingface.co/Jageen/music-4func)

**Documentation**:
- [FunctionGemma Official Docs](https://ai.google.dev/gemma/docs/functiongemma)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)

## 🤝 Contributing

This project is open for contributions. Feel free to:
- Add more function examples
- Improve training efficiency
- Expand to more functions
- Share deployment experiences

## 📄 License

This project uses FunctionGemma, which requires accepting the [Gemma license](https://huggingface.co/google/functiongemma-270m-it).

## 🙏 Acknowledgments

- Google for FunctionGemma
- HuggingFace for transformers, PEFT, and TRL
- Open-source community for LoRA research

---

**Happy Training! 🎵🤖**

For detailed technical notes, troubleshooting, and development history, see [DEVELOPMENT_NOTES.md](./DEVELOPMENT_NOTES.md).
