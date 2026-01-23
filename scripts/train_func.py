#!/usr/bin/env python3
"""Train function model."""

import sys
from pathlib import Path
from datetime import datetime
from datasets import load_from_disk
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    print("=" * 80)
    print("🚀 Training Function Model")
    print("=" * 80)
    print()

    # Paths
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "func_dataset"

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_from_disk(str(dataset_path / "train"))
    eval_dataset = load_from_disk(str(dataset_path / "eval"))

    print(f"✅ Train: {len(train_dataset)} examples")
    print(f"✅ Eval: {len(eval_dataset)} examples")
    print()

    # Load model and tokenizer
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "google/functiongemma-270m-it",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    tokenizer.padding_side = "right"

    print("✅ Model loaded")
    print()

    # LoRA configuration
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    print()

    # Training arguments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "models" / f"func-{timestamp}"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        #bf16=False #If CPU training
    )

    # Formatting function for SFTTrainer
    def formatting_func(example):
        """Extract the formatted text from the example."""
        return example["text"]

    # Trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print("=" * 80)
    print("🏋️ Starting Training...")
    print("=" * 80)
    print()

    # Train
    trainer.train()
    
    # 1. Merge the weights
    print("Merging LoRA weights into base model...")
    merged_model = trainer.model.merge_and_unload()
    
    # Cast to half-precision (FP16 or BF16) before saving
    merged_model = merged_model.to(torch.float16)

    # 2. Save the full model
    final_merged_path = output_dir / "full_model"
    merged_model.save_pretrained(final_merged_path, safe_serialization=True)
    tokenizer.save_pretrained(final_merged_path, safe_serialization=True)

    print(f"✅ Full standalone model saved to: {final_merged_path}")
    
    '''
    
    # Save final model
    final_model_path = output_dir / "final"
    print(f"\n💾 Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))

    print("\n✅ Training complete!")
    print("=" * 80)
    print(f"\n📂 Model saved to: {final_model_path}")
    print("=" * 80)
    '''
if __name__ == "__main__":
    main()
