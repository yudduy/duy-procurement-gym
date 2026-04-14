# PLAN: SFT training on search-optimized procurement lot design data.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
# ADAPTED FROM: TRL SFTTrainer docs (huggingface/trl, conversational format)
"""SFT training script for procurement lot design."""

from __future__ import annotations

import sys

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
    data_path = sys.argv[2] if len(sys.argv) > 2 else "sft_data.jsonl"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "checkpoints/sft"

    # Load dataset (messages format auto-detected by TRL)
    dataset = load_dataset("json", data_files=data_path, split="train")
    # Remove metadata column — SFTTrainer only needs 'messages'
    if "metadata" in dataset.column_names:
        dataset = dataset.remove_columns("metadata")

    print(f"Dataset: {len(dataset)} examples")
    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")

    # LoRA config — efficient fine-tuning
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )

    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_length=4096,
        logging_steps=10,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir + "/final")
    print(f"SFT training complete. Model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
