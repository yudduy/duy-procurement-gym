# PLAN: GRPO training with ProcurementVerifier as reward function.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
# ADAPTED FROM: TRL GRPOTrainer docs (huggingface/trl, conversational reward_fn)
"""GRPO training script for procurement lot design."""

from __future__ import annotations

import json
import sys

import numpy as np
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# Add project source to path
sys.path.insert(0, "src")

from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.reward import ProcurementVerifier
from procurement_gym.verifier.serializer import serialize_instance


# --- Reward function (module-level for pickling) ---

_SC = SupplierModelConfig(coupling_strength=1.0)
_IC = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
_GEN = TransportInstanceGenerator(_IC, _SC)

# Cache verifiers per seed to avoid re-creating
_VERIFIER_CACHE: dict[int, ProcurementVerifier] = {}


def _get_verifier(seed: int) -> ProcurementVerifier:
    if seed not in _VERIFIER_CACHE:
        instance = _GEN.generate(seed=seed)
        _VERIFIER_CACHE[seed] = ProcurementVerifier(instance, _SC)
    return _VERIFIER_CACHE[seed]


def procurement_reward_fn(completions, instance_seed=None, **kwargs):
    """TRL GRPOTrainer-compatible reward function.

    completions: list[list[dict]] in conversational format
                 (each is [{role: assistant, content: ...}])
    instance_seed: list[int] from dataset column
    """
    rewards = []
    for i, completion in enumerate(completions):
        seed = int(instance_seed[i])
        verifier = _get_verifier(seed)
        # completion is already [{"role": "assistant", "content": "..."}]
        r = verifier.reward_fn([completion])
        rewards.append(r[0])
    return rewards


# --- Dataset generation ---


def make_grpo_dataset(n_instances: int = 1000, base_seed: int = 600000) -> Dataset:
    """Generate GRPO training dataset (prompts only, no completions)."""
    sc = SupplierModelConfig(coupling_strength=1.0)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)

    system_msg = (
        "You are a procurement optimization expert. "
        "Analyze the items and suppliers, then output your partition. "
        "Keep reasoning brief. You MUST end with the partition in the exact format: "
        "<partition>[lot_id_for_item_0, lot_id_for_item_1, ...]</partition>"
    )

    records = []
    for i in range(n_instances):
        seed = base_seed + i
        instance = gen.generate(seed=seed)
        prompt_text = serialize_instance(instance)
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt_text},
                ],
                "instance_seed": seed,
            }
        )

    return Dataset.from_list(records)


# --- Main ---


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/sft/final"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/grpo"
    n_instances = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    print(f"Model: {model_name}")
    print(f"Output: {output_dir}")
    print(f"Instances: {n_instances}")

    # Generate dataset
    dataset = make_grpo_dataset(n_instances=n_instances)
    print(f"Dataset: {len(dataset)} prompts")

    # LoRA config
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

    # GRPO config
    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_generations=8,
        max_completion_length=256,
        max_prompt_length=4096,
        beta=0.04,
        logging_steps=1,
        save_steps=50,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
        use_vllm=False,
    )

    # Load model — use base model name for GRPO (it applies its own LoRA)
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Train
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=procurement_reward_fn,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(output_dir + "/final")
    print(f"GRPO training complete. Model saved to {output_dir}/final")


if __name__ == "__main__":
    main()
