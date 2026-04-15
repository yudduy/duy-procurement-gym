# PLAN: GPU smoke test that runs 3 GRPO steps to verify the full pipeline works
#        before committing to a long training run. Checks for mode collapse,
#        reward variance, and completion diversity on the actual GPU hardware.
# ADAPTED FROM: scripts/train_grpo.py (this repo) — stripped to 3 steps for verification
# SOURCE: scripts/train_grpo.py + TRL GRPOTrainer docs
"""GPU smoke test for GRPO training pipeline."""

from __future__ import annotations

import sys

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, "src")

from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.reward import ProcurementVerifier, format_reward
from procurement_gym.verifier.serializer import SYSTEM_PROMPT, serialize_instance


# --- Reward function (copied from train_grpo.py) ---

_SC = SupplierModelConfig(coupling_strength=1.0)
_IC = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
_GEN = TransportInstanceGenerator(_IC, _SC)
_VERIFIER_CACHE: dict[int, ProcurementVerifier] = {}


def _get_verifier(seed: int) -> ProcurementVerifier:
    if seed not in _VERIFIER_CACHE:
        instance = _GEN.generate(seed=seed)
        _VERIFIER_CACHE[seed] = ProcurementVerifier(instance, _SC)
    return _VERIFIER_CACHE[seed]


def procurement_reward_fn(completions, instance_seed=None, **kwargs):
    rewards = []
    for i, completion in enumerate(completions):
        seed = int(instance_seed[i])
        verifier = _get_verifier(seed)
        r = verifier.reward_fn([completion])
        rewards.append(r[0])
    return rewards


# --- Build tiny dataset (copied from train_grpo.py make_grpo_dataset) ---


def make_dataset(n: int = 10, base_seed: int = 900000) -> Dataset:
    records = []
    for i in range(n):
        seed = base_seed + i
        instance = _GEN.generate(seed=seed)
        prompt_text = serialize_instance(instance)
        records.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                "instance_seed": seed,
            }
        )
    return Dataset.from_list(records)


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-7B-Instruct"
    temperature = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    print("=" * 60)
    print("SMOKE TEST: GRPO Pipeline Verification")
    print("=" * 60)

    # 1. Check GPU
    print(
        f"\n[1/6] GPU: {torch.cuda.get_device_name(0)}, "
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    )

    # 2. Build dataset
    dataset = make_dataset(n=10)
    print(f"[2/6] Dataset: {len(dataset)} prompts")

    # 3. Load model
    print(f"[3/6] Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # 4. Configure GRPO — 3 steps only
    training_args = GRPOConfig(
        output_dir="/tmp/smoke_grpo",
        num_train_epochs=1,
        max_steps=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        num_generations=4,
        generation_batch_size=4,
        max_completion_length=512,
        temperature=temperature,
        beta=0.04,
        scale_rewards="group",
        reward_weights=[1.0, 0.5],
        logging_steps=1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
        use_vllm=False,
        log_completions=True,
    )

    print(f"[4/6] GRPOConfig OK (temperature={temperature}, num_gen=4, max_completion=512)")

    # 5. Create trainer and run 3 steps
    print("[5/6] Creating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[procurement_reward_fn, format_reward],
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("[5/6] Training 3 steps...")
    trainer.train()

    # 6. Analyze completions from logs
    print("\n" + "=" * 60)
    print("[6/6] DIAGNOSTIC SUMMARY")
    print("=" * 60)

    log_history = trainer.state.log_history
    for entry in log_history:
        if "completions/mean_length" in entry:
            mean_len = entry["completions/mean_length"]
            min_len = entry["completions/min_length"]
            max_len = entry["completions/max_length"]
            reward_mean = entry.get("reward", "?")
            reward_std = entry.get("reward_std", "?")
            entropy = entry.get("entropy", "?")
            fmt_mean = entry.get("rewards/format_reward/mean", "?")

            print(f"  Step {entry.get('step', '?')}:")
            print(f"    completion_length: mean={mean_len}, min={min_len}, max={max_len}")
            print(f"    reward: mean={reward_mean}, std={reward_std}")
            print(f"    format_reward: {fmt_mean}")
            print(f"    entropy: {entropy}")

            if mean_len == min_len == max_len:
                print(f"    WARNING: MODE COLLAPSE — all completions identical length ({mean_len})")
            if isinstance(reward_std, (int, float)) and reward_std < 10:
                print(f"    WARNING: LOW REWARD VARIANCE — std={reward_std}")
            if isinstance(entropy, (int, float)) and entropy < 0.3:
                print(f"    WARNING: LOW ENTROPY — {entropy}")

    # Generate a few completions manually to inspect content
    print("\n--- Sample completions (manual generation) ---")
    instance = _GEN.generate(seed=999999)
    prompt_text = serialize_instance(instance)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    for i in range(3):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
            )
        completion = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"\n  Completion {i + 1} ({len(completion)} chars):")
        print(f"    {completion[:300]}{'...' if len(completion) > 300 else ''}")
        has_tags = "<partition>" in completion and "</partition>" in completion
        print(f"    has_tags: {has_tags}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
