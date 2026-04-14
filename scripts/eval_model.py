# PLAN: Evaluate a trained model on held-out procurement instances.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
# ADAPTED FROM: scripts/run_smoke_test_mlx.py (eval loop pattern)
"""Evaluate a LoRA model on held-out instances vs baselines."""

from __future__ import annotations

import sys

import numpy as np
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

sys.path.insert(0, "src")

from procurement_gym.baselines import (
    category_partition,
    equal_size_partition,
    geographic_partition,
)
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.evaluation import evaluate_partition
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.parser import parse_partition
from procurement_gym.verifier.serializer import serialize_instance


def main() -> None:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "grpo-output/final"
    n_instances = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    base_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 700000

    model = AutoPeftModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    sc = SupplierModelConfig(coupling_strength=1.0)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)

    system_msg = (
        "You are a procurement optimization expert. "
        "Analyze the items and suppliers, then output your partition. "
        "Keep reasoning brief. You MUST end with the partition in the exact format: "
        "<partition>[lot_id_for_item_0, lot_id_for_item_1, ...]</partition>"
    )

    parse_ok = 0
    rewards_model: list[float] = []
    rewards_baseline: list[float] = []
    beats = 0

    for i in range(n_instances):
        seed = base_seed + i
        instance = gen.generate(seed=seed)
        prompt = serialize_instance(instance)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        partition = parse_partition(completion, 20)

        n_cats = len({item.category for item in instance.items})
        rng = np.random.default_rng(seed)
        cat_r = evaluate_partition(category_partition(instance), instance, sc, seed=0).reward
        geo_r = evaluate_partition(
            geographic_partition(instance, n_lots=n_cats, rng=rng), instance, sc, seed=0
        ).reward
        eq_r = evaluate_partition(
            equal_size_partition(instance, n_lots=n_cats), instance, sc, seed=0
        ).reward
        best_bl = max(cat_r, geo_r, eq_r)
        rewards_baseline.append(best_bl)

        if partition is not None:
            parse_ok += 1
            r = evaluate_partition(partition, instance, sc, seed=0).reward
            rewards_model.append(r)
            if r >= best_bl:
                beats += 1
            print(
                f"{i:>2}: reward={r:>10.1f} baseline={best_bl:>10.1f} "
                f"gap={r - best_bl:>+8.1f} lots={partition.n_lots}"
            )
        else:
            rewards_model.append(float("nan"))
            print(f"{i:>2}: PARSE_FAIL  baseline={best_bl:>10.1f}  output={completion[:80]}")

    print()
    print(f"Parse: {parse_ok}/{n_instances}")
    valid_rewards = [r for r in rewards_model if not np.isnan(r)]
    if valid_rewards:
        print(
            f"Model reward:    mean={np.mean(valid_rewards):.1f}, std={np.std(valid_rewards):.1f}"
        )
        print(f"Baseline reward: mean={np.mean(rewards_baseline):.1f}")
        gap = np.mean(valid_rewards) - np.mean(rewards_baseline)
        print(f"Beats baseline:  {beats}/{parse_ok}")
        print(f"Gap: {gap:+.1f} ({gap / np.mean(rewards_baseline) * 100:+.2f}%)")


if __name__ == "__main__":
    main()
