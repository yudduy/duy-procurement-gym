# PLAN: Zero-shot smoke test on Apple Silicon via MLX. REQ-6.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
# ADAPTED FROM: mlx_lm standard generate API (mlx-examples/llms)
# ADAPTED FROM: scripts/generate_sft_data.py (instance generation + baseline eval pattern)
"""Zero-shot smoke test using MLX on Apple Silicon."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from procurement_gym.baselines import (
    category_partition,
    equal_size_partition,
    geographic_partition,
)
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.evaluation import evaluate_partition
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.parser import parse_partition
from procurement_gym.verifier.serializer import serialize_instance, serialize_partition

# MLX imports
from mlx_lm import generate, load


def main() -> None:
    n_instances = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    model_name = sys.argv[2] if len(sys.argv) > 2 else "mlx-community/Qwen2.5-7B-Instruct-4bit"

    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    print("Model loaded.\n")

    sc = SupplierModelConfig(coupling_strength=1.0)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)

    results: list[dict] = []
    t0 = time.monotonic()

    for i in range(n_instances):
        seed = 900000 + i
        instance = gen.generate(seed=seed)
        n_items = len(instance.items)
        n_cats = len({item.category for item in instance.items})
        rng = np.random.default_rng(seed)

        prompt = serialize_instance(instance)

        # Format as chat using tokenizer's chat template
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a procurement optimization expert. "
                    "Analyze the items and suppliers, then output your partition. "
                    "Keep reasoning brief. You MUST end with the partition in the exact format: "
                    "<partition>[lot_id_for_item_0, lot_id_for_item_1, ...]</partition>"
                ),
            },
            {"role": "user", "content": prompt},
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate completion
        t1 = time.monotonic()
        completion = generate(model, tokenizer, prompt=chat_prompt, max_tokens=1024)
        gen_time = time.monotonic() - t1

        # Parse LLM output into partition
        partition = parse_partition(completion, n_items)
        parsed = partition is not None

        # Score if parsed successfully
        reward = None
        if parsed:
            result = evaluate_partition(partition, instance, sc, seed=0)
            reward = result.reward

        # Evaluate baselines for comparison
        cat_r = evaluate_partition(category_partition(instance), instance, sc, seed=0).reward
        geo_r = evaluate_partition(
            geographic_partition(instance, n_lots=n_cats, rng=rng), instance, sc, seed=0
        ).reward
        eq_r = evaluate_partition(
            equal_size_partition(instance, n_lots=n_cats), instance, sc, seed=0
        ).reward
        best_baseline = max(cat_r, geo_r, eq_r)

        entry = {
            "seed": seed,
            "parsed": parsed,
            "reward": reward,
            "best_baseline": best_baseline,
            "beats_baseline": reward is not None and reward >= best_baseline,
            "gen_time": gen_time,
            "completion": completion,
        }
        results.append(entry)

        status = "VALID" if parsed else "FAIL "
        rwd = f"{reward:>10.1f}" if reward is not None else "       N/A"
        print(
            f"[{i + 1:>2}/{n_instances}] {status} | reward={rwd} | baseline={best_baseline:>10.1f} | "
            f"beats={entry['beats_baseline']} | {gen_time:.1f}s"
        )
        if not parsed:
            preview = completion[:150].replace("\n", "\\n")
            print(f"       Output: {preview}")

    elapsed = time.monotonic() - t0
    parse_ok = sum(1 for r in results if r["parsed"])
    valid = sum(1 for r in results if r["reward"] is not None and r["reward"] > -10000)
    beats = sum(1 for r in results if r["beats_baseline"])

    print("\n" + "=" * 70)
    print(f"SMOKE TEST: {model_name}")
    print("=" * 70)
    print(f"Instances:          {n_instances}")
    print(f"Parse success:      {parse_ok}/{n_instances} ({parse_ok / n_instances:.0%})")
    print(f"Valid partitions:   {valid}/{n_instances} ({valid / n_instances:.0%})")
    print(f"Beats baseline:     {beats}/{valid if valid else 1}")
    print(f"Total time:         {elapsed:.1f}s ({elapsed / n_instances:.1f}s/instance)")

    if valid > 0:
        valid_rewards = [
            r["reward"] for r in results if r["reward"] is not None and r["reward"] > -10000
        ]
        baselines = [
            r["best_baseline"] for r in results if r["reward"] is not None and r["reward"] > -10000
        ]
        print(
            f"LLM reward:         mean={np.mean(valid_rewards):.1f}, std={np.std(valid_rewards):.1f}"
        )
        print(f"Baseline reward:    mean={np.mean(baselines):.1f}")
        gap = np.mean(valid_rewards) - np.mean(baselines)
        print(f"Gap vs baseline:    {gap:+.1f}")
    print("=" * 70)

    out_path = "smoke_test_results.json"
    with open(out_path, "w") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
