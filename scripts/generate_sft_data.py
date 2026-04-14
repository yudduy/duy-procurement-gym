# PLAN: SFT data generator — uses local search + hill-climbing to find
#        near-optimal partitions, then formats as (prompt, completion) pairs.
#        Search uses the deterministic verifier as oracle (the CO paper pattern).
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
# ADAPTED FROM: CO paper (2509.16865) — solver-generated SFT data for combinatorial optimization
"""Generate SFT training data via search-optimized solutions."""

from __future__ import annotations

import json
import logging
import multiprocessing
import sys
import time

import numpy as np

from procurement_gym.baselines import (
    category_partition,
    equal_size_partition,
    geographic_partition,
)
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.evaluation import evaluate_partition
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.serializer import (
    SYSTEM_PROMPT,
    serialize_instance,
    serialize_partition,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def hill_climb(
    instance,
    sc: SupplierModelConfig,
    n_lots: int,
    rng: np.random.Generator,
    max_steps: int = 100,
) -> tuple[LotPartition, float]:
    """Hill-climb from a random partition using the verifier as oracle."""
    n_items = len(instance.items)
    assignments = list(rng.integers(0, n_lots, size=n_items))
    # Ensure all lots have at least 1 item
    for lot in range(n_lots):
        if lot not in assignments:
            assignments[int(rng.integers(0, n_items))] = lot

    current = LotPartition(assignments=tuple(assignments), n_items=n_items)
    current_reward = evaluate_partition(current, instance, sc, seed=0).reward

    for _step in range(max_steps):
        improved = False
        for item_idx in rng.permutation(n_items):
            current_lot = current.assignments[int(item_idx)]
            for target_lot in range(n_lots):
                if target_lot == current_lot:
                    continue
                new_assignments = list(current.assignments)
                new_assignments[int(item_idx)] = target_lot
                candidate = LotPartition(assignments=tuple(new_assignments), n_items=n_items)
                candidate_reward = evaluate_partition(candidate, instance, sc, seed=0).reward
                if candidate_reward > current_reward:
                    current = candidate
                    current_reward = candidate_reward
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return current, current_reward


def search_best_partition(
    instance,
    sc: SupplierModelConfig,
    seed: int,
    n_restarts: int = 20,
    max_steps: int = 100,
) -> tuple[LotPartition, float, int]:
    """Search for the best partition across multiple lot counts and restarts."""
    n_items = len(instance.items)
    best_reward = float("-inf")
    best_partition = None

    for n_lots in range(2, 6):
        for restart in range(n_restarts):
            rng = np.random.default_rng(seed * 10000 + n_lots * 1000 + restart)
            part, reward = hill_climb(instance, sc, n_lots, rng, max_steps)
            if reward > best_reward:
                best_reward = reward
                best_partition = part

    assert best_partition is not None
    return best_partition, best_reward, best_partition.n_lots


def _process_instance(args: tuple[int, float, int, int]) -> dict:
    """Worker function for parallel SFT data generation."""
    i, coupling, n_restarts, max_steps = args
    sc = SupplierModelConfig(coupling_strength=coupling)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)

    seed = 500000 + i
    instance = gen.generate(seed=seed)
    n_cats = len({item.category for item in instance.items})
    rng = np.random.default_rng(501000 + i)

    # Evaluate baselines
    strategies = {
        "category": category_partition(instance),
        "geographic": geographic_partition(instance, n_lots=n_cats, rng=rng),
        "equal_size": equal_size_partition(instance, n_lots=n_cats),
    }

    baseline_best_name = ""
    baseline_best_reward = float("-inf")
    for name, part in strategies.items():
        result = evaluate_partition(part, instance, sc, seed=0)
        if result.reward > baseline_best_reward:
            baseline_best_reward = result.reward
            baseline_best_name = name

    # Search
    search_part, search_reward, search_lots = search_best_partition(
        instance, sc, seed=seed, n_restarts=n_restarts, max_steps=max_steps
    )

    if search_reward > baseline_best_reward:
        best_part = search_part
        best_reward = search_reward
        best_source = f"search_{search_lots}lots"
    else:
        best_part = strategies[baseline_best_name]
        best_reward = baseline_best_reward
        best_source = baseline_best_name

    prompt = serialize_instance(instance)
    completion = serialize_partition(best_part)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "metadata": {
            "instance_seed": seed,
            "best_source": best_source,
            "reward": best_reward,
            "baseline_best": baseline_best_reward,
            "baseline_strategy": baseline_best_name,
            "search_reward": search_reward,
            "search_lots": search_lots,
            "gap_vs_baseline": search_reward - baseline_best_reward,
        },
    }


def generate_sft_data(
    n_instances: int = 500,
    coupling: float = 1.0,
    output_path: str = "sft_data.jsonl",
    n_restarts: int = 20,
    max_steps: int = 100,
    n_workers: int = 8,
) -> None:
    """Generate SFT training data using search-optimized solutions."""
    t0 = time.monotonic()

    args_list = [(i, coupling, n_restarts, max_steps) for i in range(n_instances)]

    if n_workers > 1:
        multiprocessing.set_start_method("forkserver", force=True)
        with multiprocessing.Pool(n_workers) as pool:
            records = []
            for j, record in enumerate(pool.imap_unordered(_process_instance, args_list)):
                records.append(record)
                if (j + 1) % 10 == 0:
                    elapsed = time.monotonic() - t0
                    search_wins = sum(
                        1 for r in records if r["metadata"]["best_source"].startswith("search")
                    )
                    log.info(
                        "%d/%d done (%.1fs, %.0f/hr) | search wins: %d/%d",
                        j + 1,
                        n_instances,
                        elapsed,
                        (j + 1) / elapsed * 3600,
                        search_wins,
                        j + 1,
                    )
    else:
        records = [_process_instance(a) for a in args_list]

    # Sort by seed for reproducibility
    records.sort(key=lambda r: r["metadata"]["instance_seed"])

    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    elapsed = time.monotonic() - t0
    search_wins = sum(1 for r in records if r["metadata"]["best_source"].startswith("search"))
    total_gap = sum(
        r["metadata"]["gap_vs_baseline"]
        for r in records
        if r["metadata"]["best_source"].startswith("search")
    )

    log.info("Generated %d SFT examples in %.1fs → %s", len(records), elapsed, output_path)

    from collections import Counter

    source_counts = Counter(r["metadata"]["best_source"] for r in records)
    log.info("Source distribution: %s", dict(source_counts))
    log.info(
        "Search wins: %d/%d (%.1f%%)", search_wins, n_instances, search_wins / n_instances * 100
    )
    if search_wins > 0:
        log.info("Avg gap when search wins: +%.1f", total_gap / search_wins)

    rewards = [r["metadata"]["reward"] for r in records]
    log.info(
        "Reward stats: mean=%.1f, std=%.1f, min=%.1f, max=%.1f",
        np.mean(rewards),
        np.std(rewards),
        np.min(rewards),
        np.max(rewards),
    )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    generate_sft_data(n_instances=n, n_workers=workers)
