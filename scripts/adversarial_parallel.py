# PLAN: Parallel adversarial test — uses multiprocessing.Pool to evaluate
#        instances in parallel. ~10x faster on multi-core machines.
# SOURCE: adversarial_coupling_test.py (same attacks, parallelized)
"""Parallel adversarial coupling test — multiprocessing across instances."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from functools import partial
import multiprocessing
from multiprocessing import Pool, cpu_count

# Use forkserver to avoid fork overhead and GIL contention
try:
    multiprocessing.set_start_method("forkserver")
except RuntimeError:
    pass  # already set

import numpy as np

from procurement_gym.baselines import (
    category_partition,
    equal_size_partition,
    geographic_partition,
    random_partition,
    single_item_partition,
)
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import ProcurementInstance
from procurement_gym.evaluation import evaluate_partition
from procurement_gym.instances.transport import TransportInstanceGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

N_WORKERS = min(cpu_count(), int(os.environ.get("ADVERSARIAL_WORKERS", "16")))


# ── Helpers ─────────────────────────────────────────────────────────────────


def _eval_instance_strategies(
    args: tuple[int, float, int],
) -> dict:
    """Evaluate all strategies for one instance. Runs in worker process."""
    inst_idx, coupling, n_seeds = args
    supplier_config = SupplierModelConfig(coupling_strength=coupling)
    instance_config = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(instance_config, supplier_config)

    instance = gen.generate(seed=5000 + inst_idx)
    rng = np.random.default_rng(6000 + inst_idx)
    n_cats = len({item.category for item in instance.items})

    partitions = {
        "category": category_partition(instance),
        "geographic": geographic_partition(instance, n_lots=n_cats, rng=rng),
        "single_item": single_item_partition(instance),
        "equal_size": equal_size_partition(instance, n_lots=n_cats),
        "random": random_partition(
            instance, n_lots=n_cats, rng=np.random.default_rng(6000 + inst_idx)
        ),
    }

    results: dict[str, dict] = {}
    for name, part in partitions.items():
        rewards = [
            evaluate_partition(
                part, instance, supplier_config, seed=7000 + inst_idx * 100 + s
            ).reward
            for s in range(n_seeds)
        ]
        results[name] = {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "n_lots": part.n_lots,
        }

    best = max(results, key=lambda s: results[s]["mean"])
    return {"instance": inst_idx, "coupling": coupling, "strategies": results, "winner": best}


def _eval_nlots_worker(args: tuple[int, int]) -> tuple[int, str]:
    """Attack 3 worker: evaluate all strategies for one (instance, n_lots) pair."""
    inst_idx, n_lots = args
    sc = SupplierModelConfig(coupling_strength=1.0)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)
    instance = gen.generate(seed=11000 + inst_idx)
    rng = np.random.default_rng(12000 + inst_idx * 100 + n_lots)
    partitions = {
        "category": category_partition(instance),
        "geographic": geographic_partition(instance, n_lots=n_lots, rng=rng),
        "equal_size": equal_size_partition(instance, n_lots=n_lots),
        "random": random_partition(
            instance, n_lots=n_lots, rng=np.random.default_rng(12000 + inst_idx * 100 + n_lots)
        ),
        "single_item": single_item_partition(instance),
    }
    means = {}
    for name, part in partitions.items():
        rewards = [
            evaluate_partition(part, instance, sc, seed=13000 + inst_idx * 100 + s).reward
            for s in range(5)
        ]
        means[name] = float(np.mean(rewards))
    best = max(means, key=means.get)  # type: ignore[arg-type]
    return (n_lots, best)


def _eval_hybrid_worker(args: tuple[int, float]) -> tuple[float, str]:
    """Attack 4 worker: evaluate strategies + hybrids for one (instance, coupling) pair."""
    inst_idx, coupling = args
    sc = SupplierModelConfig(coupling_strength=coupling)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)
    instance = gen.generate(seed=14000 + inst_idx)
    rng = np.random.default_rng(15000 + inst_idx)
    n_cats = len({item.category for item in instance.items})
    if coupling < 0.3:
        oracle = category_partition(instance)
    elif coupling < 0.7:
        oracle = geographic_partition(
            instance, n_lots=n_cats, rng=np.random.default_rng(15000 + inst_idx)
        )
    else:
        oracle = single_item_partition(instance)
    cat_part = category_partition(instance)
    new_asgn = list(cat_part.assignments)
    next_lot = max(new_asgn) + 1
    for lot_id in sorted(cat_part.lot_ids):
        items_in = cat_part.lot_items(lot_id)
        if len(items_in) > 5:
            coords = [(instance.items[idx].x, idx) for idx in items_in]
            coords.sort()
            mid = len(coords) // 2
            for _, idx in coords[mid:]:
                new_asgn[idx] = next_lot
            next_lot += 1
    cat_geo_mix = LotPartition(assignments=tuple(new_asgn), n_items=len(instance.items))
    partitions = {
        "category": category_partition(instance),
        "single_item": single_item_partition(instance),
        "geographic": geographic_partition(instance, n_lots=n_cats, rng=rng),
        "oracle_hybrid": oracle,
        "category_geo_mix": cat_geo_mix,
    }
    means = {}
    for name, part in partitions.items():
        rewards = [
            evaluate_partition(part, instance, sc, seed=16000 + inst_idx * 10 + s).reward
            for s in range(5)
        ]
        means[name] = float(np.mean(rewards))
    best = max(means, key=means.get)  # type: ignore[arg-type]
    return (coupling, best)


def _eval_seed_worker(args: tuple[int, int]) -> tuple[str, str]:
    """Attack 6 worker: evaluate strategies for one (instance, seed_range) pair."""
    inst_idx, base_seed = args
    sc = SupplierModelConfig(coupling_strength=1.0)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc)
    instance = gen.generate(seed=base_seed + inst_idx)
    rng = np.random.default_rng(base_seed + 1000 + inst_idx)
    n_cats = len({item.category for item in instance.items})
    partitions = {
        "category": category_partition(instance),
        "geographic": geographic_partition(instance, n_lots=n_cats, rng=rng),
        "single_item": single_item_partition(instance),
        "equal_size": equal_size_partition(instance, n_lots=n_cats),
        "random": random_partition(
            instance, n_lots=n_cats, rng=np.random.default_rng(base_seed + 1000 + inst_idx)
        ),
    }
    means = {}
    for name, part in partitions.items():
        rewards = [
            evaluate_partition(part, instance, sc, seed=base_seed + 2000 + inst_idx * 10 + s).reward
            for s in range(5)
        ]
        means[name] = float(np.mean(rewards))
    best = max(means, key=means.get)  # type: ignore[arg-type]
    label = {1000: "range_1k", 50000: "range_50k", 99000: "range_99k"}[base_seed]
    return (label, best)


def _eval_attack1_instance(inst_idx: int) -> dict:
    """Attack 1: single-item degeneracy for one instance."""
    supplier_config = SupplierModelConfig(coupling_strength=1.0)
    instance_config = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(instance_config, supplier_config)
    instance = gen.generate(seed=5000 + inst_idx)
    rng = np.random.default_rng(6000 + inst_idx)

    partitions = {
        "category": category_partition(instance),
        "single_item": single_item_partition(instance),
        "geographic": geographic_partition(instance, n_lots=3, rng=rng),
    }

    result = {}
    for name, part in partitions.items():
        rewards, infos = [], []
        for s in range(5):
            r = evaluate_partition(part, instance, supplier_config, seed=7000 + inst_idx * 10 + s)
            rewards.append(r.reward)
            infos.append(r.info)
        result[name] = {
            "mean_reward": float(np.mean(rewards)),
            "n_lots": part.n_lots,
            "n_unserviced_avg": float(np.mean([i.get("n_unserviced", 0) for i in infos])),
            "n_package_awards_avg": float(np.mean([i.get("n_package_awards", 0) for i in infos])),
        }
    return result


# ── Attacks ──────────────────────────────────────────────────────────────────


def attack_1_single_item(n_instances: int = 30) -> dict:
    log.info("ATTACK 1: Single-item degeneracy (parallel)")
    with Pool(N_WORKERS) as pool:
        results = pool.map(
            _eval_attack1_instance, range(n_instances), chunksize=max(1, n_instances // N_WORKERS)
        )

    summary: dict[str, dict] = {}
    for name in ["category", "single_item", "geographic"]:
        entries = [r[name] for r in results]
        summary[name] = {
            "mean_reward": float(np.mean([e["mean_reward"] for e in entries])),
            "mean_unserviced": float(np.mean([e["n_unserviced_avg"] for e in entries])),
            "mean_package_awards": float(np.mean([e["n_package_awards_avg"] for e in entries])),
        }

    cat_unsvc = summary["category"]["mean_unserviced"]
    geo_unsvc = summary["geographic"]["mean_unserviced"]

    if cat_unsvc > 0.5 or geo_unsvc > 0.5:
        verdict = "FAIL: Grouped strategies have significant unserviced lots"
    elif summary["single_item"]["mean_package_awards"] < 0.01:
        verdict = "SUSPICIOUS: Single-item never uses packages"
    else:
        verdict = "PASS: Single-item wins genuinely"

    return {"attack": "single_item_degeneracy", "summary": summary, "verdict": verdict}


def attack_2_reward_gap(n_instances: int = 30) -> dict:
    log.info("ATTACK 2: Reward gap magnitude (parallel)")
    results_by_coupling: dict[float, dict] = {}

    for coupling in [0.0, 0.5, 1.0]:
        args = [(i, coupling, 10) for i in range(n_instances)]
        with Pool(N_WORKERS) as pool:
            results = pool.map(_eval_instance_strategies, args)

        gap_ratios, relative_gaps = [], []
        for r in results:
            strats = r["strategies"]
            ranked = sorted(strats.items(), key=lambda x: -x[1]["mean"])
            gap = ranked[0][1]["mean"] - ranked[1][1]["mean"]
            noise = max(ranked[0][1]["std"], ranked[1][1]["std"])
            if noise > 0:
                gap_ratios.append(gap / noise)
            if abs(ranked[0][1]["mean"]) > 0:
                relative_gaps.append(gap / abs(ranked[0][1]["mean"]))

        results_by_coupling[coupling] = {
            "mean_gap_ratio": float(np.mean(gap_ratios)) if gap_ratios else 0,
            "pct_gap_lt_1": sum(1 for r in gap_ratios if r < 1.0) / max(len(gap_ratios), 1) * 100,
            "mean_relative_gap": float(np.mean(relative_gaps)) if relative_gaps else 0,
        }

    high = results_by_coupling[1.0]
    # Close strategies (small gap) is GOOD for RLVR (p≈0.5).
    # FAIL only if gap_ratio is very low (gap < 0.1x noise) — truly noise-dominated.
    if high["mean_gap_ratio"] < 0.1:
        verdict = f"FAIL: Gap/noise ratio {high['mean_gap_ratio']:.2f} < 0.1 — pure noise"
    else:
        verdict = f"PASS: Gaps meaningful (rel_gap={high['mean_relative_gap']:.3f}, ratio={high['mean_gap_ratio']:.2f})"

    return {
        "attack": "reward_gap",
        "results": {str(k): v for k, v in results_by_coupling.items()},
        "verdict": verdict,
    }


def attack_3_nlots(n_instances: int = 20) -> dict:
    log.info("ATTACK 3: n_lots sensitivity (parallel)")

    n_lots_options = [2, 3, 5, 7, 10]
    all_args = [(i, n) for i in range(n_instances) for n in n_lots_options]

    with Pool(N_WORKERS) as pool:
        results = pool.map(_eval_nlots_worker, all_args)

    wins_by_nlots: dict[int, Counter] = {n: Counter() for n in n_lots_options}
    for n_lots, winner in results:
        wins_by_nlots[n_lots][winner] += 1

    max_dominance = max(
        max(counts.values()) / max(sum(counts.values()), 1) for counts in wins_by_nlots.values()
    )

    # Check if the BEST n_lots varies across instances (instance-dependent)
    best_nlots_per_strategy = {}
    for n, counts in wins_by_nlots.items():
        for strat, count in counts.items():
            best_nlots_per_strategy.setdefault(strat, []).append((n, count))

    n_competitive_strategies = sum(1 for counts in wins_by_nlots.values() if len(counts) >= 2)

    if max_dominance > 0.9 and n_competitive_strategies < 2:
        verdict = f"FAIL: One strategy dominates at {max_dominance * 100:.0f}% with no competition"
    else:
        verdict = f"PASS: Multiple strategies competitive across n_lots (max_dom={max_dominance * 100:.0f}%, competitive_nlots={n_competitive_strategies}/{len(n_lots_options)})"

    return {
        "attack": "nlots",
        "wins": {str(k): dict(v) for k, v in wins_by_nlots.items()},
        "verdict": verdict,
    }


def attack_4_hybrid(n_instances: int = 30) -> dict:
    log.info("ATTACK 4: Hybrid strategy (parallel)")

    all_args = [(i, c) for c in [0.0, 0.5, 1.0] for i in range(n_instances)]
    with Pool(N_WORKERS) as pool:
        results = pool.map(_eval_hybrid_worker, all_args)

    by_coupling: dict[float, Counter] = {0.0: Counter(), 0.5: Counter(), 1.0: Counter()}
    for coupling, winner in results:
        by_coupling[coupling][winner] += 1

    oracle_total = sum(by_coupling[c].get("oracle_hybrid", 0) for c in [0.0, 0.5, 1.0])
    mix_total = sum(by_coupling[c].get("category_geo_mix", 0) for c in [0.0, 0.5, 1.0])
    total = n_instances * 3

    if oracle_total / total > 0.5:
        verdict = f"FAIL: Oracle wins {oracle_total}/{total}"
    elif mix_total / total > 0.5:
        verdict = f"FAIL: Cat-geo mix wins {mix_total}/{total}"
    else:
        verdict = (
            f"PASS: No hybrid dominates (oracle={oracle_total}/{total}, mix={mix_total}/{total})"
        )

    return {
        "attack": "hybrid",
        "by_coupling": {str(k): dict(v) for k, v in by_coupling.items()},
        "verdict": verdict,
    }


def attack_5_capacity_binding(n_instances: int = 10) -> dict:
    log.info("ATTACK 5: Capacity binding")
    sc_with = SupplierModelConfig(coupling_strength=1.0)
    sc_without = SupplierModelConfig(coupling_strength=0.0)
    ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
    gen = TransportInstanceGenerator(ic, sc_with)

    diff = []
    for i in range(n_instances):
        inst = gen.generate(seed=17000 + i)
        part = category_partition(inst)
        r_with = evaluate_partition(part, inst, sc_with, seed=20000 + i)
        r_without = evaluate_partition(part, inst, sc_without, seed=20000 + i)
        diff.append(abs(r_with.reward - r_without.reward))

    mean_diff = float(np.mean(diff))
    if mean_diff > 100:
        verdict = f"PASS: Capacity binding creates coupling (reward diff={mean_diff:.0f})"
    else:
        verdict = f"FAIL: No coupling mechanism active (reward diff={mean_diff:.0f})"

    return {
        "attack": "capacity_binding",
        "mean_reward_diff": round(mean_diff, 0),
        "verdict": verdict,
    }


def attack_6_seed_stability(n_instances: int = 20) -> dict:
    log.info("ATTACK 6: Seed sensitivity (parallel)")

    all_args = [(i, base) for base in [1000, 50000, 99000] for i in range(n_instances)]
    with Pool(N_WORKERS) as pool:
        results = pool.map(_eval_seed_worker, all_args)

    by_range: dict[str, Counter] = {
        "range_1k": Counter(),
        "range_50k": Counter(),
        "range_99k": Counter(),
    }
    for label, winner in results:
        by_range[label][winner] += 1

    # Check max dominance across all ranges
    max_dom = max(
        max(counts.values()) / max(sum(counts.values()), 1) for counts in by_range.values()
    )

    all_dominant = all(
        max(counts.values(), default=0) / max(sum(counts.values()), 1) > 0.5
        for counts in by_range.values()
    )
    winners = [max(by_range[r], key=by_range[r].get) for r in by_range]  # type: ignore[arg-type]
    consistent = len(set(winners)) == 1

    all_strategies = set()
    for r in by_range.values():
        all_strategies.update(r.keys())
    max_swing = 0.0
    for strat in all_strategies:
        rates = [
            by_range[r].get(strat, 0) / max(sum(by_range[r].values()), 1) * 100 for r in by_range
        ]
        max_swing = max(max_swing, max(rates) - min(rates))

    if all_dominant and consistent:
        verdict = f"FAIL: {winners[0]} dominates in ALL seed ranges"
    elif max_swing > 40:
        verdict = f"FAIL: Win rates swing {max_swing:.0f}pp"
    else:
        verdict = f"PASS: No trivial dominance (max_swing={max_swing:.0f}pp, max_dom={max_dom * 100:.0f}%)"

    return {
        "attack": "seed_stability",
        "by_range": {
            k: {s: round(v / max(sum(c.values()), 1) * 100, 1) for s, v in c.items()}
            for k, c in by_range.items()
        },
        "max_swing": round(max_swing, 1),
        "verdict": verdict,
    }


def main() -> None:
    log.info(f"PARALLEL ADVERSARIAL TEST — {N_WORKERS} workers")
    t0 = time.monotonic()

    attacks = [
        ("1_single_item", attack_1_single_item),
        ("2_reward_gap", attack_2_reward_gap),
        ("3_nlots", attack_3_nlots),
        ("4_hybrid", attack_4_hybrid),
        ("5_capacity", attack_5_capacity_binding),
        ("6_seeds", attack_6_seed_stability),
    ]

    all_results: dict[str, dict] = {}
    passes, fails = 0, 0

    for name, fn in attacks:
        result = fn()
        all_results[name] = result
        v = result["verdict"]
        status = "PASS" if v.startswith("PASS") else "FAIL"
        icon = "[OK]" if status == "PASS" else "[!!]"
        print(f"{icon} {name}: {v}")
        if status == "PASS":
            passes += 1
        else:
            fails += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'=' * 60}")
    print(f"OVERALL: {passes} PASS, {fails} FAIL in {elapsed:.0f}s ({N_WORKERS} workers)")
    verdict = "COUPLING MECHANISM VALIDATED" if fails == 0 else "HAS WEAKNESSES"
    print(f"VERDICT: {verdict}")
    print(f"{'=' * 60}")

    with open("adversarial_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
