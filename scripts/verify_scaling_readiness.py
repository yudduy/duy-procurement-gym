"""Scaling-readiness audit for the procurement verifier/training substrate.

This script separates four questions that were previously conflated:
1. Is the high-coupling verifier economically sane and deterministic?
2. Is there meaningful strategy-level signal to learn from?
3. Is the verifier operationally tractable under degenerate exploration?
4. Is the sim-to-real calibration bridge implemented?

The output is a JSON report with pass/fail gates plus a concise terminal summary.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from procurement_gym.baselines import (
    category_partition,
    equal_size_partition,
    geographic_partition,
    random_partition,
    single_item_partition,
)
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.evaluation import evaluate_partition
from procurement_gym.instances.transport import TransportInstanceGenerator
from generate_sft_data import search_best_partition


@dataclass(frozen=True)
class AuditConfig:
    coupling_strength: float = 1.0
    n_items: int = 20
    n_suppliers: int = 10
    n_categories: int = 3
    determinism_instances: int = 5
    signal_instances: int = 30
    runtime_curve_instances: int = 3
    search_probe_instances: int = 8
    search_probe_restarts: int = 3
    search_probe_max_steps: int = 15
    # Gates
    min_mean_relative_gap: float = 0.02
    min_package_award_fraction: float = 0.25
    max_single_baseline_win_rate: float = 0.80
    max_worst_case_p95_wall_s: float = 1.0
    min_search_win_rate_gt_2pct: float = 0.20


def _make_generator(cfg: AuditConfig) -> TransportInstanceGenerator:
    sc = SupplierModelConfig(coupling_strength=cfg.coupling_strength)
    ic = InstanceConfig(
        n_items=cfg.n_items,
        n_suppliers=cfg.n_suppliers,
        n_categories=cfg.n_categories,
    )
    return TransportInstanceGenerator(ic, sc)


def _strategy_partitions(instance, rng: np.random.Generator) -> dict[str, object]:
    n_cats = len({item.category for item in instance.items})
    return {
        "category": category_partition(instance),
        "geographic": geographic_partition(instance, n_lots=n_cats, rng=rng),
        "equal_size": equal_size_partition(instance, n_lots=n_cats),
        "random": random_partition(instance, n_lots=n_cats, rng=rng),
        "single_item": single_item_partition(instance),
    }


def audit_determinism_and_sanity(cfg: AuditConfig) -> dict[str, object]:
    sc = SupplierModelConfig(coupling_strength=cfg.coupling_strength)
    gen = _make_generator(cfg)

    max_reward_diff = 0.0
    negative_total_cost_evals = 0
    total_evals = 0

    for i in range(cfg.determinism_instances):
        instance = gen.generate(seed=30_000 + i)
        rng = np.random.default_rng(40_000 + i)
        for _, part in _strategy_partitions(instance, rng).items():
            r0 = evaluate_partition(part, instance, sc, seed=0)
            r1 = evaluate_partition(part, instance, sc, seed=1)
            max_reward_diff = max(max_reward_diff, abs(r0.reward - r1.reward))
            negative_total_cost_evals += int(r0.info.get("total_cost", 0.0) < 0.0)
            negative_total_cost_evals += int(r1.info.get("total_cost", 0.0) < 0.0)
            total_evals += 2

    passed = max_reward_diff == 0.0 and negative_total_cost_evals == 0
    return {
        "max_reward_diff_across_seeds": max_reward_diff,
        "negative_total_cost_evals": negative_total_cost_evals,
        "total_evals": total_evals,
        "passed": passed,
    }


def audit_signal_and_mechanism(cfg: AuditConfig) -> dict[str, object]:
    sc = SupplierModelConfig(coupling_strength=cfg.coupling_strength)
    gen = _make_generator(cfg)

    win_counts: Counter[str] = Counter()
    rewards_by_strategy: dict[str, list[float]] = defaultdict(list)
    status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    package_bids: list[int] = []
    package_awards: list[int] = []
    top_two_gaps: list[float] = []
    top_two_relative_gaps: list[float] = []

    for i in range(cfg.signal_instances):
        instance = gen.generate(seed=10_000 + i)
        rng = np.random.default_rng(20_000 + i)
        ranked: list[tuple[str, float]] = []
        for name, part in _strategy_partitions(instance, rng).items():
            result = evaluate_partition(part, instance, sc, seed=0)
            rewards_by_strategy[name].append(result.reward)
            status_counts[name][result.info.get("status", "UNKNOWN")] += 1
            package_bids.append(int(result.info.get("n_package_bids", 0)))
            package_awards.append(int(result.info.get("n_package_awards", 0)))
            ranked.append((name, result.reward))

        ranked.sort(key=lambda x: -x[1])
        win_counts[ranked[0][0]] += 1
        gap = ranked[0][1] - ranked[1][1]
        top_two_gaps.append(gap)
        top_two_relative_gaps.append(gap / abs(ranked[0][1]))

    mean_rewards = {k: float(np.mean(v)) for k, v in rewards_by_strategy.items()}
    win_rates = {k: v / cfg.signal_instances for k, v in win_counts.items()}
    package_award_fraction = float(np.mean([x > 0 for x in package_awards]))
    dominant_win_rate = max(win_rates.values()) if win_rates else 0.0

    passed = (
        float(np.mean(top_two_relative_gaps)) >= cfg.min_mean_relative_gap
        and package_award_fraction >= cfg.min_package_award_fraction
        and dominant_win_rate <= cfg.max_single_baseline_win_rate
    )

    return {
        "win_counts": dict(win_counts),
        "win_rates": win_rates,
        "mean_rewards": mean_rewards,
        "status_counts": {k: dict(v) for k, v in status_counts.items()},
        "mean_top1_top2_gap": float(np.mean(top_two_gaps)),
        "median_top1_top2_gap": float(np.median(top_two_gaps)),
        "mean_relative_gap": float(np.mean(top_two_relative_gaps)),
        "package_award_fraction": package_award_fraction,
        "mean_package_bids": float(np.mean(package_bids)),
        "p95_package_bids": float(np.percentile(package_bids, 95)),
        "max_package_bids": int(max(package_bids)) if package_bids else 0,
        "mean_package_awards": float(np.mean(package_awards)),
        "p95_package_awards": float(np.percentile(package_awards, 95)),
        "max_single_baseline_win_rate": dominant_win_rate,
        "passed": passed,
    }


def audit_runtime_curve(cfg: AuditConfig) -> dict[str, object]:
    sc = SupplierModelConfig(coupling_strength=cfg.coupling_strength)
    gen = _make_generator(cfg)

    curve: dict[int, dict[str, list[float]]] = {
        n_lots: {"wall_s": [], "package_bids": [], "package_awards": []}
        for n_lots in (2, 3, 5, 10, cfg.n_items)
    }

    for i in range(cfg.runtime_curve_instances):
        instance = gen.generate(seed=50_000 + i)
        for n_lots in curve:
            if n_lots == cfg.n_items:
                partition = single_item_partition(instance)
            else:
                partition = equal_size_partition(instance, n_lots=n_lots)
            t0 = time.perf_counter()
            result = evaluate_partition(partition, instance, sc, seed=0)
            wall_s = time.perf_counter() - t0
            curve[n_lots]["wall_s"].append(wall_s)
            curve[n_lots]["package_bids"].append(int(result.info.get("n_package_bids", 0)))
            curve[n_lots]["package_awards"].append(int(result.info.get("n_package_awards", 0)))

    summary = {}
    for n_lots, values in curve.items():
        wall_arr = np.array(values["wall_s"])
        bid_arr = np.array(values["package_bids"])
        award_arr = np.array(values["package_awards"])
        summary[str(n_lots)] = {
            "mean_wall_s": float(np.mean(wall_arr)),
            "p95_wall_s": float(np.percentile(wall_arr, 95)),
            "max_wall_s": float(np.max(wall_arr)),
            "mean_package_bids": float(np.mean(bid_arr)),
            "max_package_bids": int(np.max(bid_arr)),
            "mean_package_awards": float(np.mean(award_arr)),
        }

    worst_case = summary[str(cfg.n_items)]
    passed = worst_case["p95_wall_s"] <= cfg.max_worst_case_p95_wall_s
    return {
        "curve": summary,
        "worst_case_n_lots": cfg.n_items,
        "passed": passed,
    }


def audit_search_uplift(cfg: AuditConfig) -> dict[str, object]:
    sc = SupplierModelConfig(coupling_strength=cfg.coupling_strength)
    gen = _make_generator(cfg)

    abs_gain_vs_category: list[float] = []
    abs_gain_vs_best_baseline: list[float] = []
    rel_gain_vs_best_baseline: list[float] = []

    for i in range(cfg.search_probe_instances):
        instance = gen.generate(seed=60_000 + i)
        rng = np.random.default_rng(70_000 + i)
        partitions = _strategy_partitions(instance, rng)
        baseline_scores = {
            name: evaluate_partition(part, instance, sc, seed=0).reward
            for name, part in partitions.items()
            if name != "single_item"
        }
        best_baseline = max(baseline_scores.values())
        category_score = baseline_scores["category"]

        _, search_reward, _ = search_best_partition(
            instance,
            sc,
            seed=80_000 + i,
            n_restarts=cfg.search_probe_restarts,
            max_steps=cfg.search_probe_max_steps,
        )
        abs_gain_vs_category.append(search_reward - category_score)
        abs_gain_vs_best_baseline.append(search_reward - best_baseline)
        rel_gain_vs_best_baseline.append(
            (search_reward - best_baseline) / max(abs(best_baseline), 1.0)
        )

    win_rate_gt_zero = float(np.mean([gain > 0 for gain in abs_gain_vs_best_baseline]))
    win_rate_gt_2pct = float(np.mean([gain > 0.02 for gain in rel_gain_vs_best_baseline]))
    passed = win_rate_gt_2pct >= cfg.min_search_win_rate_gt_2pct

    return {
        "mean_abs_gain_vs_category": float(np.mean(abs_gain_vs_category)),
        "median_abs_gain_vs_category": float(np.median(abs_gain_vs_category)),
        "mean_abs_gain_vs_best_baseline": float(np.mean(abs_gain_vs_best_baseline)),
        "median_abs_gain_vs_best_baseline": float(np.median(abs_gain_vs_best_baseline)),
        "search_win_rate_gt_zero": win_rate_gt_zero,
        "search_win_rate_gt_2pct": win_rate_gt_2pct,
        "passed": passed,
    }


def audit_calibration_bridge() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    required = {
        "gpv_estimator": root / "src/procurement_gym/suppliers/gpv.py",
        "calibrated_generator": root / "src/procurement_gym/instances/calibrated.py",
        "calibration_script": root / "scripts/calibrate.py",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return {
        "required_artifacts": {k: str(v) for k, v in required.items()},
        "missing": missing,
        "passed": not missing,
    }


def run_audit(cfg: AuditConfig) -> dict[str, object]:
    determinism = audit_determinism_and_sanity(cfg)
    signal = audit_signal_and_mechanism(cfg)
    runtime = audit_runtime_curve(cfg)
    search_uplift = audit_search_uplift(cfg)
    calibration = audit_calibration_bridge()

    results = {
        "config": asdict(cfg),
        "determinism_and_sanity": determinism,
        "signal_and_mechanism": signal,
        "runtime_curve": runtime,
        "search_uplift": search_uplift,
        "calibration_bridge": calibration,
    }
    results["oracle_correctness_ready"] = bool(determinism["passed"])
    results["simulated_training_ready"] = bool(
        determinism["passed"] and signal["passed"] and runtime["passed"] and search_uplift["passed"]
    )
    results["production_readiness"] = bool(
        determinism["passed"] and signal["passed"] and runtime["passed"] and calibration["passed"]
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Scaling-readiness audit for procurement RL.")
    parser.add_argument("--output", default="results/scaling_readiness.json")
    parser.add_argument("--signal-instances", type=int, default=30)
    parser.add_argument("--determinism-instances", type=int, default=5)
    parser.add_argument("--runtime-curve-instances", type=int, default=3)
    args = parser.parse_args()

    cfg = AuditConfig(
        signal_instances=args.signal_instances,
        determinism_instances=args.determinism_instances,
        runtime_curve_instances=args.runtime_curve_instances,
    )
    results = run_audit(cfg)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print("SCALING READINESS AUDIT")
    print(f"oracle_correctness_ready: {results['oracle_correctness_ready']}")
    print(f"simulated_training_ready: {results['simulated_training_ready']}")
    print(f"production_readiness: {results['production_readiness']}")
    print(
        "gates:",
        {
            "determinism_and_sanity": results["determinism_and_sanity"]["passed"],
            "signal_and_mechanism": results["signal_and_mechanism"]["passed"],
            "runtime_curve": results["runtime_curve"]["passed"],
            "search_uplift": results["search_uplift"]["passed"],
            "calibration_bridge": results["calibration_bridge"]["passed"],
        },
    )
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
