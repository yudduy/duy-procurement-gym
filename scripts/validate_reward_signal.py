# PLAN: Reward landscape validation — the go/no-go gate for RLVR viability.
#        Generates instances at 3 coupling levels, evaluates 5 baseline strategies,
#        runs each through verifier with multiple seeds, computes SNR/tau/gap.
# SOURCE: Deliberation convergence (Gemini + Claude) on validation approach.
"""Reward signal validation for RLVR viability.

Measures:
- SNR = (R_best - R_random) / std(noise) — signal vs simulation noise
- Kendall's tau — rank preservation under noise
- Reward gap — absolute separation between strategies

Go/no-go thresholds:
- SNR > 3.0 → reward signal is learnable
- Kendall's tau > 0.7 → strategy ranking survives noise
- Gap > 2 * std(noise) → strategies are distinguishable
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau

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


@dataclass(frozen=True)
class ValidationConfig:
    """Config for the validation experiment."""

    n_instances: int = 50
    n_seeds_per_eval: int = 10
    coupling_levels: tuple[float, ...] = (0.0, 0.5, 1.0)
    n_items: int = 20
    n_suppliers: int = 10
    n_categories: int = 3
    # Go/no-go thresholds
    snr_threshold: float = 3.0
    tau_threshold: float = 0.7
    gap_ratio_threshold: float = 2.0


def _generate_partitions(
    instance: ProcurementInstance,
    rng: np.random.Generator,
) -> dict[str, LotPartition]:
    """Generate all baseline partitions for an instance."""
    n_cats = len({item.category for item in instance.items})
    return {
        "category": category_partition(instance),
        "geographic": geographic_partition(instance, n_lots=n_cats, rng=rng),
        "equal_size": equal_size_partition(instance, n_lots=n_cats),
        "random": random_partition(instance, n_lots=n_cats, rng=rng),
        "single_item": single_item_partition(instance),
    }


def _evaluate_with_noise(
    partition: LotPartition,
    instance: ProcurementInstance,
    config: SupplierModelConfig,
    n_seeds: int,
    base_seed: int,
) -> list[float]:
    """Evaluate a partition across multiple seeds."""
    return [
        evaluate_partition(partition, instance, config, seed=base_seed + s).reward
        for s in range(n_seeds)
    ]


def run_validation(cfg: ValidationConfig) -> dict[str, object]:
    """Run the full reward signal validation."""
    results: dict[str, object] = {}

    for coupling in cfg.coupling_levels:
        log.info("=== coupling_strength=%.1f ===", coupling)
        level_key = f"coupling_{coupling:.1f}"

        supplier_config = SupplierModelConfig(coupling_strength=coupling)
        instance_config = InstanceConfig(
            n_items=cfg.n_items,
            n_suppliers=cfg.n_suppliers,
            n_categories=cfg.n_categories,
        )
        generator = TransportInstanceGenerator(
            instance_config=instance_config,
            supplier_config=supplier_config,
        )

        # Per-strategy rewards across all instances
        strategy_means: dict[str, list[float]] = {
            "category": [],
            "geographic": [],
            "equal_size": [],
            "random": [],
            "single_item": [],
        }
        # Per-instance noise measurements
        noise_stds: list[float] = []
        # Per-instance rank correlations
        taus: list[float] = []

        t0 = time.monotonic()
        for inst_idx in range(cfg.n_instances):
            instance = generator.generate(seed=1000 + inst_idx)
            part_rng = np.random.default_rng(2000 + inst_idx)
            partitions = _generate_partitions(instance, part_rng)

            # Evaluate each strategy with multiple seeds
            strategy_rewards: dict[str, list[float]] = {}
            for name, part in partitions.items():
                rewards = _evaluate_with_noise(
                    part,
                    instance,
                    supplier_config,
                    n_seeds=cfg.n_seeds_per_eval,
                    base_seed=3000 + inst_idx * 100,
                )
                strategy_rewards[name] = rewards
                strategy_means[name].append(float(np.mean(rewards)))

            # Noise: std across seeds for category partition (representative)
            cat_std = float(np.std(strategy_rewards["category"]))
            noise_stds.append(cat_std)

            # Rank correlation: true ranking (by mean) vs noisy ranking (single seed)
            true_means = [np.mean(strategy_rewards[s]) for s in partitions]
            single_seed = [strategy_rewards[s][0] for s in partitions]
            tau_val, _ = kendalltau(true_means, single_seed)
            if not np.isnan(tau_val):
                taus.append(float(tau_val))

            if (inst_idx + 1) % 10 == 0:
                log.info(
                    "  %d/%d instances done (%.1fs)",
                    inst_idx + 1,
                    cfg.n_instances,
                    time.monotonic() - t0,
                )

        elapsed = time.monotonic() - t0

        # Compute metrics
        mean_noise_std = float(np.mean(noise_stds)) if noise_stds else 0.0
        best_strategy_mean = max(float(np.mean(strategy_means[s])) for s in strategy_means)
        random_mean = float(np.mean(strategy_means["random"]))
        gap = best_strategy_mean - random_mean
        snr = gap / mean_noise_std if mean_noise_std > 0 else float("inf")
        mean_tau = float(np.mean(taus)) if taus else 0.0
        gap_ratio = gap / mean_noise_std if mean_noise_std > 0 else float("inf")

        # Per-strategy summary
        strategy_summary = {}
        for name in strategy_means:
            arr = np.array(strategy_means[name])
            strategy_summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

        # Find best strategy
        best_name = max(strategy_summary, key=lambda s: strategy_summary[s]["mean"])

        level_result = {
            "coupling_strength": coupling,
            "n_instances": cfg.n_instances,
            "n_seeds_per_eval": cfg.n_seeds_per_eval,
            "elapsed_s": round(elapsed, 1),
            "metrics": {
                "snr": round(snr, 2),
                "kendall_tau": round(mean_tau, 3),
                "gap_best_vs_random": round(gap, 2),
                "gap_ratio": round(gap_ratio, 2),
                "mean_noise_std": round(mean_noise_std, 2),
            },
            "thresholds": {
                "snr_pass": snr > cfg.snr_threshold,
                "tau_pass": mean_tau > cfg.tau_threshold,
                "gap_pass": gap_ratio > cfg.gap_ratio_threshold,
            },
            "best_strategy": best_name,
            "strategies": strategy_summary,
        }

        all_pass = all(level_result["thresholds"].values())
        level_result["verdict"] = "GO" if all_pass else "NO-GO"

        log.info(
            "  SNR=%.2f (>%.1f? %s)",
            snr,
            cfg.snr_threshold,
            "PASS" if level_result["thresholds"]["snr_pass"] else "FAIL",
        )
        log.info(
            "  tau=%.3f (>%.1f? %s)",
            mean_tau,
            cfg.tau_threshold,
            "PASS" if level_result["thresholds"]["tau_pass"] else "FAIL",
        )
        log.info(
            "  gap_ratio=%.2f (>%.1f? %s)",
            gap_ratio,
            cfg.gap_ratio_threshold,
            "PASS" if level_result["thresholds"]["gap_pass"] else "FAIL",
        )
        log.info(
            "  best=%s (mean=%.1f) vs random (mean=%.1f)",
            best_name,
            best_strategy_mean,
            random_mean,
        )
        log.info("  VERDICT: %s", level_result["verdict"])

        results[level_key] = level_result

    # Overall verdict
    verdicts = [results[k]["verdict"] for k in results]  # type: ignore[union-attr]
    overall = "GO" if all(v == "GO" for v in verdicts) else "NO-GO"
    results["overall_verdict"] = overall

    return results


def main() -> None:
    cfg = ValidationConfig()
    log.info("Starting reward signal validation")
    log.info(
        "Config: %d instances × %d seeds × %d coupling levels",
        cfg.n_instances,
        cfg.n_seeds_per_eval,
        len(cfg.coupling_levels),
    )

    results = run_validation(cfg)

    # Print summary
    print("\n" + "=" * 60)
    print("REWARD SIGNAL VALIDATION RESULTS")
    print("=" * 60)

    for coupling in cfg.coupling_levels:
        key = f"coupling_{coupling:.1f}"
        r = results[key]
        m = r["metrics"]  # type: ignore[union-attr]
        print(f"\n--- coupling_strength={coupling} ---")
        print(f"  SNR:       {m['snr']:>8.2f}  (threshold: {cfg.snr_threshold})")
        print(f"  Tau:       {m['kendall_tau']:>8.3f}  (threshold: {cfg.tau_threshold})")
        print(f"  Gap ratio: {m['gap_ratio']:>8.2f}  (threshold: {cfg.gap_ratio_threshold})")
        print(f"  Noise std: {m['mean_noise_std']:>8.2f}")
        print(f"  Best:      {r['best_strategy']}")  # type: ignore[union-attr]
        print(f"  Verdict:   {r['verdict']}")  # type: ignore[union-attr]

        # Strategy table
        print(f"\n  {'Strategy':<15} {'Mean':>10} {'Std':>10}")
        for name, stats in r["strategies"].items():  # type: ignore[union-attr]
            print(f"  {name:<15} {stats['mean']:>10.1f} {stats['std']:>10.1f}")

    print(f"\n{'=' * 60}")
    print(f"OVERALL VERDICT: {results['overall_verdict']}")
    print(f"{'=' * 60}")

    # Save JSON
    out_path = "validation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {out_path}")

    sys.exit(0 if results["overall_verdict"] == "GO" else 1)


if __name__ == "__main__":
    main()
