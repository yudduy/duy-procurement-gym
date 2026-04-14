# PLAN: Standalone reward evaluator — extracts the terminal reward logic from env.py
#        so partitions can be scored without running the full Gymnasium step loop.
#        Needed for reward landscape validation (SNR, Kendall's tau).
# ADAPTED FROM: procurement_gym/env.py _compute_terminal_reward (lines 151-233)
"""Standalone partition evaluation — reward computation without the env loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import ProcurementInstance
from procurement_gym.solver.ortools_solver import ORToolsWDPSolver
from procurement_gym.suppliers.simulator import ThreeLayerSupplierSimulator


@dataclass(frozen=True)
class EvalResult:
    """Result of evaluating a partition."""

    reward: float
    info: dict[str, Any] = field(default_factory=dict)


def evaluate_partition(
    partition: LotPartition,
    instance: ProcurementInstance,
    config: SupplierModelConfig,
    seed: int,
    area_size: float = 100.0,
    solver_timeout_s: float = 30.0,
) -> EvalResult:
    """Evaluate a partition's buyer welfare via supplier simulation + WDP solver.

    This is the same computation as env._compute_terminal_reward but callable
    standalone without the Gymnasium env loop.
    """
    rng = np.random.default_rng(seed)
    simulator = ThreeLayerSupplierSimulator(config=config, area_size=area_size)
    solver = ORToolsWDPSolver(timeout_s=solver_timeout_s)

    sim_result = simulator.simulate(partition, instance, rng)

    # Build solver inputs
    bids: dict[int, dict[int, float]] = {}
    entrants: dict[int, frozenset[int]] = {}
    volumes: dict[int, float] = {}
    capacities = {s.id: s.capacity for s in instance.suppliers}

    n_unserviced = 0
    for lot_id in sorted(partition.lot_ids):
        lot_bids = sim_result.bids.get(lot_id, {})
        lot_entrants = sim_result.entrants.get(lot_id, frozenset())

        if not lot_entrants:
            n_unserviced += 1
            continue

        bids[lot_id] = lot_bids
        entrants[lot_id] = lot_entrants
        item_idxs = partition.lot_items(lot_id)
        volumes[lot_id] = sum(instance.items[i].volume for i in item_idxs)

    # Value of serviced items only
    serviced_item_idxs: set[int] = set()
    for lot_id in bids:
        serviced_item_idxs.update(partition.lot_items(lot_id))
    serviced_value = sum(instance.items[i].base_value for i in serviced_item_idxs)
    mean_item_value = sum(item.base_value for item in instance.items) / len(instance.items)

    if not bids:
        penalty = -10.0 * mean_item_value * n_unserviced
        return EvalResult(
            reward=float(penalty),
            info={"status": "ALL_UNSERVICED", "n_unserviced": n_unserviced},
        )

    # Volume tiers from config
    vol_tiers: tuple[tuple[float, float], ...] = ()
    if config.coupling_strength > 0:
        vol_tiers = tuple(
            zip(config.volume_tier_thresholds, config.volume_tier_discounts, strict=False)
        )

    solver_result = solver.solve(
        bids=bids,
        entrants=entrants,
        volumes=volumes,
        capacities=capacities,
        package_bids=sim_result.package_bids,
        volume_tiers=vol_tiers,
    )

    if solver_result.status == "INFEASIBLE":
        penalty = -10.0 * mean_item_value * (len(bids) + n_unserviced)
        return EvalResult(
            reward=float(penalty),
            info={"status": "INFEASIBLE", "n_unserviced": n_unserviced},
        )

    reward = serviced_value - solver_result.total_cost
    if n_unserviced > 0:
        reward -= 10.0 * mean_item_value * n_unserviced

    # Lot overhead: penalizes lot proliferation (gated by coupling)
    lot_overhead = config.lot_overhead_base * config.coupling_strength
    lot_overhead_total = 0.0
    if lot_overhead > 0:
        lot_overhead_total = lot_overhead * partition.n_lots
        reward -= lot_overhead_total

    info = {
        "status": solver_result.status,
        "total_cost": solver_result.total_cost,
        "n_lots": partition.n_lots,
        "n_unserviced": n_unserviced,
        "lot_overhead_total": lot_overhead_total,
        "optimality_gap": solver_result.optimality_gap,
        "solve_time_s": solver_result.solve_time_s,
        "n_package_bids": len(sim_result.package_bids),
        "n_package_awards": len(solver_result.package_awards),
    }
    return EvalResult(reward=float(reward), info=info)


def evaluate_multi_seed(
    partition: LotPartition,
    instance: ProcurementInstance,
    config: SupplierModelConfig,
    n_seeds: int,
    base_seed: int = 0,
    area_size: float = 100.0,
    solver_timeout_s: float = 30.0,
) -> dict[str, Any]:
    """Evaluate a partition across multiple seeds and return statistics."""
    rewards: list[float] = []
    for i in range(n_seeds):
        result = evaluate_partition(
            partition,
            instance,
            config,
            seed=base_seed + i,
            area_size=area_size,
            solver_timeout_s=solver_timeout_s,
        )
        rewards.append(result.reward)

    arr = np.array(rewards)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "rewards": rewards,
    }
