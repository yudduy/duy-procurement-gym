# PLAN: Generate cross-lot package bids. Suppliers offer discounts for winning
#        bundles of lots, creating the coupling that makes lot design NP-hard.
# SOURCE: Planner agent — enriched procurement RLVR gym spec
"""Package bid generation for cross-lot coupling."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np

from procurement_gym.config import SupplierModelConfig
from procurement_gym.core.types import Item, PackageBid, Supplier


def generate_package_bids(
    per_lot_bids: dict[int, dict[int, float]],
    items_by_lot: dict[int, tuple[int, ...]],
    items: Sequence[Item],
    suppliers: Sequence[Supplier],
    config: SupplierModelConfig,
    area_size: float,
    rng: np.random.Generator,
) -> tuple[PackageBid, ...]:
    """Generate package bids from per-lot bids with synergy-based filtering.

    Returns empty tuple when coupling_strength == 0.
    """
    if config.coupling_strength <= 0.0:
        return ()

    lot_ids = sorted(per_lot_bids.keys())
    if len(lot_ids) < 2:
        return ()

    # Precompute lot centroids for synergy scoring
    lot_centroids: dict[int, tuple[float, float]] = {}
    for lid in lot_ids:
        idxs = items_by_lot[lid]
        if not idxs:
            continue
        xs = [items[i].x for i in idxs]
        ys = [items[i].y for i in idxs]
        lot_centroids[lid] = (sum(xs) / len(xs), sum(ys) / len(ys))

    # Build supplier -> lots they bid on
    supplier_lots: dict[int, list[int]] = {}
    for lid in lot_ids:
        for sid in per_lot_bids[lid]:
            supplier_lots.setdefault(sid, []).append(lid)

    packages: list[PackageBid] = []

    for sid, s_lots in supplier_lots.items():
        if len(s_lots) < 2:
            continue
        supplier_candidates: list[tuple[float, float, PackageBid]] = []

        # Enumerate subsets of size 2..max_package_size
        for size in range(2, min(config.max_package_size, len(s_lots)) + 1):
            for combo in itertools.combinations(s_lots, size):
                # Synergy score: geographic proximity of lot centroids
                synergy = _lot_synergy(combo, lot_centroids, area_size)
                # Soft threshold: at coupling=0 uses base threshold, at coupling=1 threshold=0
                effective_threshold = config.package_synergy_threshold * (
                    1.0 - config.coupling_strength
                )
                if synergy < effective_threshold:
                    continue

                # Deterministic synergy-linked discount (higher synergy → bigger discount)
                discount = config.coupling_strength * (
                    config.package_discount_min
                    + (config.package_discount_max - config.package_discount_min) * synergy
                )

                # Package price = sum of individual bids * (1 - discount)
                individual_sum = sum(per_lot_bids[lid][sid] for lid in combo)
                pkg_price = individual_sum * (1.0 - discount)
                package = PackageBid(
                    supplier_id=sid,
                    lot_ids=frozenset(combo),
                    price=pkg_price,
                )
                savings = individual_sum - pkg_price
                supplier_candidates.append((savings, synergy, package))

        # Keep the most economically meaningful packages per supplier.
        supplier_candidates.sort(key=lambda x: (-x[0], -x[1], len(x[2].lot_ids)))
        packages.extend(
            package
            for _, _, package in supplier_candidates[: config.max_packages_per_supplier]
        )

    return tuple(packages)


def _lot_synergy(
    lot_ids: tuple[int, ...],
    lot_centroids: dict[int, tuple[float, float]],
    area_size: float,
) -> float:
    """Compute geographic synergy for a set of lots (0 to 1, higher = closer)."""
    centroids = [lot_centroids[lid] for lid in lot_ids if lid in lot_centroids]
    if len(centroids) < 2:
        return 0.0

    # Mean pairwise distance between centroids, normalized by area
    total_dist = 0.0
    n_pairs = 0
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            total_dist += math.sqrt(dx * dx + dy * dy)
            n_pairs += 1

    mean_dist = total_dist / n_pairs
    # Proximity: 1 when lots are co-located, 0 when maximally spread
    return max(0.0, 1.0 - mean_dist / area_size)
