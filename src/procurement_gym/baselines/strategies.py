# PLAN: Five baseline lot partition strategies for reward signal validation.
#        Each takes a ProcurementInstance and returns a complete LotPartition.
#        Used to generate "good", "mediocre", and "bad" partitions for SNR analysis.
# SOURCE: Standard algorithmic primitives.
# ADAPTED FROM: sktime random_partition pattern; scipy.cluster.vq.kmeans2 for geographic.
"""Baseline lot partition strategies."""

from __future__ import annotations

import numpy as np
from scipy.cluster.vq import kmeans2

from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import ProcurementInstance


def random_partition(
    instance: ProcurementInstance,
    n_lots: int,
    rng: np.random.Generator,
) -> LotPartition:
    """Assign each item to a random lot uniformly."""
    n = len(instance.items)
    assignments = rng.integers(0, n_lots, size=n)
    # Ensure all lots are used — reassign one item per missing lot
    used = set(assignments.tolist())
    missing = [k for k in range(n_lots) if k not in used]
    for i, lot_id in enumerate(missing):
        assignments[i] = lot_id
    return LotPartition(assignments=tuple(int(a) for a in assignments), n_items=n)


def single_item_partition(instance: ProcurementInstance) -> LotPartition:
    """Each item gets its own lot."""
    n = len(instance.items)
    return LotPartition(assignments=tuple(range(n)), n_items=n)


def category_partition(instance: ProcurementInstance) -> LotPartition:
    """Group items by category."""
    cats = sorted({item.category for item in instance.items})
    cat_to_lot = {c: i for i, c in enumerate(cats)}
    assignments = tuple(cat_to_lot[item.category] for item in instance.items)
    return LotPartition(assignments=assignments, n_items=len(instance.items))


def geographic_partition(
    instance: ProcurementInstance,
    n_lots: int,
    rng: np.random.Generator,
) -> LotPartition:
    """K-means clustering on item (x, y) coordinates."""
    n = len(instance.items)
    coords = np.array([[item.x, item.y] for item in instance.items], dtype=np.float64)
    seed = int(rng.integers(0, 2**31))
    _, labels = kmeans2(coords, n_lots, minit="points", seed=seed)
    # Relabel to contiguous 0..k-1
    unique = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(unique)}
    assignments = tuple(remap[int(l)] for l in labels)
    return LotPartition(assignments=assignments, n_items=n)


def equal_size_partition(
    instance: ProcurementInstance,
    n_lots: int,
) -> LotPartition:
    """Distribute items as evenly as possible across lots (round-robin)."""
    n = len(instance.items)
    assignments = tuple(i % n_lots for i in range(n))
    return LotPartition(assignments=assignments, n_items=n)
