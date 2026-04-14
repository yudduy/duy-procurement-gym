# PLAN: Package exports for 5 baseline lot partition strategies needed for
#        reward signal validation. Each strategy: ProcurementInstance → LotPartition.
# SOURCE: Standard algorithmic primitives (random assignment, k-means, category grouping).
# ADAPTED FROM: sktime/sktime random_partition pattern; scipy.cluster.vq.kmeans2 for geographic.
"""Baseline lot partition strategies."""

from procurement_gym.baselines.strategies import (
    category_partition,
    equal_size_partition,
    geographic_partition,
    random_partition,
    single_item_partition,
)

__all__ = [
    "category_partition",
    "equal_size_partition",
    "geographic_partition",
    "random_partition",
    "single_item_partition",
]
