# PLAN: Additive cost model with geographic synergy. Layer 1 of the three-layer
#        supplier model. Synergy is proportional to lot base cost and geographic proximity.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 584-599 (synergy formula)
#         Plan synergy fix: normalize by area_size, scale by total_base_cost.
"""Cost model: additive base cost + geographic synergy + noise."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.spatial.distance import pdist

from procurement_gym.config import SupplierModelConfig
from procurement_gym.core.types import Item, Supplier


def compute_lot_cost(
    items: Sequence[Item],
    supplier: Supplier,
    config: SupplierModelConfig,
    area_size: float,
    rng: np.random.Generator,
) -> float:
    """Compute supplier's cost for serving a lot of items.

    Cost = sum(base_costs) + synergy + noise.
    Synergy is negative (savings) when items are geographically close.
    """
    if not items:
        return 0.0

    # Base cost: fixed cost ONCE per lot + distance-based per item
    distance_costs: list[float] = []
    for item in items:
        dist = math.sqrt((item.x - supplier.base_x) ** 2 + (item.y - supplier.base_y) ** 2)
        distance_costs.append(supplier.cost_per_unit_distance * dist)

    total_base = supplier.fixed_cost + sum(distance_costs)

    # Economy of scale: larger lots get bulk discount (gated by coupling)
    if len(items) >= 2 and config.coupling_strength > 0 and config.economy_of_scale > 0:
        log_ratio = math.log(len(items)) / math.log(20)
        scale_factor = 1.0 - config.coupling_strength * config.economy_of_scale * min(
            1.0, log_ratio
        )
        total_base *= scale_factor

    # Synergy: proportional to lot base cost and geographic proximity
    syn = _synergy(items, supplier, config, area_size, total_base)

    # Noise: per-lot idiosyncratic cost (eliminated at high coupling for deterministic verification)
    determinism = config.coupling_strength * config.noise_reduction
    if determinism >= 0.99:
        noise = 0.0
    else:
        effective_noise_std = config.cost_noise_std * (1.0 - determinism)
        noise = float(rng.normal(0.0, effective_noise_std * total_base))

    return max(0.01, total_base + syn + noise)


def _synergy(
    items: Sequence[Item],
    supplier: Supplier,
    config: SupplierModelConfig,
    area_size: float,
    total_base_cost: float,
) -> float:
    """Geographic proximity synergy — closer items = cost savings."""
    if len(items) <= 1:
        return 0.0

    coords = np.array([(item.x, item.y) for item in items])
    norm_pairwise = float(np.mean(pdist(coords))) / area_size

    centroid = coords.mean(axis=0)
    hub_dist = float(
        math.sqrt((centroid[0] - supplier.base_x) ** 2 + (centroid[1] - supplier.base_y) ** 2)
    )
    norm_hub = hub_dist / area_size

    proximity = max(0.0, 1.0 - norm_pairwise) * max(0.0, 1.0 - norm_hub)
    return -config.synergy_weight * total_base_cost * proximity
