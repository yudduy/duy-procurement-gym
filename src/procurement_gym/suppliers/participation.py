# PLAN: Logistic participation model. Layer 2 of three-layer supplier model.
#        P(enter) = sigmoid(w_cap*match + w_size*log|lot| + w_comp*|eligible| + bias).
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 120-121
"""Participation model: logistic probability of supplier entering a lot."""

from __future__ import annotations

import math

from procurement_gym.config import SupplierModelConfig


def compute_participation_prob(
    capability_match: float,
    lot_size: int,
    n_eligible: int,
    supplier_bias: float,
    config: SupplierModelConfig,
    geographic_proximity: float = 0.0,
) -> float:
    """Compute probability a supplier participates in a lot.

    Args:
        capability_match: fraction of lot categories the supplier covers [0, 1]
        lot_size: number of items in the lot
        n_eligible: number of eligible suppliers for this lot
        supplier_bias: per-supplier bias term (drawn from config distribution)
        config: supplier model configuration
        geographic_proximity: 1 - (distance to lot centroid / area_size), in [0, 1]
    """
    z = (
        config.participation_w_cap * capability_match
        + config.participation_w_size * math.log(max(1, lot_size))
        + config.participation_w_comp * n_eligible
        + config.participation_w_geo * config.coupling_strength * geographic_proximity
        + supplier_bias
    )
    return _sigmoid(z)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)
