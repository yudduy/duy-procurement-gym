# PLAN: Competition-compressed markup model. Layer 3 of three-layer supplier model.
#        markup = alpha * (1/N_competitors)^beta. GPV-calibratable.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 89-93
"""Markup model: competition compresses supplier markups."""

from __future__ import annotations


def compute_markup(
    n_competitors: int,
    markup_alpha: float,
    markup_beta: float,
) -> float:
    """Compute markup multiplier for a supplier.

    markup = alpha * (1 / max(1, n_competitors))^beta

    More competitors → lower markup. With 1 competitor, markup = alpha.
    With 0 competitors, caps at alpha (monopolist markup).
    """
    effective_n = max(1, n_competitors)
    return float(markup_alpha * (1.0 / effective_n) ** markup_beta)
