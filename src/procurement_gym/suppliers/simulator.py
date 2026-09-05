# PLAN: Composes cost, participation, and markup into a complete supplier simulator.
#        Sequential pipeline: eligible → participate → cost → markup → bid. REQ-5.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 116-133
"""Three-layer supplier simulator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from procurement_gym.config import SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import Item, PackageBid, ProcurementInstance
from procurement_gym.suppliers.cost import compute_lot_cost
from procurement_gym.suppliers.markup import compute_markup
from procurement_gym.suppliers.packages import generate_package_bids
from procurement_gym.suppliers.participation import compute_participation_prob


@dataclass(frozen=True)
class SimulationResult:
    """Result of simulating supplier behavior for a partition."""

    bids: dict[int, dict[int, float]] = field(default_factory=dict)
    entrants: dict[int, frozenset[int]] = field(default_factory=dict)
    costs: dict[int, dict[int, float]] = field(default_factory=dict)
    package_bids: tuple[PackageBid, ...] = ()


class ThreeLayerSupplierSimulator:
    """Simulates supplier behavior using cost + participation + markup layers."""

    def __init__(self, config: SupplierModelConfig, area_size: float) -> None:
        self._config = config
        self._area_size = area_size

    def simulate(
        self,
        partition: LotPartition,
        instance: ProcurementInstance,
        rng: np.random.Generator,
    ) -> SimulationResult:
        """Simulate supplier bids for a complete partition."""
        all_bids: dict[int, dict[int, float]] = {}
        all_entrants: dict[int, frozenset[int]] = {}
        all_costs: dict[int, dict[int, float]] = {}

        # Pre-compute per-supplier participation biases
        # At high coupling, use deterministic bias (no random draw) for reproducible verification
        determinism = self._config.coupling_strength * self._config.noise_reduction
        effective_bias_mean = (
            self._config.participation_bias_mean
            + self._config.coupling_strength * self._config.participation_coupling_boost
        )
        effective_bias_std = self._config.participation_bias_std * (1.0 - determinism)
        effective_bias_std = max(0.0, effective_bias_std)

        supplier_biases: dict[int, float] = {}
        for s in instance.suppliers:
            if effective_bias_std < 0.01:
                # Fully deterministic: use mean bias (no random draw)
                supplier_biases[s.id] = effective_bias_mean
            else:
                supplier_biases[s.id] = float(rng.normal(effective_bias_mean, effective_bias_std))

        for lot_id in sorted(partition.lot_ids):
            item_idxs = partition.lot_items(lot_id)
            lot_items: list[Item] = [instance.items[i] for i in item_idxs]
            lot_categories = {item.category for item in lot_items}
            lot_volume = sum(item.volume for item in lot_items)
            qualified_for_all_items = frozenset.intersection(
                *(item.qualified_supplier_ids for item in lot_items)
            )

            # Step 1: Determine eligible suppliers.
            # A supplier must be feasible for the whole lot, not just overlap one category.
            eligible: list[int] = []
            for s in instance.suppliers:
                covers_lot = lot_categories <= s.categories
                qualified_for_lot = s.id in qualified_for_all_items
                can_serve_volume = lot_volume <= s.capacity
                if covers_lot and qualified_for_lot and can_serve_volume:
                    eligible.append(s.id)

            n_eligible = len(eligible)

            # Lot centroid for geographic proximity
            lot_cx = sum(item.x for item in lot_items) / len(lot_items)
            lot_cy = sum(item.y for item in lot_items) / len(lot_items)

            # Step 2: Participation sampling (uses eligible count, not entrant count)
            entrant_ids: list[int] = []
            for sid in eligible:
                supplier = instance.suppliers[sid]
                # Capability match = fraction of lot categories covered
                match = len(supplier.categories & lot_categories) / len(lot_categories)
                # Geographic proximity = 1 - normalized distance to lot centroid
                dist = math.sqrt((supplier.base_x - lot_cx) ** 2 + (supplier.base_y - lot_cy) ** 2)
                geo_prox = max(0.0, 1.0 - dist / self._area_size)
                # Dampen geo proximity for small lots (single-item centroid is degenerate)
                lot_size_factor = min(1.0, math.log(max(2, len(lot_items))) / math.log(5))
                geo_prox *= lot_size_factor
                prob = compute_participation_prob(
                    capability_match=match,
                    lot_size=len(lot_items),
                    n_eligible=n_eligible,
                    supplier_bias=supplier_biases[sid],
                    config=self._config,
                    geographic_proximity=geo_prox,
                )
                if determinism >= 0.99:
                    # Deterministic: threshold-based participation (no randomness)
                    if prob >= 0.5:
                        entrant_ids.append(sid)
                else:
                    if float(rng.random()) < prob:
                        entrant_ids.append(sid)

            all_entrants[lot_id] = frozenset(entrant_ids)
            n_entrants = len(entrant_ids)

            # Steps 3-4: Cost and markup for each entrant
            lot_bids: dict[int, float] = {}
            lot_costs: dict[int, float] = {}
            for sid in entrant_ids:
                supplier = instance.suppliers[sid]

                # Layer 1: Cost
                cost = compute_lot_cost(
                    items=lot_items,
                    supplier=supplier,
                    config=self._config,
                    area_size=self._area_size,
                    rng=rng,
                )
                lot_costs[sid] = cost

                # Layer 3: Markup. Use total bidders in the lot market so duopoly
                # is more competitive than monopoly.
                markup = compute_markup(
                    n_competitors=n_entrants,
                    markup_alpha=supplier.markup_alpha,
                    markup_beta=supplier.markup_beta,
                )

                # Bid = cost * (1 + markup)
                lot_bids[sid] = cost * (1.0 + markup)

            all_bids[lot_id] = lot_bids
            all_costs[lot_id] = lot_costs

        # Generate cross-lot package bids if coupling is enabled
        items_by_lot: dict[int, tuple[int, ...]] = {}
        for lot_id in sorted(partition.lot_ids):
            items_by_lot[lot_id] = partition.lot_items(lot_id)

        pkg_bids = generate_package_bids(
            per_lot_bids=all_bids,
            items_by_lot=items_by_lot,
            items=instance.items,
            suppliers=instance.suppliers,
            config=self._config,
            area_size=self._area_size,
            rng=rng,
        )

        return SimulationResult(
            bids=all_bids,
            entrants=all_entrants,
            costs=all_costs,
            package_bids=pkg_bids,
        )
