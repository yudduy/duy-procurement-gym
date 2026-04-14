# PLAN: Generates transport procurement instances with geographic coordinates,
#        distance-based costs, and carrier capability sets. REQ-4.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 456-503
# ADAPTED FROM: kevinlb1/CATS regions distribution (grid-based spatial model)
"""Transport procurement instance generator."""

from __future__ import annotations

import numpy as np

from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.core.types import Item, ProcurementInstance, Supplier


class TransportInstanceGenerator:
    """Generates synthetic transport procurement instances."""

    def __init__(
        self,
        instance_config: InstanceConfig,
        supplier_config: SupplierModelConfig,
    ) -> None:
        self._ic = instance_config
        self._sc = supplier_config

    def generate(self, seed: int) -> ProcurementInstance:
        """Generate a procurement instance with the given seed."""
        rng = np.random.default_rng(seed)

        categories = [f"cat_{i}" for i in range(self._ic.n_categories)]

        # Generate suppliers first so we can compute qualified sets
        suppliers = self._generate_suppliers(rng, categories)

        # Generate items with qualified supplier references
        items = self._generate_items(rng, categories, suppliers)

        return ProcurementInstance(items=tuple(items), suppliers=tuple(suppliers), seed=seed)

    def _generate_suppliers(
        self, rng: np.random.Generator, categories: list[str]
    ) -> list[Supplier]:
        suppliers: list[Supplier] = []
        # At high coupling, broaden supplier category coverage
        extra_cats = max(
            0,
            int(
                round(
                    self._sc.coupling_strength
                    * (len(categories) - self._ic.max_supplier_categories)
                )
            ),
        )
        effective_max = min(len(categories), self._ic.max_supplier_categories + extra_cats)
        effective_min = min(self._ic.min_supplier_categories, effective_max)

        # Capacity scarcity: tighten at high coupling
        scarcity_factor = 1.0 / (1.0 + self._ic.capacity_scarcity * self._sc.coupling_strength)

        for sid in range(self._ic.n_suppliers):
            n_cats = rng.integers(effective_min, effective_max + 1)
            chosen_cats = frozenset(
                rng.choice(categories, size=int(n_cats), replace=False).tolist()
            )
            suppliers.append(
                Supplier(
                    id=sid,
                    categories=chosen_cats,
                    capacity=float(rng.uniform(50.0, 200.0)) * scarcity_factor,
                    base_x=float(rng.uniform(0.0, self._ic.area_size)),
                    base_y=float(rng.uniform(0.0, self._ic.area_size)),
                    cost_per_unit_distance=float(rng.uniform(0.5, 2.0)),
                    fixed_cost=float(rng.uniform(50.0, 300.0)),
                    markup_alpha=max(
                        0.01,
                        float(
                            rng.normal(
                                self._sc.markup_alpha_mean,
                                self._sc.markup_alpha_std
                                * (1.0 - self._sc.coupling_strength * self._sc.noise_reduction),
                            )
                        ),
                    ),
                    markup_beta=max(
                        0.01,
                        float(
                            rng.normal(
                                self._sc.markup_beta_mean,
                                self._sc.markup_beta_std
                                * (1.0 - self._sc.coupling_strength * self._sc.noise_reduction),
                            )
                        ),
                    ),
                )
            )

        # Ensure every category is covered by at least one supplier
        covered_cats: set[str] = set()
        for s in suppliers:
            covered_cats.update(s.categories)
        missing = set(categories) - covered_cats
        for cat in missing:
            # Add missing category to a random supplier
            idx = int(rng.integers(0, len(suppliers)))
            old = suppliers[idx]
            suppliers[idx] = Supplier(
                id=old.id,
                categories=old.categories | frozenset({cat}),
                capacity=old.capacity,
                base_x=old.base_x,
                base_y=old.base_y,
                cost_per_unit_distance=old.cost_per_unit_distance,
                fixed_cost=old.fixed_cost,
                markup_alpha=old.markup_alpha,
                markup_beta=old.markup_beta,
            )

        return suppliers

    def _generate_items(
        self,
        rng: np.random.Generator,
        categories: list[str],
        suppliers: list[Supplier],
    ) -> list[Item]:
        items: list[Item] = []
        for iid in range(self._ic.n_items):
            cat = str(rng.choice(categories))
            x = float(rng.uniform(0.0, self._ic.area_size))
            y = float(rng.uniform(0.0, self._ic.area_size))
            base_value = float(
                max(10.0, rng.normal(self._ic.base_value_mean, self._ic.base_value_std))
            )
            volume = float(max(0.1, rng.normal(5.0, 2.0)))

            # Qualified suppliers = those whose categories include this item's category
            qualified = frozenset(s.id for s in suppliers if cat in s.categories)

            items.append(
                Item(
                    id=iid,
                    category=cat,
                    base_value=base_value,
                    x=x,
                    y=y,
                    volume=volume,
                    qualified_supplier_ids=qualified,
                )
            )

        return items
