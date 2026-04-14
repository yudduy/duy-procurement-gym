# PLAN: Defines frozen dataclasses (Item, Supplier, ProcurementInstance) for the
#        procurement domain. These are the foundational data objects used by all other
#        modules. Immutability is enforced via frozen=True per REQ-3.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 476-503
"""Frozen dataclasses for procurement domain objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    """A procurement item (e.g., a shipping lane)."""

    id: int
    category: str
    base_value: float
    x: float
    y: float
    volume: float
    qualified_supplier_ids: frozenset[int]


@dataclass(frozen=True)
class Supplier:
    """A supplier (e.g., a freight carrier)."""

    id: int
    categories: frozenset[str]
    capacity: float
    base_x: float
    base_y: float
    cost_per_unit_distance: float
    fixed_cost: float
    markup_alpha: float
    markup_beta: float


@dataclass(frozen=True)
class PackageBid:
    """A discounted bid for a bundle of lots."""

    supplier_id: int
    lot_ids: frozenset[int]
    price: float  # total discounted price for the bundle


@dataclass(frozen=True)
class ProcurementInstance:
    """A complete procurement problem instance."""

    items: tuple[Item, ...]
    suppliers: tuple[Supplier, ...]
    seed: int
