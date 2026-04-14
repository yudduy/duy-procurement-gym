# PLAN: Immutable LotPartition — assigns items to lots via functional updates.
#        assign() returns a NEW partition. Core data structure for the RL env.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 506-536
"""Immutable lot partition data structure."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LotPartition:
    """An immutable partition of items into lots.

    assignments[i] = lot_id for item i, or -1 if unassigned.
    """

    assignments: tuple[int, ...]
    n_items: int

    @staticmethod
    def empty(n_items: int) -> LotPartition:
        """Create a partition with all items unassigned."""
        return LotPartition(assignments=tuple(-1 for _ in range(n_items)), n_items=n_items)

    def assign(self, item_idx: int, lot_id: int) -> LotPartition:
        """Return a NEW partition with item_idx assigned to lot_id."""
        if item_idx < 0 or item_idx >= self.n_items:
            raise ValueError(f"Item index {item_idx} out of range [0, {self.n_items})")
        if lot_id < 0:
            raise ValueError(f"Lot ID must be non-negative, got {lot_id}")
        if self.assignments[item_idx] != -1:
            raise ValueError(
                f"Item {item_idx} already assigned to lot {self.assignments[item_idx]}"
            )
        new_assignments = list(self.assignments)
        new_assignments[item_idx] = lot_id
        return LotPartition(assignments=tuple(new_assignments), n_items=self.n_items)

    def next_lot_id(self) -> int:
        """Return the next available lot ID (max existing + 1, or 0 if empty)."""
        assigned = [a for a in self.assignments if a >= 0]
        if not assigned:
            return 0
        return max(assigned) + 1

    def lot_items(self, lot_id: int) -> tuple[int, ...]:
        """Return item indices assigned to this lot."""
        return tuple(i for i, a in enumerate(self.assignments) if a == lot_id)

    def is_complete(self) -> bool:
        """True if all items are assigned."""
        return all(a >= 0 for a in self.assignments)

    @property
    def n_lots(self) -> int:
        """Number of distinct lots."""
        return len(self.lot_ids)

    @property
    def lot_ids(self) -> frozenset[int]:
        """Set of lot IDs in use."""
        return frozenset(a for a in self.assignments if a >= 0)
