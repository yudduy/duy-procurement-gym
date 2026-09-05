"""Tests for LotPartition — immutable partition data structure."""

import pytest

from procurement_gym.core.partition import LotPartition


class TestLotPartitionEmpty:
    def test_empty_creates_unassigned(self) -> None:
        p = LotPartition.empty(5)
        assert p.n_items == 5
        assert p.assignments == (-1, -1, -1, -1, -1)
        assert p.n_lots == 0
        assert not p.is_complete()

    def test_empty_zero_items(self) -> None:
        p = LotPartition.empty(0)
        assert p.n_items == 0
        assert p.is_complete()


class TestLotPartitionAssign:
    def test_assign_returns_new_partition(self) -> None:
        p1 = LotPartition.empty(3)
        p2 = p1.assign(0, 0)
        assert p1.assignments == (-1, -1, -1)  # original unchanged
        assert p2.assignments == (0, -1, -1)

    def test_assign_multiple_items(self) -> None:
        p = LotPartition.empty(4)
        p = p.assign(0, 0)
        p = p.assign(1, 0)
        p = p.assign(2, 1)
        p = p.assign(3, 1)
        assert p.assignments == (0, 0, 1, 1)
        assert p.n_lots == 2
        assert p.is_complete()

    def test_assign_invalid_item_raises(self) -> None:
        p = LotPartition.empty(3)
        with pytest.raises((IndexError, ValueError)):
            p.assign(-1, 0)
        with pytest.raises((IndexError, ValueError)):
            p.assign(3, 0)

    def test_assign_negative_lot_raises(self) -> None:
        p = LotPartition.empty(3)
        with pytest.raises(ValueError):
            p.assign(0, -1)

    def test_assign_already_assigned_raises(self) -> None:
        p = LotPartition.empty(3).assign(0, 0)
        with pytest.raises(ValueError):
            p.assign(0, 1)


class TestLotPartitionQueries:
    def test_next_lot_id_empty(self) -> None:
        p = LotPartition.empty(3)
        assert p.next_lot_id() == 0

    def test_next_lot_id_after_assign(self) -> None:
        p = LotPartition.empty(3).assign(0, 0)
        assert p.next_lot_id() == 1

    def test_next_lot_id_with_gap(self) -> None:
        """If lots 0 and 2 exist (gap at 1), next_lot_id = 3."""
        p = LotPartition.empty(3).assign(0, 0).assign(1, 2)
        assert p.next_lot_id() == 3

    def test_lot_items(self) -> None:
        p = LotPartition.empty(4).assign(0, 0).assign(1, 1).assign(2, 0).assign(3, 1)
        assert p.lot_items(0) == (0, 2)
        assert p.lot_items(1) == (1, 3)

    def test_lot_items_nonexistent(self) -> None:
        p = LotPartition.empty(3).assign(0, 0)
        assert p.lot_items(99) == ()

    def test_lot_ids(self) -> None:
        p = LotPartition.empty(3).assign(0, 0).assign(1, 2).assign(2, 0)
        assert p.lot_ids == frozenset({0, 2})

    def test_n_lots(self) -> None:
        p = LotPartition.empty(3).assign(0, 0).assign(1, 0).assign(2, 1)
        assert p.n_lots == 2


class TestLotPartitionFrozen:
    def test_frozen(self) -> None:
        p = LotPartition.empty(3)
        with pytest.raises(AttributeError):
            p.n_items = 5  # type: ignore[misc]
        with pytest.raises(AttributeError):
            p.assignments = (0, 0, 0)  # type: ignore[misc]
