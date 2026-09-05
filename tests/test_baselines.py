"""Tests for baseline lot partition strategies."""

from __future__ import annotations

import numpy as np
import pytest

from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import ProcurementInstance
from procurement_gym.instances.transport import TransportInstanceGenerator


def _make_instance(
    n_items: int = 20,
    n_suppliers: int = 10,
    n_categories: int = 3,
    seed: int = 42,
) -> ProcurementInstance:
    gen = TransportInstanceGenerator(
        instance_config=InstanceConfig(
            n_items=n_items, n_suppliers=n_suppliers, n_categories=n_categories
        ),
        supplier_config=SupplierModelConfig(),
    )
    return gen.generate(seed=seed)


def _assert_valid_partition(partition: LotPartition, n_items: int) -> None:
    """Assert a partition is complete and well-formed."""
    assert partition.n_items == n_items
    assert partition.is_complete(), "All items must be assigned"
    assert partition.n_lots >= 1, "Must have at least one lot"
    # Every item assigned to a non-negative lot
    for a in partition.assignments:
        assert a >= 0


class TestRandomBaseline:
    def test_returns_complete_partition(self) -> None:
        from procurement_gym.baselines import random_partition

        instance = _make_instance()
        partition = random_partition(instance, n_lots=5, rng=np.random.default_rng(0))
        _assert_valid_partition(partition, len(instance.items))

    def test_respects_n_lots(self) -> None:
        from procurement_gym.baselines import random_partition

        instance = _make_instance()
        partition = random_partition(instance, n_lots=4, rng=np.random.default_rng(0))
        assert partition.n_lots == 4

    def test_deterministic_with_same_seed(self) -> None:
        from procurement_gym.baselines import random_partition

        instance = _make_instance()
        p1 = random_partition(instance, n_lots=5, rng=np.random.default_rng(99))
        p2 = random_partition(instance, n_lots=5, rng=np.random.default_rng(99))
        assert p1 == p2

    def test_different_seeds_differ(self) -> None:
        from procurement_gym.baselines import random_partition

        instance = _make_instance()
        p1 = random_partition(instance, n_lots=5, rng=np.random.default_rng(0))
        p2 = random_partition(instance, n_lots=5, rng=np.random.default_rng(1))
        assert p1 != p2


class TestSingleItemBaseline:
    def test_each_item_own_lot(self) -> None:
        from procurement_gym.baselines import single_item_partition

        instance = _make_instance(n_items=10)
        partition = single_item_partition(instance)
        _assert_valid_partition(partition, 10)
        assert partition.n_lots == 10

    def test_lot_sizes_all_one(self) -> None:
        from procurement_gym.baselines import single_item_partition

        instance = _make_instance(n_items=8)
        partition = single_item_partition(instance)
        for lot_id in partition.lot_ids:
            assert len(partition.lot_items(lot_id)) == 1


class TestCategoryBaseline:
    def test_groups_by_category(self) -> None:
        from procurement_gym.baselines import category_partition

        instance = _make_instance(n_items=20, n_categories=3)
        partition = category_partition(instance)
        _assert_valid_partition(partition, 20)
        # Number of lots should equal number of distinct categories present
        cats_in_items = {item.category for item in instance.items}
        assert partition.n_lots == len(cats_in_items)

    def test_items_in_same_lot_share_category(self) -> None:
        from procurement_gym.baselines import category_partition

        instance = _make_instance(n_items=20, n_categories=3)
        partition = category_partition(instance)
        for lot_id in partition.lot_ids:
            item_idxs = partition.lot_items(lot_id)
            cats = {instance.items[i].category for i in item_idxs}
            assert len(cats) == 1, f"Lot {lot_id} has mixed categories: {cats}"


class TestGeographicBaseline:
    def test_returns_complete_partition(self) -> None:
        from procurement_gym.baselines import geographic_partition

        instance = _make_instance()
        partition = geographic_partition(instance, n_lots=4, rng=np.random.default_rng(0))
        _assert_valid_partition(partition, len(instance.items))

    def test_respects_n_lots(self) -> None:
        from procurement_gym.baselines import geographic_partition

        instance = _make_instance(n_items=20)
        partition = geographic_partition(instance, n_lots=3, rng=np.random.default_rng(0))
        assert partition.n_lots == 3

    def test_nearby_items_same_lot(self) -> None:
        """Items close geographically should tend to be in the same lot."""
        from procurement_gym.baselines import geographic_partition

        instance = _make_instance(n_items=20, n_categories=1)
        partition = geographic_partition(instance, n_lots=2, rng=np.random.default_rng(42))
        # Compute mean x per lot — lots should have distinct centroids
        centroids = []
        for lot_id in sorted(partition.lot_ids):
            items = [instance.items[i] for i in partition.lot_items(lot_id)]
            cx = sum(it.x for it in items) / len(items)
            cy = sum(it.y for it in items) / len(items)
            centroids.append((cx, cy))
        # With 2 lots, centroids should be separated (not identical)
        if len(centroids) == 2:
            dist = (
                (centroids[0][0] - centroids[1][0]) ** 2 + (centroids[0][1] - centroids[1][1]) ** 2
            ) ** 0.5
            assert dist > 1.0, "Geographic clustering should separate spatially"


class TestEqualSizeBaseline:
    def test_returns_complete_partition(self) -> None:
        from procurement_gym.baselines import equal_size_partition

        instance = _make_instance(n_items=20)
        partition = equal_size_partition(instance, n_lots=4)
        _assert_valid_partition(partition, 20)

    def test_lot_sizes_balanced(self) -> None:
        from procurement_gym.baselines import equal_size_partition

        instance = _make_instance(n_items=20)
        partition = equal_size_partition(instance, n_lots=4)
        sizes = [len(partition.lot_items(lid)) for lid in partition.lot_ids]
        # All lots should be within 1 of each other
        assert max(sizes) - min(sizes) <= 1

    def test_exact_division(self) -> None:
        from procurement_gym.baselines import equal_size_partition

        instance = _make_instance(n_items=12)
        partition = equal_size_partition(instance, n_lots=3)
        sizes = [len(partition.lot_items(lid)) for lid in partition.lot_ids]
        assert all(s == 4 for s in sizes)

    def test_uneven_division(self) -> None:
        from procurement_gym.baselines import equal_size_partition

        instance = _make_instance(n_items=10)
        partition = equal_size_partition(instance, n_lots=3)
        sizes = sorted(len(partition.lot_items(lid)) for lid in partition.lot_ids)
        # 10 / 3 = 3 remainder 1 → two lots of 3, one of 4
        assert sizes == [3, 3, 4]


class TestBaselineInterface:
    """All baselines should work with various instance sizes."""

    @pytest.mark.parametrize("n_items", [5, 10, 50])
    def test_all_baselines_run(self, n_items: int) -> None:
        from procurement_gym.baselines import (
            category_partition,
            equal_size_partition,
            geographic_partition,
            random_partition,
            single_item_partition,
        )

        instance = _make_instance(n_items=n_items, n_categories=2)
        rng = np.random.default_rng(42)

        strategies = [
            random_partition(instance, n_lots=3, rng=rng),
            single_item_partition(instance),
            category_partition(instance),
            geographic_partition(instance, n_lots=3, rng=np.random.default_rng(42)),
            equal_size_partition(instance, n_lots=3),
        ]

        for partition in strategies:
            _assert_valid_partition(partition, n_items)
