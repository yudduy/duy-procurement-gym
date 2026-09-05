"""Tests for frozen dataclasses: Item, Supplier, ProcurementInstance."""

import pytest

from procurement_gym.core.types import Item, Supplier, ProcurementInstance


class TestItem:
    def test_construction(self) -> None:
        item = Item(
            id=0,
            category="freight",
            base_value=1000.0,
            x=10.0,
            y=20.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0, 1, 2}),
        )
        assert item.id == 0
        assert item.category == "freight"
        assert item.base_value == 1000.0
        assert item.x == 10.0
        assert item.y == 20.0
        assert item.volume == 5.0
        assert item.qualified_supplier_ids == frozenset({0, 1, 2})

    def test_frozen(self) -> None:
        item = Item(
            id=0,
            category="freight",
            base_value=1000.0,
            x=10.0,
            y=20.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0}),
        )
        with pytest.raises(AttributeError):
            item.id = 1  # type: ignore[misc]
        with pytest.raises(AttributeError):
            item.base_value = 999.0  # type: ignore[misc]

    def test_equality(self) -> None:
        kwargs = dict(
            id=0,
            category="freight",
            base_value=1000.0,
            x=10.0,
            y=20.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0}),
        )
        assert Item(**kwargs) == Item(**kwargs)

    def test_hashable(self) -> None:
        item = Item(
            id=0,
            category="freight",
            base_value=1000.0,
            x=10.0,
            y=20.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0}),
        )
        assert hash(item) is not None
        assert item in {item}


class TestSupplier:
    def test_construction(self) -> None:
        supplier = Supplier(
            id=0,
            categories=frozenset({"freight", "express"}),
            capacity=100.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.5,
            fixed_cost=200.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        assert supplier.id == 0
        assert "freight" in supplier.categories
        assert supplier.capacity == 100.0
        assert supplier.markup_alpha == 0.15

    def test_frozen(self) -> None:
        supplier = Supplier(
            id=0,
            categories=frozenset({"freight"}),
            capacity=100.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.5,
            fixed_cost=200.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        with pytest.raises(AttributeError):
            supplier.capacity = 200.0  # type: ignore[misc]

    def test_hashable(self) -> None:
        supplier = Supplier(
            id=0,
            categories=frozenset({"freight"}),
            capacity=100.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.5,
            fixed_cost=200.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        assert hash(supplier) is not None


class TestProcurementInstance:
    def test_construction(self) -> None:
        item = Item(
            id=0,
            category="freight",
            base_value=1000.0,
            x=10.0,
            y=20.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0}),
        )
        supplier = Supplier(
            id=0,
            categories=frozenset({"freight"}),
            capacity=100.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.5,
            fixed_cost=200.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        instance = ProcurementInstance(
            items=(item,),
            suppliers=(supplier,),
            seed=42,
        )
        assert len(instance.items) == 1
        assert len(instance.suppliers) == 1
        assert instance.seed == 42

    def test_frozen(self) -> None:
        item = Item(
            id=0,
            category="freight",
            base_value=1000.0,
            x=10.0,
            y=20.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0}),
        )
        supplier = Supplier(
            id=0,
            categories=frozenset({"freight"}),
            capacity=100.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.5,
            fixed_cost=200.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        instance = ProcurementInstance(
            items=(item,),
            suppliers=(supplier,),
            seed=42,
        )
        with pytest.raises(AttributeError):
            instance.seed = 99  # type: ignore[misc]

    def test_tuple_immutability(self) -> None:
        """Items and suppliers are tuples, not lists — cannot be mutated."""
        item = Item(
            id=0,
            category="freight",
            base_value=1000.0,
            x=10.0,
            y=20.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0}),
        )
        supplier = Supplier(
            id=0,
            categories=frozenset({"freight"}),
            capacity=100.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.5,
            fixed_cost=200.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        instance = ProcurementInstance(
            items=(item,),
            suppliers=(supplier,),
            seed=42,
        )
        assert isinstance(instance.items, tuple)
        assert isinstance(instance.suppliers, tuple)
