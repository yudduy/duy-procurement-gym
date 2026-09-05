"""Tests for TransportInstanceGenerator."""

from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.instances.transport import TransportInstanceGenerator


class TestTransportGenerator:
    def _make_generator(self, **kwargs: object) -> TransportInstanceGenerator:
        return TransportInstanceGenerator(
            instance_config=InstanceConfig(**kwargs),  # type: ignore[arg-type]
            supplier_config=SupplierModelConfig(),
        )

    def test_correct_item_count(self) -> None:
        gen = self._make_generator(n_items=10, n_suppliers=5)
        instance = gen.generate(seed=42)
        assert len(instance.items) == 10

    def test_correct_supplier_count(self) -> None:
        gen = self._make_generator(n_items=10, n_suppliers=5)
        instance = gen.generate(seed=42)
        assert len(instance.suppliers) == 5

    def test_item_coordinates_in_bounds(self) -> None:
        gen = self._make_generator(n_items=20, n_suppliers=5, area_size=100.0)
        instance = gen.generate(seed=42)
        for item in instance.items:
            assert 0.0 <= item.x <= 100.0
            assert 0.0 <= item.y <= 100.0

    def test_supplier_coordinates_in_bounds(self) -> None:
        gen = self._make_generator(n_items=10, n_suppliers=5, area_size=100.0)
        instance = gen.generate(seed=42)
        for supplier in instance.suppliers:
            assert 0.0 <= supplier.base_x <= 100.0
            assert 0.0 <= supplier.base_y <= 100.0

    def test_items_have_valid_categories(self) -> None:
        gen = self._make_generator(n_items=20, n_suppliers=5, n_categories=3)
        instance = gen.generate(seed=42)
        for item in instance.items:
            assert item.category in {f"cat_{i}" for i in range(3)}

    def test_supplier_categories_cover_items(self) -> None:
        """Every item category should be serviceable by at least one supplier."""
        gen = self._make_generator(n_items=20, n_suppliers=10, n_categories=3)
        instance = gen.generate(seed=42)
        all_supplier_cats = set()
        for s in instance.suppliers:
            all_supplier_cats.update(s.categories)
        item_cats = {item.category for item in instance.items}
        assert item_cats.issubset(all_supplier_cats)

    def test_qualified_supplier_ids_valid(self) -> None:
        """Each item's qualified_supplier_ids should be non-empty and reference real suppliers."""
        gen = self._make_generator(n_items=20, n_suppliers=10)
        instance = gen.generate(seed=42)
        supplier_ids = {s.id for s in instance.suppliers}
        for item in instance.items:
            assert len(item.qualified_supplier_ids) > 0
            assert item.qualified_supplier_ids.issubset(supplier_ids)

    def test_seeded_reproducibility(self) -> None:
        gen = self._make_generator(n_items=10, n_suppliers=5)
        i1 = gen.generate(seed=123)
        i2 = gen.generate(seed=123)
        assert i1 == i2

    def test_different_seeds_different_instances(self) -> None:
        gen = self._make_generator(n_items=10, n_suppliers=5)
        i1 = gen.generate(seed=1)
        i2 = gen.generate(seed=2)
        assert i1 != i2

    def test_positive_values(self) -> None:
        gen = self._make_generator(n_items=20, n_suppliers=5)
        instance = gen.generate(seed=42)
        for item in instance.items:
            assert item.base_value > 0
            assert item.volume > 0
        for supplier in instance.suppliers:
            assert supplier.capacity > 0
            assert supplier.cost_per_unit_distance > 0

    def test_seed_stored_in_instance(self) -> None:
        gen = self._make_generator(n_items=5, n_suppliers=3)
        instance = gen.generate(seed=99)
        assert instance.seed == 99
