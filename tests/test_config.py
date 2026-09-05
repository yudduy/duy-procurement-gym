"""Tests for Pydantic configuration schema."""

import pytest

from procurement_gym.config import EnvConfig, InstanceConfig, SupplierModelConfig


class TestInstanceConfig:
    def test_defaults(self) -> None:
        c = InstanceConfig()
        assert c.n_items == 20
        assert c.n_suppliers == 10
        assert c.n_categories == 3
        assert c.area_size == 100.0

    def test_validation_bounds(self) -> None:
        with pytest.raises(Exception):
            InstanceConfig(n_items=1)  # min 2
        with pytest.raises(Exception):
            InstanceConfig(n_items=501)  # max 500
        with pytest.raises(Exception):
            InstanceConfig(n_suppliers=1)  # min 2


class TestSupplierModelConfig:
    def test_defaults(self) -> None:
        c = SupplierModelConfig()
        assert c.synergy_weight == 0.1
        assert c.markup_alpha_mean == 0.15
        assert c.markup_beta_mean == 0.5
        assert c.cost_noise_std == 0.05

    def test_participation_bias_fields(self) -> None:
        c = SupplierModelConfig()
        assert hasattr(c, "participation_bias_mean")
        assert hasattr(c, "participation_bias_std")

    def test_category_clamping(self) -> None:
        """max_supplier_categories > n_categories should be clamped, not crash."""
        c = InstanceConfig(n_categories=1, max_supplier_categories=2)
        assert c.max_supplier_categories <= c.n_categories
        assert c.min_supplier_categories <= c.max_supplier_categories

    def test_category_clamping_min_above_max(self) -> None:
        c = InstanceConfig(n_categories=1, min_supplier_categories=3, max_supplier_categories=5)
        assert c.max_supplier_categories <= c.n_categories
        assert c.min_supplier_categories <= c.max_supplier_categories


class TestEnvConfig:
    def test_defaults(self) -> None:
        c = EnvConfig()
        assert c.k_max == 20
        assert c.solver_timeout_s == 60.0
        assert c.item_order == "ascending_qualified"
        assert c.seed == 42
        assert isinstance(c.instance, InstanceConfig)
        assert isinstance(c.supplier_model, SupplierModelConfig)

    def test_nested_override(self) -> None:
        c = EnvConfig(instance=InstanceConfig(n_items=50))
        assert c.instance.n_items == 50
        assert c.instance.n_suppliers == 10  # default preserved
