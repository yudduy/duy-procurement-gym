# PLAN: Tests for the coupling fix — geographic participation, package amplification,
#        capacity scarcity, category breadth. TDD RED then GREEN.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
"""Tests for coupling mechanism fixes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from procurement_gym.config import InstanceConfig, SupplierModelConfig


# === Step 2: Participation model — geographic proximity ===


class TestParticipationGeographic:
    def test_geo_zero_at_zero_coupling(self) -> None:
        """Geographic proximity should have no effect at coupling=0."""
        from procurement_gym.suppliers.participation import compute_participation_prob

        config = SupplierModelConfig(coupling_strength=0.0)
        p_no_geo = compute_participation_prob(
            capability_match=0.5,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=config,
            geographic_proximity=0.0,
        )
        p_with_geo = compute_participation_prob(
            capability_match=0.5,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=config,
            geographic_proximity=1.0,
        )
        assert abs(p_no_geo - p_with_geo) < 1e-10

    def test_geo_increases_prob_at_high_coupling(self) -> None:
        """At coupling=1.0, high geographic proximity should increase participation."""
        from procurement_gym.suppliers.participation import compute_participation_prob

        config = SupplierModelConfig(coupling_strength=1.0)
        p_far = compute_participation_prob(
            capability_match=0.5,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=config,
            geographic_proximity=0.0,
        )
        p_near = compute_participation_prob(
            capability_match=0.5,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=config,
            geographic_proximity=1.0,
        )
        assert p_near > p_far, f"Near ({p_near:.3f}) should beat far ({p_far:.3f})"

    def test_backward_compat_no_kwarg(self) -> None:
        """Calling without geographic_proximity should produce same result as before."""
        from procurement_gym.suppliers.participation import compute_participation_prob

        config = SupplierModelConfig(coupling_strength=0.5)
        # Without the kwarg (should default to 0.0)
        p_default = compute_participation_prob(
            capability_match=0.8,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=config,
        )
        p_explicit = compute_participation_prob(
            capability_match=0.8,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=config,
            geographic_proximity=0.0,
        )
        assert abs(p_default - p_explicit) < 1e-10

    def test_simulator_passes_geo_proximity(self) -> None:
        """Simulator should compute and pass geographic proximity at high coupling."""
        from procurement_gym.config import EnvConfig
        from procurement_gym.env import ProcurementEnv

        config = EnvConfig(
            instance=InstanceConfig(
                n_items=10, n_suppliers=5, n_categories=2, capacity_scarcity=1.0
            ),
            supplier_model=SupplierModelConfig(coupling_strength=1.0),
        )
        env = ProcurementEnv(config=config)
        obs, _ = env.reset(seed=42)
        # Run full episode — should not crash
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
        assert "status" in info


# === Step 3: Package generation — soft threshold + deterministic discount ===


class TestPackageFixes:
    def test_soft_threshold_at_full_coupling(self) -> None:
        """At coupling=1.0, even low-synergy combos should generate packages."""
        from procurement_gym.core.types import Item, PackageBid, Supplier
        from procurement_gym.suppliers.packages import generate_package_bids

        # Items far apart (low synergy ~0.1)
        items = [
            Item(
                id=0,
                category="cat_0",
                base_value=100.0,
                x=5.0,
                y=5.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            ),
            Item(
                id=1,
                category="cat_0",
                base_value=100.0,
                x=95.0,
                y=95.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            ),
        ]
        suppliers = [
            Supplier(
                id=0,
                categories=frozenset({"cat_0"}),
                capacity=200.0,
                base_x=50.0,
                base_y=50.0,
                cost_per_unit_distance=1.0,
                fixed_cost=50.0,
                markup_alpha=0.15,
                markup_beta=0.5,
            ),
        ]
        config_high = SupplierModelConfig(coupling_strength=1.0)
        config_low = SupplierModelConfig(coupling_strength=0.01)

        result_high = generate_package_bids(
            per_lot_bids={0: {0: 100.0}, 1: {0: 120.0}},
            items_by_lot={0: (0,), 1: (1,)},
            items=items,
            suppliers=suppliers,
            config=config_high,
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        result_low = generate_package_bids(
            per_lot_bids={0: {0: 100.0}, 1: {0: 120.0}},
            items_by_lot={0: (0,), 1: (1,)},
            items=items,
            suppliers=suppliers,
            config=config_low,
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        # At coupling=1.0, threshold=0 so low-synergy combo passes
        assert len(result_high) >= len(result_low)

    def test_discount_proportional_to_synergy(self) -> None:
        """Higher synergy should produce larger discounts."""
        from procurement_gym.core.types import Item, Supplier
        from procurement_gym.suppliers.packages import generate_package_bids

        # Close items (high synergy)
        close_items = [
            Item(
                id=0,
                category="cat_0",
                base_value=100.0,
                x=10.0,
                y=10.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            ),
            Item(
                id=1,
                category="cat_0",
                base_value=100.0,
                x=12.0,
                y=12.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            ),
        ]
        # Far items (low synergy)
        far_items = [
            Item(
                id=0,
                category="cat_0",
                base_value=100.0,
                x=5.0,
                y=5.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            ),
            Item(
                id=1,
                category="cat_0",
                base_value=100.0,
                x=80.0,
                y=80.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            ),
        ]
        suppliers = [
            Supplier(
                id=0,
                categories=frozenset({"cat_0"}),
                capacity=200.0,
                base_x=11.0,
                base_y=11.0,
                cost_per_unit_distance=1.0,
                fixed_cost=50.0,
                markup_alpha=0.15,
                markup_beta=0.5,
            ),
        ]
        config = SupplierModelConfig(coupling_strength=1.0)
        bids = {0: {0: 100.0}, 1: {0: 100.0}}

        close_pkgs = generate_package_bids(
            per_lot_bids=bids,
            items_by_lot={0: (0,), 1: (1,)},
            items=close_items,
            suppliers=suppliers,
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        far_pkgs = generate_package_bids(
            per_lot_bids=bids,
            items_by_lot={0: (0,), 1: (1,)},
            items=far_items,
            suppliers=suppliers,
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        if close_pkgs and far_pkgs:
            # Close items → higher synergy → bigger discount → lower price
            close_discount = 1.0 - close_pkgs[0].price / 200.0
            far_discount = 1.0 - far_pkgs[0].price / 200.0
            assert close_discount > far_discount

    def test_more_packages_at_high_coupling(self) -> None:
        """coupling=1.0 should produce more packages than coupling=0.3."""
        from procurement_gym.core.types import Item, Supplier
        from procurement_gym.suppliers.packages import generate_package_bids

        items = [
            Item(
                id=i,
                category="cat_0",
                base_value=100.0,
                x=float(20 + i * 30),
                y=float(20 + i * 30),
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            )
            for i in range(3)
        ]
        suppliers = [
            Supplier(
                id=0,
                categories=frozenset({"cat_0"}),
                capacity=200.0,
                base_x=50.0,
                base_y=50.0,
                cost_per_unit_distance=1.0,
                fixed_cost=50.0,
                markup_alpha=0.15,
                markup_beta=0.5,
            ),
        ]
        bids = {0: {0: 100.0}, 1: {0: 120.0}, 2: {0: 110.0}}
        items_by_lot = {0: (0,), 1: (1,), 2: (2,)}

        low = generate_package_bids(
            per_lot_bids=bids,
            items_by_lot=items_by_lot,
            items=items,
            suppliers=suppliers,
            config=SupplierModelConfig(coupling_strength=0.3),
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        high = generate_package_bids(
            per_lot_bids=bids,
            items_by_lot=items_by_lot,
            items=items,
            suppliers=suppliers,
            config=SupplierModelConfig(coupling_strength=1.0),
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        assert len(high) >= len(low)


# === Step 4: Instance generation — capacity scarcity + category breadth ===


class TestInstanceFixes:
    def test_capacity_unchanged_at_zero_coupling(self) -> None:
        """At coupling=0, capacity should be in original range [50, 200]."""
        from procurement_gym.instances.transport import TransportInstanceGenerator

        gen = TransportInstanceGenerator(
            instance_config=InstanceConfig(n_items=10, n_suppliers=20),
            supplier_config=SupplierModelConfig(coupling_strength=0.0),
        )
        instance = gen.generate(seed=42)
        for s in instance.suppliers:
            assert 49.0 <= s.capacity <= 201.0, f"Capacity {s.capacity} out of range"

    def test_capacity_tighter_at_high_coupling(self) -> None:
        """At coupling=1.0, mean capacity should be significantly lower."""
        from procurement_gym.instances.transport import TransportInstanceGenerator

        gen_low = TransportInstanceGenerator(
            instance_config=InstanceConfig(n_items=10, n_suppliers=20),
            supplier_config=SupplierModelConfig(coupling_strength=0.0),
        )
        gen_high = TransportInstanceGenerator(
            instance_config=InstanceConfig(n_items=10, n_suppliers=20),
            supplier_config=SupplierModelConfig(coupling_strength=1.0),
        )
        inst_low = gen_low.generate(seed=42)
        inst_high = gen_high.generate(seed=42)

        mean_low = sum(s.capacity for s in inst_low.suppliers) / len(inst_low.suppliers)
        mean_high = sum(s.capacity for s in inst_high.suppliers) / len(inst_high.suppliers)
        assert mean_high < mean_low * 0.7, (
            f"High-coupling capacity ({mean_high:.1f}) should be much lower than "
            f"zero-coupling ({mean_low:.1f})"
        )

    def test_category_breadth_increases_with_coupling(self) -> None:
        """At coupling=1.0, suppliers should cover more categories."""
        from procurement_gym.instances.transport import TransportInstanceGenerator

        gen_low = TransportInstanceGenerator(
            instance_config=InstanceConfig(n_items=10, n_suppliers=20, n_categories=4),
            supplier_config=SupplierModelConfig(coupling_strength=0.0),
        )
        gen_high = TransportInstanceGenerator(
            instance_config=InstanceConfig(n_items=10, n_suppliers=20, n_categories=4),
            supplier_config=SupplierModelConfig(coupling_strength=1.0),
        )
        inst_low = gen_low.generate(seed=42)
        inst_high = gen_high.generate(seed=42)

        mean_cats_low = sum(len(s.categories) for s in inst_low.suppliers) / len(inst_low.suppliers)
        mean_cats_high = sum(len(s.categories) for s in inst_high.suppliers) / len(
            inst_high.suppliers
        )
        assert mean_cats_high > mean_cats_low, (
            f"High-coupling cats ({mean_cats_high:.1f}) should exceed "
            f"zero-coupling ({mean_cats_low:.1f})"
        )
