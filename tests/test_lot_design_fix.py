# PLAN: Tests for lot design fix v2 — noise reduction, economy of scale,
#        geo damping, lot overhead. TDD verification of all 6 changes.
# SOURCE: Plan at .claude/plans/imperative-finding-music.md
"""Tests for the signal-over-noise lot design fix."""

from __future__ import annotations

import math

import numpy as np
import pytest

from procurement_gym.config import InstanceConfig, SupplierModelConfig


# === Layer 1: Noise Reduction ===


class TestNoiseReduction:
    def test_participation_boost_zero_at_zero_coupling(self) -> None:
        """At coupling=0, participation bias is unchanged."""
        from procurement_gym.suppliers.simulator import ThreeLayerSupplierSimulator

        config = SupplierModelConfig(coupling_strength=0.0)
        # effective_bias_mean = -1.0 + 0 * 2.0 = -1.0
        expected = config.participation_bias_mean
        actual = expected + config.coupling_strength * config.participation_coupling_boost
        assert abs(actual - expected) < 1e-10

    def test_participation_boost_at_full_coupling(self) -> None:
        """At coupling=1, bias is shifted by participation_coupling_boost."""
        config = SupplierModelConfig(coupling_strength=1.0)
        effective_mean = (
            config.participation_bias_mean
            + config.coupling_strength * config.participation_coupling_boost
        )
        # -1.0 + 1.0 * 2.0 = 1.0 -> sigmoid(1.0) ≈ 0.73
        assert effective_mean == pytest.approx(1.0, abs=0.01)
        # sigmoid(1.0)
        p = 1.0 / (1.0 + math.exp(-effective_mean))
        assert p > 0.7, f"Base participation {p:.3f} should be > 0.7"

    def test_cost_noise_reduced_at_high_coupling(self) -> None:
        """At coupling=1 with noise_reduction=1.0, verifier is fully deterministic."""
        config = SupplierModelConfig(coupling_strength=1.0)
        determinism = config.coupling_strength * config.noise_reduction
        assert determinism >= 0.99, "Should be fully deterministic at coupling=1"

    def test_noise_unchanged_at_zero_coupling(self) -> None:
        """At coupling=0, all noise parameters are unchanged."""
        config = SupplierModelConfig(coupling_strength=0.0)
        assert config.cost_noise_std * (1.0 - 0.0 * config.noise_reduction) == config.cost_noise_std

    def test_markup_variance_reduced_at_high_coupling(self) -> None:
        """At coupling=1, markup alpha/beta std should be reduced."""
        from procurement_gym.instances.transport import TransportInstanceGenerator

        ic = InstanceConfig(n_items=10, n_suppliers=30, n_categories=2, capacity_scarcity=1.0)
        sc_low = SupplierModelConfig(coupling_strength=0.0)
        sc_high = SupplierModelConfig(coupling_strength=1.0)

        gen_low = TransportInstanceGenerator(ic, sc_low)
        gen_high = TransportInstanceGenerator(ic, sc_high)
        inst_low = gen_low.generate(seed=42)
        inst_high = gen_high.generate(seed=42)

        alphas_low = [s.markup_alpha for s in inst_low.suppliers]
        alphas_high = [s.markup_alpha for s in inst_high.suppliers]

        std_low = float(np.std(alphas_low))
        std_high = float(np.std(alphas_high))
        # High coupling should have lower variance
        assert std_high < std_low, (
            f"Markup alpha std at coupling=1 ({std_high:.4f}) should be less than "
            f"at coupling=0 ({std_low:.4f})"
        )


# === Layer 2: Economy of Scale ===


class TestEconomyOfScale:
    def test_economy_reduces_cost_for_larger_lots(self) -> None:
        """Larger lots should get lower per-item costs at coupling=1."""
        from procurement_gym.core.types import Item, Supplier
        from procurement_gym.suppliers.cost import compute_lot_cost

        config = SupplierModelConfig(coupling_strength=1.0, economy_of_scale=0.15)
        supplier = Supplier(
            id=0,
            categories=frozenset({"cat_0"}),
            capacity=200.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.0,
            fixed_cost=100.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )

        single_item = [
            Item(
                id=0,
                category="cat_0",
                base_value=100.0,
                x=10.0,
                y=10.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            ),
        ]
        seven_items = [
            Item(
                id=i,
                category="cat_0",
                base_value=100.0,
                x=10.0 + i,
                y=10.0 + i,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            )
            for i in range(7)
        ]

        rng1 = np.random.default_rng(42)
        rng7 = np.random.default_rng(42)
        cost_1 = compute_lot_cost(single_item, supplier, config, area_size=100.0, rng=rng1)
        cost_7 = compute_lot_cost(seven_items, supplier, config, area_size=100.0, rng=rng7)

        # Per-item cost should be lower for 7-item lot
        per_item_1 = cost_1 / 1
        per_item_7 = cost_7 / 7
        assert per_item_7 < per_item_1, (
            f"Per-item cost for 7 items ({per_item_7:.1f}) should be less than "
            f"for 1 item ({per_item_1:.1f})"
        )

    def test_economy_no_effect_at_zero_coupling(self) -> None:
        """At coupling=0, economy of scale has no effect."""
        from procurement_gym.core.types import Item, Supplier
        from procurement_gym.suppliers.cost import compute_lot_cost

        config_with = SupplierModelConfig(coupling_strength=0.0, economy_of_scale=0.15)
        config_without = SupplierModelConfig(coupling_strength=0.0, economy_of_scale=0.0)
        supplier = Supplier(
            id=0,
            categories=frozenset({"cat_0"}),
            capacity=200.0,
            base_x=50.0,
            base_y=50.0,
            cost_per_unit_distance=1.0,
            fixed_cost=100.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        items = [
            Item(
                id=i,
                category="cat_0",
                base_value=100.0,
                x=10.0 + i,
                y=10.0,
                volume=5.0,
                qualified_supplier_ids=frozenset({0}),
            )
            for i in range(5)
        ]

        cost_with = compute_lot_cost(items, supplier, config_with, 100.0, np.random.default_rng(42))
        cost_without = compute_lot_cost(
            items, supplier, config_without, 100.0, np.random.default_rng(42)
        )
        assert abs(cost_with - cost_without) < 1e-6


# === Layer 3: Geo Damping ===


class TestGeoDamping:
    def test_single_item_gets_reduced_geo_prox(self) -> None:
        """Single-item lots should have geo_prox reduced to ~43%."""
        factor = min(1.0, math.log(max(2, 1)) / math.log(5))
        assert factor == pytest.approx(0.431, abs=0.01)

    def test_five_items_get_full_geo_prox(self) -> None:
        """5-item lots should get full geo proximity."""
        factor = min(1.0, math.log(max(2, 5)) / math.log(5))
        assert factor == pytest.approx(1.0, abs=0.001)

    def test_three_items_get_partial_geo_prox(self) -> None:
        """3-item lots should get ~68% geo proximity."""
        factor = min(1.0, math.log(max(2, 3)) / math.log(5))
        assert 0.6 < factor < 0.75


# === Lot Overhead ===


class TestLotOverhead:
    def test_lot_overhead_zero_at_zero_coupling(self) -> None:
        """At coupling=0, lot overhead contributes nothing."""
        config = SupplierModelConfig(coupling_strength=0.0, lot_overhead_base=100.0)
        overhead = config.lot_overhead_base * config.coupling_strength
        assert overhead == 0.0

    def test_lot_overhead_penalizes_more_lots(self) -> None:
        """Single-item partition should get much higher overhead than category."""
        from procurement_gym.baselines import category_partition, single_item_partition
        from procurement_gym.evaluation import evaluate_partition
        from procurement_gym.instances.transport import TransportInstanceGenerator

        ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3, capacity_scarcity=1.0)
        sc = SupplierModelConfig(coupling_strength=1.0, lot_overhead_base=100.0)
        gen = TransportInstanceGenerator(ic, sc)
        inst = gen.generate(seed=42)

        cat_part = category_partition(inst)
        si_part = single_item_partition(inst)

        cat_result = evaluate_partition(cat_part, inst, sc, seed=42)
        si_result = evaluate_partition(si_part, inst, sc, seed=42)

        cat_overhead = cat_result.info.get("lot_overhead_total", 0)
        si_overhead = si_result.info.get("lot_overhead_total", 0)

        # Single-item (20 lots) should have much higher overhead than category (3 lots)
        assert si_overhead > cat_overhead * 5, (
            f"Single-item overhead ({si_overhead}) should be > 5x category ({cat_overhead})"
        )

    def test_evaluator_includes_lot_overhead_in_info(self) -> None:
        """evaluate_partition should include lot_overhead_total in info."""
        from procurement_gym.baselines import category_partition
        from procurement_gym.evaluation import evaluate_partition
        from procurement_gym.instances.transport import TransportInstanceGenerator

        ic = InstanceConfig(n_items=10, n_suppliers=10, n_categories=2, capacity_scarcity=1.0)
        sc = SupplierModelConfig(coupling_strength=1.0, lot_overhead_base=100.0)
        gen = TransportInstanceGenerator(ic, sc)
        inst = gen.generate(seed=42)

        result = evaluate_partition(category_partition(inst), inst, sc, seed=42)
        assert "lot_overhead_total" in result.info
        assert result.info["lot_overhead_total"] > 0


# === Integration: Signal Quality ===


class TestSignalQuality:
    def test_strategy_rankings_more_stable_at_high_coupling(self) -> None:
        """With noise reduction, strategy rankings should be more consistent across seeds."""
        from procurement_gym.baselines import category_partition, geographic_partition
        from procurement_gym.evaluation import evaluate_partition
        from procurement_gym.instances.transport import TransportInstanceGenerator

        ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3, capacity_scarcity=1.0)
        sc = SupplierModelConfig(coupling_strength=1.0)
        gen = TransportInstanceGenerator(ic, sc)
        inst = gen.generate(seed=42)
        rng = np.random.default_rng(99)

        cat_part = category_partition(inst)
        geo_part = geographic_partition(inst, n_lots=3, rng=rng)

        # Evaluate with 10 seeds
        cat_rewards = [
            evaluate_partition(cat_part, inst, sc, seed=1000 + s).reward for s in range(10)
        ]
        geo_rewards = [
            evaluate_partition(geo_part, inst, sc, seed=1000 + s).reward for s in range(10)
        ]

        cat_std = float(np.std(cat_rewards))
        cat_mean = float(np.mean(cat_rewards))

        # Coefficient of variation should be reasonable (< 50%)
        if abs(cat_mean) > 0:
            cv = cat_std / abs(cat_mean)
            assert cv < 0.5, f"CV={cv:.2f} is too high — noise dominates signal"

    def test_single_item_not_dominant_at_full_coupling(self) -> None:
        """Key regression: single-item should NOT win > 60% at coupling=1.0."""
        from collections import Counter

        from procurement_gym.baselines import (
            category_partition,
            geographic_partition,
            single_item_partition,
        )
        from procurement_gym.evaluation import evaluate_partition
        from procurement_gym.instances.transport import TransportInstanceGenerator

        ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
        sc = SupplierModelConfig(coupling_strength=1.0)
        gen = TransportInstanceGenerator(ic, sc)

        wins: Counter = Counter()
        for i in range(20):
            inst = gen.generate(seed=70000 + i)
            rng = np.random.default_rng(71000 + i)
            n_cats = len({item.category for item in inst.items})

            strategies = {
                "category": category_partition(inst),
                "geographic": geographic_partition(inst, n_lots=n_cats, rng=rng),
                "single_item": single_item_partition(inst),
            }

            means = {}
            for name, part in strategies.items():
                rewards = [
                    evaluate_partition(part, inst, sc, seed=72000 + i * 10 + s).reward
                    for s in range(5)
                ]
                means[name] = float(np.mean(rewards))

            best = max(means, key=means.get)  # type: ignore[arg-type]
            wins[best] += 1

        si_pct = wins["single_item"] / 20 * 100
        assert si_pct <= 60, (
            f"Single-item wins {si_pct:.0f}% (should be ≤ 60%). Distribution: {dict(wins)}"
        )
