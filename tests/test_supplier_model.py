"""Tests for the three-layer supplier model (cost + participation + markup)."""

import numpy as np
import pytest

from procurement_gym.config import EnvConfig, InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.suppliers.cost import compute_lot_cost
from procurement_gym.suppliers.markup import compute_markup
from procurement_gym.suppliers.participation import compute_participation_prob
from procurement_gym.suppliers.simulator import SimulationResult, ThreeLayerSupplierSimulator


class TestCostModel:
    def test_cost_positive(self) -> None:
        """Cost should be positive for any valid item-supplier pair."""
        ic = InstanceConfig(n_items=5, n_suppliers=3)
        sc = SupplierModelConfig()
        gen = TransportInstanceGenerator(instance_config=ic, supplier_config=sc)
        instance = gen.generate(seed=42)
        rng = np.random.default_rng(42)
        for supplier in instance.suppliers:
            for item in instance.items:
                cost = compute_lot_cost(
                    items=[item],
                    supplier=supplier,
                    config=sc,
                    area_size=ic.area_size,
                    rng=rng,
                )
                assert cost > 0, f"Cost must be positive, got {cost}"

    def test_synergy_reduces_cost(self) -> None:
        """Multiple nearby items should cost less per item than separate items."""
        ic = InstanceConfig(n_items=10, n_suppliers=3, area_size=100.0)
        sc = SupplierModelConfig(synergy_weight=0.1)
        gen = TransportInstanceGenerator(instance_config=ic, supplier_config=sc)
        instance = gen.generate(seed=42)
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)

        supplier = instance.suppliers[0]
        # Pick items with same category as supplier
        cat = next(iter(supplier.categories))
        matching_items = [it for it in instance.items if it.category == cat][:3]
        if len(matching_items) < 2:
            pytest.skip("Not enough matching items for synergy test")

        # Cost of items together
        cost_together = compute_lot_cost(
            items=matching_items,
            supplier=supplier,
            config=sc,
            area_size=ic.area_size,
            rng=rng1,
        )
        # Cost of items separately
        cost_separate = sum(
            compute_lot_cost(
                items=[it],
                supplier=supplier,
                config=sc,
                area_size=ic.area_size,
                rng=rng2,
            )
            for it in matching_items
        )
        # Synergy should make bundled cost <= separate cost (per unit)
        # (synergy is negative cost = savings)
        assert cost_together <= cost_separate * 1.05  # allow noise


class TestParticipationModel:
    def test_probability_in_bounds(self) -> None:
        sc = SupplierModelConfig()
        prob = compute_participation_prob(
            capability_match=1.0,
            lot_size=5,
            n_eligible=10,
            supplier_bias=0.0,
            config=sc,
        )
        assert 0.0 <= prob <= 1.0

    def test_higher_match_higher_prob(self) -> None:
        sc = SupplierModelConfig()
        prob_high = compute_participation_prob(
            capability_match=1.0,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=sc,
        )
        prob_low = compute_participation_prob(
            capability_match=0.0,
            lot_size=5,
            n_eligible=5,
            supplier_bias=0.0,
            config=sc,
        )
        assert prob_high > prob_low

    def test_more_competition_lower_prob(self) -> None:
        """Negative w_comp means more competition → lower participation."""
        sc = SupplierModelConfig(participation_w_comp=-0.3)
        prob_few = compute_participation_prob(
            capability_match=1.0,
            lot_size=5,
            n_eligible=3,
            supplier_bias=0.0,
            config=sc,
        )
        prob_many = compute_participation_prob(
            capability_match=1.0,
            lot_size=5,
            n_eligible=20,
            supplier_bias=0.0,
            config=sc,
        )
        assert prob_few > prob_many


class TestMarkupModel:
    def test_markup_positive(self) -> None:
        markup = compute_markup(
            n_competitors=5,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        assert markup > 0

    def test_more_competition_lower_markup(self) -> None:
        m_few = compute_markup(n_competitors=2, markup_alpha=0.15, markup_beta=0.5)
        m_many = compute_markup(n_competitors=10, markup_alpha=0.15, markup_beta=0.5)
        assert m_few > m_many

    def test_single_competitor(self) -> None:
        m = compute_markup(n_competitors=1, markup_alpha=0.15, markup_beta=0.5)
        assert m == pytest.approx(0.15, rel=0.01)

    def test_zero_competitors_returns_max(self) -> None:
        """With 0 competitors markup should be capped."""
        m = compute_markup(n_competitors=0, markup_alpha=0.15, markup_beta=0.5)
        assert m > 0

    def test_duopoly_markup_lower_than_monopoly(self) -> None:
        """Two bidders total must be more competitive than a monopoly."""
        monopoly_markup = compute_markup(n_competitors=1, markup_alpha=0.15, markup_beta=0.5)
        duopoly_markup = compute_markup(n_competitors=2, markup_alpha=0.15, markup_beta=0.5)
        assert duopoly_markup < monopoly_markup


class TestThreeLayerSimulator:
    def _make_env(
        self, seed: int = 42
    ) -> tuple[
        ThreeLayerSupplierSimulator,
        "TransportInstanceGenerator",
        LotPartition,
    ]:
        ic = InstanceConfig(n_items=10, n_suppliers=5)
        sc = SupplierModelConfig()
        gen = TransportInstanceGenerator(instance_config=ic, supplier_config=sc)
        sim = ThreeLayerSupplierSimulator(config=sc, area_size=ic.area_size)
        return sim, gen, LotPartition.empty(10)

    def test_simulation_result_structure(self) -> None:
        sim, gen, _ = self._make_env()
        instance = gen.generate(seed=42)
        partition = LotPartition.empty(10)
        for i in range(10):
            partition = partition.assign(i, i % 3)  # 3 lots

        rng = np.random.default_rng(42)
        result = sim.simulate(partition, instance, rng)
        assert isinstance(result, SimulationResult)
        assert len(result.bids) == 3  # 3 lots
        assert len(result.entrants) == 3

    def test_bids_exceed_costs(self) -> None:
        """Bids should be costs * (1 + markup), so bids > costs."""
        sim, gen, _ = self._make_env()
        instance = gen.generate(seed=42)
        partition = LotPartition.empty(10)
        for i in range(10):
            partition = partition.assign(i, i % 2)

        rng = np.random.default_rng(42)
        result = sim.simulate(partition, instance, rng)
        for lot_id in result.bids:
            for sid in result.bids[lot_id]:
                if sid in result.costs.get(lot_id, {}):
                    assert result.bids[lot_id][sid] >= result.costs[lot_id][sid]

    def test_reward_variance_across_structures(self) -> None:
        """Same instance with different lot structures should produce >5% reward variance."""
        from procurement_gym.solver.ortools_solver import ORToolsWDPSolver

        # Keep the market fully coverable so variance reflects lot-structure
        # economics rather than impossible multi-category lots.
        ic = InstanceConfig(
            n_items=20,
            n_suppliers=10,
            n_categories=2,
            min_supplier_categories=2,
            max_supplier_categories=2,
        )
        # Moderate participation + strong synergy to amplify lot structure effects
        sc = SupplierModelConfig(
            participation_bias_mean=0.5,
            participation_bias_std=0.3,
            synergy_weight=0.3,
            markup_alpha_mean=0.25,
            markup_alpha_std=0.05,
        )
        gen = TransportInstanceGenerator(instance_config=ic, supplier_config=sc)
        sim = ThreeLayerSupplierSimulator(config=sc, area_size=ic.area_size)
        solver = ORToolsWDPSolver(timeout_s=10.0)
        instance = gen.generate(seed=42)

        rewards: list[float] = []
        # Strategy 1: 2 lots of 10 items each
        p1 = LotPartition.empty(20)
        for i in range(20):
            p1 = p1.assign(i, i // 10)

        # Strategy 2: 5 lots of 4 items each
        p2 = LotPartition.empty(20)
        for i in range(20):
            p2 = p2.assign(i, i // 4)

        # Strategy 3: 10 lots of 2 items each
        p3 = LotPartition.empty(20)
        for i in range(20):
            p3 = p3.assign(i, i // 2)

        for partition in [p1, p2, p3]:
            rng = np.random.default_rng(42)
            sim_result = sim.simulate(partition, instance, rng)

            # Build solver inputs
            bids_dict: dict[int, dict[int, float]] = {}
            entrants_dict: dict[int, frozenset[int]] = {}
            volumes: dict[int, float] = {}
            capacities: dict[int, float] = {s.id: s.capacity for s in instance.suppliers}

            for lot_id in partition.lot_ids:
                bids_dict[lot_id] = sim_result.bids.get(lot_id, {})
                entrants_dict[lot_id] = sim_result.entrants.get(lot_id, frozenset())
                lot_item_idxs = partition.lot_items(lot_id)
                volumes[lot_id] = sum(instance.items[i].volume for i in lot_item_idxs)

            result = solver.solve(
                bids=bids_dict,
                entrants=entrants_dict,
                volumes=volumes,
                capacities=capacities,
            )

            if result.status in ("OPTIMAL", "FEASIBLE"):
                total_value = sum(item.base_value for item in instance.items)
                reward = total_value - result.total_cost
                rewards.append(reward)

        assert len(rewards) >= 2, "At least 2 strategies must produce feasible solutions"
        reward_range = max(rewards) - min(rewards)
        mean_reward = sum(rewards) / len(rewards)
        variance_pct = reward_range / abs(mean_reward) if mean_reward != 0 else 0
        assert variance_pct > 0.05, (
            f"Reward variance {variance_pct:.2%} must be >5%. Rewards: {rewards}"
        )

    def test_mixed_category_lot_requires_full_category_coverage(self) -> None:
        """Suppliers must cover every category in a lot, not just overlap one category."""
        from procurement_gym.core.types import Item, ProcurementInstance, Supplier

        supplier_cat0 = Supplier(
            id=0,
            categories=frozenset({"cat_0"}),
            capacity=100.0,
            base_x=10.0,
            base_y=10.0,
            cost_per_unit_distance=1.0,
            fixed_cost=50.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        supplier_both = Supplier(
            id=1,
            categories=frozenset({"cat_0", "cat_1"}),
            capacity=100.0,
            base_x=20.0,
            base_y=20.0,
            cost_per_unit_distance=1.0,
            fixed_cost=50.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        instance = ProcurementInstance(
            items=(
                Item(
                    id=0,
                    category="cat_0",
                    base_value=100.0,
                    x=0.0,
                    y=0.0,
                    volume=5.0,
                    qualified_supplier_ids=frozenset({0, 1}),
                ),
                Item(
                    id=1,
                    category="cat_1",
                    base_value=100.0,
                    x=1.0,
                    y=1.0,
                    volume=5.0,
                    qualified_supplier_ids=frozenset({1}),
                ),
            ),
            suppliers=(supplier_cat0, supplier_both),
            seed=42,
        )
        partition = LotPartition(assignments=(0, 0), n_items=2)
        sim = ThreeLayerSupplierSimulator(
            config=SupplierModelConfig(coupling_strength=1.0),
            area_size=100.0,
        )

        result = sim.simulate(partition, instance, np.random.default_rng(42))
        entrants = result.entrants[0]
        assert 0 not in entrants
        assert 1 in entrants

    def test_lot_volume_above_supplier_capacity_blocks_entry(self) -> None:
        """A supplier should not enter a lot that exceeds its standalone capacity."""
        from procurement_gym.core.types import Item, ProcurementInstance, Supplier

        supplier_small = Supplier(
            id=0,
            categories=frozenset({"cat_0"}),
            capacity=6.0,
            base_x=10.0,
            base_y=10.0,
            cost_per_unit_distance=1.0,
            fixed_cost=50.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        supplier_large = Supplier(
            id=1,
            categories=frozenset({"cat_0"}),
            capacity=20.0,
            base_x=20.0,
            base_y=20.0,
            cost_per_unit_distance=1.0,
            fixed_cost=50.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        instance = ProcurementInstance(
            items=(
                Item(
                    id=0,
                    category="cat_0",
                    base_value=100.0,
                    x=0.0,
                    y=0.0,
                    volume=5.0,
                    qualified_supplier_ids=frozenset({0, 1}),
                ),
                Item(
                    id=1,
                    category="cat_0",
                    base_value=100.0,
                    x=1.0,
                    y=1.0,
                    volume=5.0,
                    qualified_supplier_ids=frozenset({0, 1}),
                ),
            ),
            suppliers=(supplier_small, supplier_large),
            seed=42,
        )
        partition = LotPartition(assignments=(0, 0), n_items=2)
        sim = ThreeLayerSupplierSimulator(
            config=SupplierModelConfig(coupling_strength=1.0),
            area_size=100.0,
        )

        result = sim.simulate(partition, instance, np.random.default_rng(42))
        entrants = result.entrants[0]
        assert 0 not in entrants
        assert 1 in entrants

    def test_second_entrant_reduces_markup_from_monopoly_level(self) -> None:
        """Two entrants should reduce the bid markup relative to a single entrant."""
        from procurement_gym.core.types import Item, ProcurementInstance, Supplier

        supplier_monopoly = Supplier(
            id=0,
            categories=frozenset({"cat_0"}),
            capacity=20.0,
            base_x=10.0,
            base_y=10.0,
            cost_per_unit_distance=0.0,
            fixed_cost=100.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        supplier_competitor = Supplier(
            id=1,
            categories=frozenset({"cat_0"}),
            capacity=20.0,
            base_x=12.0,
            base_y=12.0,
            cost_per_unit_distance=0.0,
            fixed_cost=100.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        )
        item = Item(
            id=0,
            category="cat_0",
            base_value=100.0,
            x=0.0,
            y=0.0,
            volume=5.0,
            qualified_supplier_ids=frozenset({0, 1}),
        )
        partition = LotPartition(assignments=(0,), n_items=1)
        config = SupplierModelConfig(
            coupling_strength=1.0,
            noise_reduction=1.0,
            participation_bias_mean=10.0,
            participation_coupling_boost=0.0,
            economy_of_scale=0.0,
            synergy_weight=0.0,
        )
        sim = ThreeLayerSupplierSimulator(config=config, area_size=100.0)

        monopoly_instance = ProcurementInstance(items=(item,), suppliers=(supplier_monopoly,), seed=1)
        duopoly_instance = ProcurementInstance(
            items=(item,), suppliers=(supplier_monopoly, supplier_competitor), seed=1
        )

        monopoly_bid = sim.simulate(
            partition, monopoly_instance, np.random.default_rng(1)
        ).bids[0][0]
        duopoly_bid = sim.simulate(
            partition, duopoly_instance, np.random.default_rng(1)
        ).bids[0][0]

        assert duopoly_bid < monopoly_bid
