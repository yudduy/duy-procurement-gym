"""Tests for standalone reward evaluator."""

from __future__ import annotations

import numpy as np
import pytest

from procurement_gym.baselines import category_partition, random_partition, single_item_partition
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.instances.transport import TransportInstanceGenerator


def _make_instance(
    n_items: int = 10,
    n_suppliers: int = 5,
    n_categories: int = 2,
    seed: int = 42,
) -> "ProcurementInstance":
    from procurement_gym.core.types import ProcurementInstance

    gen = TransportInstanceGenerator(
        instance_config=InstanceConfig(
            n_items=n_items, n_suppliers=n_suppliers, n_categories=n_categories
        ),
        supplier_config=SupplierModelConfig(),
    )
    return gen.generate(seed=seed)


class TestEvaluatePartition:
    def test_returns_float_reward(self) -> None:
        from procurement_gym.evaluation import evaluate_partition

        instance = _make_instance()
        partition = category_partition(instance)
        config = SupplierModelConfig()
        result = evaluate_partition(partition, instance, config, seed=0)
        assert isinstance(result.reward, float)

    def test_returns_info_dict(self) -> None:
        from procurement_gym.evaluation import evaluate_partition

        instance = _make_instance()
        partition = category_partition(instance)
        config = SupplierModelConfig()
        result = evaluate_partition(partition, instance, config, seed=0)
        assert "status" in result.info
        assert "total_cost" in result.info

    def test_deterministic_same_seed(self) -> None:
        from procurement_gym.evaluation import evaluate_partition

        instance = _make_instance()
        partition = category_partition(instance)
        config = SupplierModelConfig()
        r1 = evaluate_partition(partition, instance, config, seed=42)
        r2 = evaluate_partition(partition, instance, config, seed=42)
        assert r1.reward == r2.reward

    def test_different_seeds_may_differ(self) -> None:
        from procurement_gym.evaluation import evaluate_partition

        instance = _make_instance()
        partition = category_partition(instance)
        config = SupplierModelConfig()
        rewards = [
            evaluate_partition(partition, instance, config, seed=s).reward for s in range(10)
        ]
        # With stochastic participation, not all seeds should give identical rewards
        assert len(set(rewards)) > 1, "Different seeds should produce some variation"

    def test_better_partition_scores_higher_on_average(self) -> None:
        """Category partition should beat random on average at coupling=0."""
        from procurement_gym.evaluation import evaluate_partition

        instance = _make_instance(n_items=20, n_suppliers=10, n_categories=3)
        config = SupplierModelConfig(coupling_strength=0.0)

        cat_part = category_partition(instance)
        rng = np.random.default_rng(0)
        rand_part = random_partition(instance, n_lots=5, rng=rng)

        cat_rewards = [
            evaluate_partition(cat_part, instance, config, seed=s).reward for s in range(20)
        ]
        rand_rewards = [
            evaluate_partition(rand_part, instance, config, seed=s).reward for s in range(20)
        ]

        assert np.mean(cat_rewards) > np.mean(rand_rewards), (
            f"Category ({np.mean(cat_rewards):.1f}) should beat random ({np.mean(rand_rewards):.1f})"
        )

    def test_works_with_coupling(self) -> None:
        from procurement_gym.evaluation import evaluate_partition

        instance = _make_instance(n_items=10, n_suppliers=5, n_categories=2)
        config = SupplierModelConfig(coupling_strength=0.8)
        partition = category_partition(instance)
        result = evaluate_partition(partition, instance, config, seed=0)
        assert isinstance(result.reward, float)
        assert "n_package_bids" in result.info

    def test_all_unserviced_info_keeps_package_fields(self) -> None:
        from procurement_gym.core.types import Item, ProcurementInstance, Supplier
        from procurement_gym.evaluation import evaluate_partition

        instance = ProcurementInstance(
            items=(
                Item(
                    id=0,
                    category="cat_0",
                    base_value=100.0,
                    x=0.0,
                    y=0.0,
                    volume=5.0,
                    qualified_supplier_ids=frozenset({0}),
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
                Item(
                    id=2,
                    category="cat_2",
                    base_value=100.0,
                    x=2.0,
                    y=2.0,
                    volume=5.0,
                    qualified_supplier_ids=frozenset({2}),
                ),
            ),
            suppliers=(
                Supplier(
                    id=0,
                    categories=frozenset({"cat_0"}),
                    capacity=20.0,
                    base_x=0.0,
                    base_y=0.0,
                    cost_per_unit_distance=1.0,
                    fixed_cost=50.0,
                    markup_alpha=0.15,
                    markup_beta=0.5,
                ),
                Supplier(
                    id=1,
                    categories=frozenset({"cat_1"}),
                    capacity=20.0,
                    base_x=10.0,
                    base_y=10.0,
                    cost_per_unit_distance=1.0,
                    fixed_cost=50.0,
                    markup_alpha=0.15,
                    markup_beta=0.5,
                ),
                Supplier(
                    id=2,
                    categories=frozenset({"cat_2"}),
                    capacity=20.0,
                    base_x=20.0,
                    base_y=20.0,
                    cost_per_unit_distance=1.0,
                    fixed_cost=50.0,
                    markup_alpha=0.15,
                    markup_beta=0.5,
                ),
            ),
            seed=42,
        )
        partition = LotPartition(assignments=(0, 0, 0), n_items=3)

        result = evaluate_partition(partition, instance, SupplierModelConfig(), seed=0)
        assert result.info["status"] == "ALL_UNSERVICED"
        assert result.info["n_package_bids"] == 0
        assert result.info["n_package_awards"] == 0


class TestEvaluateMultiSeed:
    def test_returns_stats(self) -> None:
        from procurement_gym.evaluation import evaluate_multi_seed

        instance = _make_instance()
        partition = category_partition(instance)
        config = SupplierModelConfig()
        stats = evaluate_multi_seed(partition, instance, config, n_seeds=10, base_seed=0)
        assert "mean" in stats
        assert "std" in stats
        assert "rewards" in stats
        assert len(stats["rewards"]) == 10

    def test_std_is_nonnegative(self) -> None:
        from procurement_gym.evaluation import evaluate_multi_seed

        instance = _make_instance()
        partition = category_partition(instance)
        config = SupplierModelConfig()
        stats = evaluate_multi_seed(partition, instance, config, n_seeds=10, base_seed=0)
        assert stats["std"] >= 0.0
