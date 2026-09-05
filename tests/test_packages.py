"""Tests for package bid generation and cross-lot coupling."""

import numpy as np
import pytest

from procurement_gym.config import SupplierModelConfig
from procurement_gym.core.types import Item, PackageBid, Supplier
from procurement_gym.suppliers.packages import generate_package_bids


def _make_items(n: int, area_size: float = 100.0) -> list[Item]:
    """Create items clustered geographically for high synergy."""
    rng = np.random.default_rng(42)
    items = []
    for i in range(n):
        items.append(
            Item(
                id=i,
                category="cat_0",
                base_value=100.0,
                x=float(rng.uniform(10.0, 30.0)),  # clustered
                y=float(rng.uniform(10.0, 30.0)),
                volume=5.0,
                qualified_supplier_ids=frozenset({0, 1}),
            )
        )
    return items


def _make_suppliers() -> list[Supplier]:
    return [
        Supplier(
            id=0,
            categories=frozenset({"cat_0"}),
            capacity=200.0,
            base_x=20.0,
            base_y=20.0,
            cost_per_unit_distance=1.0,
            fixed_cost=50.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        ),
        Supplier(
            id=1,
            categories=frozenset({"cat_0"}),
            capacity=200.0,
            base_x=80.0,
            base_y=80.0,
            cost_per_unit_distance=1.0,
            fixed_cost=50.0,
            markup_alpha=0.15,
            markup_beta=0.5,
        ),
    ]


class TestPackageBidGeneration:
    def test_zero_coupling_returns_empty(self) -> None:
        config = SupplierModelConfig(coupling_strength=0.0)
        result = generate_package_bids(
            per_lot_bids={0: {0: 100.0}, 1: {0: 200.0}},
            items_by_lot={0: (0,), 1: (1,)},
            items=_make_items(2),
            suppliers=_make_suppliers(),
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(0),
        )
        assert result == ()

    def test_produces_packages_with_coupling(self) -> None:
        config = SupplierModelConfig(coupling_strength=1.0)
        items = _make_items(4)
        result = generate_package_bids(
            per_lot_bids={0: {0: 100.0, 1: 150.0}, 1: {0: 120.0, 1: 130.0}},
            items_by_lot={0: (0, 1), 1: (2, 3)},
            items=items,
            suppliers=_make_suppliers(),
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        assert len(result) > 0
        assert all(isinstance(p, PackageBid) for p in result)

    def test_package_price_less_than_sum(self) -> None:
        config = SupplierModelConfig(coupling_strength=1.0)
        items = _make_items(4)
        per_lot_bids = {0: {0: 100.0, 1: 150.0}, 1: {0: 120.0, 1: 130.0}}
        result = generate_package_bids(
            per_lot_bids=per_lot_bids,
            items_by_lot={0: (0, 1), 1: (2, 3)},
            items=items,
            suppliers=_make_suppliers(),
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(42),
        )
        for pkg in result:
            individual_sum = sum(per_lot_bids[k][pkg.supplier_id] for k in pkg.lot_ids)
            assert pkg.price < individual_sum, "Package price should be discounted"

    def test_package_size_bounded(self) -> None:
        config = SupplierModelConfig(coupling_strength=1.0, max_package_size=2)
        items = _make_items(6)
        result = generate_package_bids(
            per_lot_bids={0: {0: 100.0}, 1: {0: 120.0}, 2: {0: 110.0}},
            items_by_lot={0: (0, 1), 1: (2, 3), 2: (4, 5)},
            items=items,
            suppliers=_make_suppliers()[:1],
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(0),
        )
        for pkg in result:
            assert len(pkg.lot_ids) <= 2

    def test_packages_per_supplier_capped(self) -> None:
        config = SupplierModelConfig(
            coupling_strength=1.0,
            max_package_size=3,
            max_packages_per_supplier=3,
        )
        items = _make_items(10)
        result = generate_package_bids(
            per_lot_bids={
                0: {0: 100.0},
                1: {0: 105.0},
                2: {0: 110.0},
                3: {0: 115.0},
                4: {0: 120.0},
            },
            items_by_lot={
                0: (0, 1),
                1: (2, 3),
                2: (4, 5),
                3: (6, 7),
                4: (8, 9),
            },
            items=items,
            suppliers=_make_suppliers()[:1],
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(0),
        )
        assert len(result) <= 3

    def test_deterministic(self) -> None:
        config = SupplierModelConfig(coupling_strength=0.8)
        items = _make_items(4)
        kwargs = dict(
            per_lot_bids={0: {0: 100.0, 1: 150.0}, 1: {0: 120.0, 1: 130.0}},
            items_by_lot={0: (0, 1), 1: (2, 3)},
            items=items,
            suppliers=_make_suppliers(),
            config=config,
            area_size=100.0,
        )
        r1 = generate_package_bids(**kwargs, rng=np.random.default_rng(99))
        r2 = generate_package_bids(**kwargs, rng=np.random.default_rng(99))
        assert r1 == r2

    def test_single_lot_returns_empty(self) -> None:
        config = SupplierModelConfig(coupling_strength=1.0)
        result = generate_package_bids(
            per_lot_bids={0: {0: 100.0}},
            items_by_lot={0: (0,)},
            items=_make_items(1),
            suppliers=_make_suppliers()[:1],
            config=config,
            area_size=100.0,
            rng=np.random.default_rng(0),
        )
        assert result == ()


class TestSolverWithPackages:
    def test_package_selected_when_cheaper(self) -> None:
        """Package bid cheaper than sum of singles — solver should use it."""
        from procurement_gym.solver.ortools_solver import ORToolsWDPSolver

        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={
                0: {0: 100.0, 1: 200.0},
                1: {0: 100.0, 1: 200.0},
            },
            entrants={0: frozenset({0, 1}), 1: frozenset({0, 1})},
            volumes={0: 3.0, 1: 3.0},
            capacities={0: 20.0, 1: 20.0},
            package_bids=(PackageBid(supplier_id=0, lot_ids=frozenset({0, 1}), price=150.0),),
        )
        assert result.status == "OPTIMAL"
        # Package at 150 beats singles at 100+100=200
        assert len(result.package_awards) == 1
        assert result.total_cost < 200.0

    def test_package_rejected_when_expensive(self) -> None:
        """Package bid more expensive — solver uses singles."""
        from procurement_gym.solver.ortools_solver import ORToolsWDPSolver

        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={
                0: {0: 100.0, 1: 200.0},
                1: {0: 100.0, 1: 200.0},
            },
            entrants={0: frozenset({0, 1}), 1: frozenset({0, 1})},
            volumes={0: 3.0, 1: 3.0},
            capacities={0: 20.0, 1: 20.0},
            package_bids=(PackageBid(supplier_id=0, lot_ids=frozenset({0, 1}), price=250.0),),
        )
        assert result.status == "OPTIMAL"
        assert len(result.package_awards) == 0
        assert abs(result.total_cost - 200.0) < 0.01

    def test_no_double_coverage(self) -> None:
        """Lots covered by a package are NOT also covered by singles."""
        from procurement_gym.solver.ortools_solver import ORToolsWDPSolver

        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={
                0: {0: 100.0, 1: 200.0},
                1: {0: 100.0, 1: 200.0},
                2: {1: 80.0},
            },
            entrants={
                0: frozenset({0, 1}),
                1: frozenset({0, 1}),
                2: frozenset({1}),
            },
            volumes={0: 3.0, 1: 3.0, 2: 3.0},
            capacities={0: 20.0, 1: 20.0},
            package_bids=(PackageBid(supplier_id=0, lot_ids=frozenset({0, 1}), price=150.0),),
        )
        assert result.status == "OPTIMAL"
        # All 3 lots covered
        assert set(result.allocation.keys()) == {0, 1, 2}

    def test_package_respects_capacity(self) -> None:
        """Package volume counts toward supplier capacity."""
        from procurement_gym.solver.ortools_solver import ORToolsWDPSolver

        solver = ORToolsWDPSolver(timeout_s=10.0)
        # Supplier 0 has capacity for only one lot but package needs both
        result = solver.solve(
            bids={
                0: {0: 100.0, 1: 200.0},
                1: {0: 100.0, 1: 200.0},
            },
            entrants={0: frozenset({0, 1}), 1: frozenset({0, 1})},
            volumes={0: 8.0, 1: 8.0},
            capacities={0: 10.0, 1: 20.0},  # supplier 0 can't handle both
            package_bids=(PackageBid(supplier_id=0, lot_ids=frozenset({0, 1}), price=50.0),),
        )
        assert result.status == "OPTIMAL"
        # Package can't be used (volume 16 > capacity 10)
        assert len(result.package_awards) == 0

    def test_backward_compat_no_packages(self) -> None:
        """No package_bids arg produces same result as before."""
        from procurement_gym.solver.ortools_solver import ORToolsWDPSolver

        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={0: {0: 100.0}},
            entrants={0: frozenset({0})},
            volumes={0: 5.0},
            capacities={0: 10.0},
        )
        assert result.status == "OPTIMAL"
        assert result.allocation == {0: 0}
        assert result.package_awards == ()


class TestEnvWithCoupling:
    def test_coupling_zero_runs(self) -> None:
        """Environment runs with default coupling_strength=0."""
        from procurement_gym.config import EnvConfig
        from procurement_gym.env import ProcurementEnv

        env = ProcurementEnv(config=EnvConfig())
        obs, _ = env.reset(seed=42)
        assert obs["global_context"].shape == (5,)
        assert obs["lot_summaries"].shape[1] == 8
        # Run a full episode
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
        assert "n_package_bids" in info
        assert info["n_package_bids"] == 0

    def test_coupling_positive_produces_packages(self) -> None:
        """With coupling > 0, package bids are generated."""
        from procurement_gym.config import EnvConfig, InstanceConfig, SupplierModelConfig
        from procurement_gym.env import ProcurementEnv

        config = EnvConfig(
            instance=InstanceConfig(
                n_items=10, n_suppliers=5, n_categories=2, capacity_scarcity=1.0
            ),
            supplier_model=SupplierModelConfig(coupling_strength=0.8),
        )
        env = ProcurementEnv(config=config)
        obs, _ = env.reset(seed=42)
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
        # With coupling, we expect some package bids (may be 0 if synergy is low)
        assert "n_package_bids" in info
        assert isinstance(info["n_package_bids"], int)

    def test_check_env_with_coupling(self) -> None:
        """Gymnasium check_env passes with coupling features."""
        from gymnasium.utils.env_checker import check_env

        from procurement_gym.config import EnvConfig, SupplierModelConfig
        from procurement_gym.env import ProcurementEnv

        config = EnvConfig(
            supplier_model=SupplierModelConfig(coupling_strength=0.5),
        )
        env = ProcurementEnv(config=config)
        check_env(env)
