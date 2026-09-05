"""Tests for OR-Tools CP-SAT WDP solver."""

import pytest

from procurement_gym.solver.ortools_solver import ORToolsWDPSolver, SolverResult


class TestSolverResult:
    def test_frozen(self) -> None:
        r = SolverResult(
            allocation={0: 0},
            total_cost=100.0,
            status="OPTIMAL",
            optimality_gap=0.0,
            solve_time_s=0.01,
        )
        with pytest.raises(AttributeError):
            r.total_cost = 0.0  # type: ignore[misc]


class TestORToolsWDPSolver:
    def test_trivial_single_lot_single_supplier(self) -> None:
        """One lot, one supplier — must award to that supplier."""
        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={0: {0: 100.0}},
            entrants={0: frozenset({0})},
            volumes={0: 5.0},
            capacities={0: 10.0},
        )
        assert result.status == "OPTIMAL"
        assert result.allocation == {0: 0}
        assert abs(result.total_cost - 100.0) < 0.01
        assert result.optimality_gap < 0.01

    def test_two_lots_two_suppliers(self) -> None:
        """Two lots, two suppliers — cheaper assignment wins."""
        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={
                0: {0: 100.0, 1: 150.0},  # lot 0: supplier 0 cheaper
                1: {0: 200.0, 1: 80.0},  # lot 1: supplier 1 cheaper
            },
            entrants={0: frozenset({0, 1}), 1: frozenset({0, 1})},
            volumes={0: 3.0, 1: 3.0},
            capacities={0: 10.0, 1: 10.0},
        )
        assert result.status == "OPTIMAL"
        assert result.allocation[0] == 0  # supplier 0 gets lot 0
        assert result.allocation[1] == 1  # supplier 1 gets lot 1
        assert abs(result.total_cost - 180.0) < 0.01

    def test_capacity_constraint_binding(self) -> None:
        """Supplier 0 is cheapest for both lots but can only handle one."""
        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={
                0: {0: 50.0, 1: 200.0},
                1: {0: 60.0, 1: 190.0},
            },
            entrants={0: frozenset({0, 1}), 1: frozenset({0, 1})},
            volumes={0: 6.0, 1: 6.0},
            capacities={0: 7.0, 1: 20.0},  # supplier 0 can only handle one lot
        )
        assert result.status == "OPTIMAL"
        # Supplier 0 gets lot 0 (cheapest), supplier 1 gets lot 1
        assert result.allocation[0] == 0
        assert result.allocation[1] == 1

    def test_no_entrants_returns_infeasible(self) -> None:
        """A lot with zero entrants makes the problem infeasible."""
        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={0: {0: 100.0}, 1: {}},
            entrants={0: frozenset({0}), 1: frozenset()},
            volumes={0: 5.0, 1: 5.0},
            capacities={0: 20.0},
        )
        assert result.status == "INFEASIBLE"

    def test_integer_scaling_precision(self) -> None:
        """Bids with decimals should be handled correctly via integer scaling."""
        solver = ORToolsWDPSolver(timeout_s=10.0)
        result = solver.solve(
            bids={0: {0: 99.999, 1: 100.001}},
            entrants={0: frozenset({0, 1})},
            volumes={0: 1.0},
            capacities={0: 10.0, 1: 10.0},
        )
        assert result.status == "OPTIMAL"
        assert result.allocation[0] == 0  # 99.999 < 100.001
        assert abs(result.total_cost - 99.999) < 0.01

    def test_deterministic(self) -> None:
        """Same inputs → same outputs."""
        solver = ORToolsWDPSolver(timeout_s=10.0)
        kwargs = dict(
            bids={0: {0: 100.0, 1: 110.0}, 1: {0: 120.0, 1: 90.0}},
            entrants={0: frozenset({0, 1}), 1: frozenset({0, 1})},
            volumes={0: 3.0, 1: 3.0},
            capacities={0: 10.0, 1: 10.0},
        )
        r1 = solver.solve(**kwargs)
        r2 = solver.solve(**kwargs)
        assert r1.allocation == r2.allocation
        assert r1.total_cost == r2.total_cost
