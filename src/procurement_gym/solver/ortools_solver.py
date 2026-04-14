# PLAN: WDP solver using OR-Tools CP-SAT. Minimizes total award cost subject to
#        lot coverage and supplier capacity constraints. Returns SolverResult with
#        allocation, cost, status, gap, and solve time. REQ-2.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 350-417
# ADAPTED FROM: google/or-tools CP-SAT assignment pattern (DeepWiki verified)
"""Winner Determination Problem solver using OR-Tools CP-SAT."""

from __future__ import annotations

import time
from dataclasses import dataclass

from ortools.sat.python import cp_model

SCALE = 1000


@dataclass(frozen=True)
class SolverResult:
    """Result of solving the WDP."""

    allocation: dict[int, int]  # lot_id -> supplier_id
    total_cost: float
    status: str  # OPTIMAL, FEASIBLE, INFEASIBLE, TIMEOUT
    optimality_gap: float
    solve_time_s: float
    package_awards: tuple[frozenset[int], ...] = ()  # lot_id sets of awarded packages


class ORToolsWDPSolver:
    """Solve WDP as set packing ILP via CP-SAT."""

    def __init__(self, timeout_s: float = 60.0) -> None:
        self._timeout_s = timeout_s

    def solve(
        self,
        bids: dict[int, dict[int, float]],
        entrants: dict[int, frozenset[int]],
        volumes: dict[int, float],
        capacities: dict[int, float],
        package_bids: tuple[object, ...] = (),
        volume_tiers: tuple[tuple[float, float], ...] = (),
    ) -> SolverResult:
        """Solve the WDP.

        Args:
            bids: {lot_id: {supplier_id: bid_amount}}
            entrants: {lot_id: frozenset of supplier_ids who entered}
            volumes: {lot_id: total_volume}
            capacities: {supplier_id: capacity}
            package_bids: tuple of PackageBid objects (cross-lot bundles)
            volume_tiers: ((threshold, discount_rate), ...) for volume discounts

        Returns:
            SolverResult with optimal or best-feasible allocation.
        """
        lots = sorted(bids.keys())
        all_suppliers = set(capacities.keys())

        # Check for lots with no entrants — infeasible
        for k in lots:
            if not entrants.get(k):
                return SolverResult(
                    allocation={},
                    total_cost=0.0,
                    status="INFEASIBLE",
                    optimality_gap=float("inf"),
                    solve_time_s=0.0,
                )

        model = cp_model.CpModel()

        # Decision variables: x[s, k] = 1 if supplier s awarded lot k (single)
        x: dict[tuple[int, int], cp_model.IntVar] = {}
        for k in lots:
            for s in entrants[k]:
                x[s, k] = model.new_bool_var(f"x_{s}_{k}")

        # Package bid variables: y[p] = 1 if package p is awarded
        y: dict[int, cp_model.IntVar] = {}
        # Filter packages: all lot_ids must be in the solver's lot set
        lot_set = set(lots)
        pkg_list = [
            pkg
            for pkg in package_bids
            if pkg.lot_ids <= lot_set  # type: ignore[union-attr]
        ]

        # Index packages by lot for coverage constraints
        packages_covering: dict[int, list[int]] = {k: [] for k in lots}

        for p_idx, pkg in enumerate(pkg_list):
            y[p_idx] = model.new_bool_var(f"y_{p_idx}")
            for k in pkg.lot_ids:  # type: ignore[union-attr]
                packages_covering[k].append(p_idx)

        # Constraint (1): each lot covered by exactly one source
        # (single-lot bid OR one package containing it)
        for k in lots:
            sources = [x[s, k] for s in entrants[k]]
            sources.extend(y[p_idx] for p_idx in packages_covering[k])
            model.add_exactly_one(sources)

        # Constraint (2): supplier capacity (integer-scaled)
        # Collect volume from both single and package awards
        for s in all_suppliers:
            terms: list[cp_model.LinearExpr] = []
            # Single-lot terms
            for k in lots:
                if (s, k) in x:
                    terms.append(int(volumes[k] * SCALE) * x[s, k])
            # Package terms
            for p_idx, pkg in enumerate(pkg_list):
                if pkg.supplier_id == s:  # type: ignore[union-attr]
                    pkg_vol = sum(volumes.get(k, 0.0) for k in pkg.lot_ids)  # type: ignore[union-attr]
                    terms.append(int(pkg_vol * SCALE) * y[p_idx])
            if terms:
                model.add(sum(terms) <= int(capacities[s] * SCALE))

        # Objective: minimize total cost (integer-scaled)
        obj_terms: list[cp_model.LinearExpr] = []
        # Single-lot bid costs
        for s, k in x:
            obj_terms.append(int(bids[k][s] * SCALE) * x[s, k])
        # Package bid costs
        for p_idx, pkg in enumerate(pkg_list):
            obj_terms.append(int(pkg.price * SCALE) * y[p_idx])  # type: ignore[union-attr]

        # Volume tier rebates (reduce cost when supplier volume exceeds thresholds)
        if volume_tiers and len(volume_tiers) > 1:
            obj_terms.extend(
                self._volume_tier_rebates(
                    model, x, y, pkg_list, volumes, bids, volume_tiers, lots, all_suppliers
                )
            )

        model.minimize(sum(obj_terms))

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self._timeout_s
        solver.parameters.random_seed = 0  # determinism

        t0 = time.monotonic()
        status_code = solver.solve(model)
        solve_time = time.monotonic() - t0

        if status_code in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            allocation: dict[int, int] = {}
            for s, k in x:
                if solver.value(x[s, k]) == 1:
                    allocation[k] = s

            # Extract package awards (lots covered by package must not already be allocated)
            awarded_pkgs: list[frozenset[int]] = []
            for p_idx, pkg in enumerate(pkg_list):
                if solver.value(y[p_idx]) == 1:
                    for k in pkg.lot_ids:  # type: ignore[union-attr]
                        assert k not in allocation, (
                            f"Lot {k} double-covered: single bid and package"
                        )
                        allocation[k] = pkg.supplier_id  # type: ignore[union-attr]
                    awarded_pkgs.append(pkg.lot_ids)  # type: ignore[union-attr]

            total_cost = solver.objective_value / SCALE
            best_bound = solver.best_objective_bound / SCALE

            if best_bound > 0:
                gap = abs(total_cost - best_bound) / best_bound
            else:
                gap = 0.0

            status_str = "OPTIMAL" if status_code == cp_model.OPTIMAL else "FEASIBLE"
            return SolverResult(
                allocation=allocation,
                total_cost=total_cost,
                status=status_str,
                optimality_gap=gap,
                solve_time_s=solve_time,
                package_awards=tuple(awarded_pkgs),
            )

        return SolverResult(
            allocation={},
            total_cost=0.0,
            status="INFEASIBLE",
            optimality_gap=float("inf"),
            solve_time_s=solve_time,
        )

    def _volume_tier_rebates(
        self,
        model: cp_model.CpModel,
        x: dict[tuple[int, int], cp_model.IntVar],
        y: dict[int, cp_model.IntVar],
        pkg_list: list[object],
        volumes: dict[int, float],
        bids: dict[int, dict[int, float]],
        volume_tiers: tuple[tuple[float, float], ...],
        lots: list[int],
        all_suppliers: set[int],
    ) -> list[cp_model.LinearExpr]:
        """Add volume tier rebate variables to the model.

        Returns negative objective terms (rebates) that reduce cost when a
        supplier's total awarded volume exceeds tier thresholds.
        """
        rebate_terms: list[cp_model.LinearExpr] = []
        # Sort tiers by threshold
        sorted_tiers = sorted(volume_tiers, key=lambda t: t[0])

        for s in all_suppliers:
            # Compute total volume expression for this supplier
            vol_terms: list[cp_model.LinearExpr] = []
            bid_terms: list[cp_model.LinearExpr] = []
            for k in lots:
                if (s, k) in x:
                    vol_terms.append(int(volumes[k] * SCALE) * x[s, k])
                    bid_terms.append(int(bids[k][s] * SCALE) * x[s, k])
            for p_idx, pkg in enumerate(pkg_list):
                if pkg.supplier_id == s:  # type: ignore[union-attr]
                    pkg_vol = sum(volumes.get(k, 0.0) for k in pkg.lot_ids)  # type: ignore[union-attr]
                    vol_terms.append(int(pkg_vol * SCALE) * y[p_idx])
                    bid_terms.append(int(pkg.price * SCALE) * y[p_idx])  # type: ignore[union-attr]

            if not vol_terms:
                continue

            total_vol = sum(vol_terms)

            # Upper bound on this supplier's total bid (for rebate scaling)
            max_supplier_bid = 0
            for k in lots:
                if (s, k) in x:
                    max_supplier_bid += int(bids[k][s] * SCALE)
            for p_idx, pkg in enumerate(pkg_list):
                if pkg.supplier_id == s:  # type: ignore[union-attr]
                    max_supplier_bid += int(pkg.price * SCALE)  # type: ignore[union-attr]

            if max_supplier_bid == 0:
                continue

            for i in range(1, len(sorted_tiers)):
                threshold, discount = sorted_tiers[i]
                prev_discount = sorted_tiers[i - 1][1]
                incremental = discount - prev_discount
                if incremental <= 0:
                    continue

                above_t = model.new_bool_var(f"tier_{s}_{i}")
                scaled_threshold = int(threshold * SCALE)

                # above_t = 1 iff total_vol >= threshold
                model.add(total_vol >= scaled_threshold).only_enforce_if(above_t)
                model.add(total_vol <= scaled_threshold - 1).only_enforce_if(above_t.negated())

                # Rebate: incremental discount * supplier's max possible bid
                # This is an upper-bound approximation that stays linear
                rebate_scaled = int(incremental * max_supplier_bid)
                rebate_terms.append(-rebate_scaled * above_t)

        return rebate_terms
