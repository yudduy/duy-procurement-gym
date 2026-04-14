# PLAN: Gymnasium env wrapper. Sequential assignment: at each step, assign one item
#        to an existing or new lot. Terminal reward = buyer welfare from exact WDP solver.
# SOURCE: docs/research/procurement-rl/PRODUCT_INTENT.md lines 538-555 (obs/action space)
# ADAPTED FROM: ai4co/rl4co constructive env pattern (step/reset lifecycle)
"""ProcurementEnv — Gymnasium environment for lot structure optimization."""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from procurement_gym.config import EnvConfig
from procurement_gym.core.partition import LotPartition
from procurement_gym.core.types import ProcurementInstance
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.solver.ortools_solver import ORToolsWDPSolver
from procurement_gym.suppliers.simulator import ThreeLayerSupplierSimulator

K_MAX = 20
D_ITEM = 6
D_LOT = 8  # n_items, mean_x, mean_y, total_volume, n_qualified_mean, n_categories, pkg_eligible, supplier_overlap
D_GLOBAL = 5  # items_remaining, n_lots, total_items, coupling_strength, n_package_bids


class ProcurementEnv(gymnasium.Env[dict[str, np.ndarray[Any, Any]], int]):
    """RL environment for procurement lot structure optimization.

    The agent assigns N items to lots one at a time. After all items are assigned,
    the environment simulates supplier behavior and solves the WDP to compute
    buyer welfare as the reward.
    """

    metadata: dict[str, Any] = {"render_modes": ["ansi"]}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self._config = config or EnvConfig()
        self.render_mode = render_mode

        self.observation_space = spaces.Dict(
            {
                "current_item": spaces.Box(-np.inf, np.inf, shape=(D_ITEM,), dtype=np.float64),
                "lot_summaries": spaces.Box(
                    -np.inf, np.inf, shape=(K_MAX, D_LOT), dtype=np.float64
                ),
                "global_context": spaces.Box(-np.inf, np.inf, shape=(D_GLOBAL,), dtype=np.float64),
                "action_mask": spaces.MultiBinary(K_MAX + 1),
            }
        )
        self.action_space = spaces.Discrete(K_MAX + 1)

        self._generator = TransportInstanceGenerator(
            instance_config=self._config.instance,
            supplier_config=self._config.supplier_model,
        )
        self._simulator = ThreeLayerSupplierSimulator(
            config=self._config.supplier_model,
            area_size=self._config.instance.area_size,
        )
        self._solver = ORToolsWDPSolver(timeout_s=self._config.solver_timeout_s)

        # State (set during reset)
        self._instance: ProcurementInstance | None = None
        self._partition: LotPartition | None = None
        self._step_idx: int = 0
        self._item_order: list[int] = []
        self._category_map: dict[str, int] = {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
        super().reset(seed=seed)

        # Derive instance seed from env's RNG
        assert self.np_random is not None
        instance_seed = int(self.np_random.integers(0, 2**31))
        self._instance = self._generator.generate(seed=instance_seed)

        n = len(self._instance.items)
        self._partition = LotPartition.empty(n)
        self._step_idx = 0

        # Build category map
        cats = sorted({item.category for item in self._instance.items})
        self._category_map = {c: i for i, c in enumerate(cats)}

        # Item presentation order from config
        instance = self._instance
        if self._config.item_order == "random":
            perm = self.np_random.permutation(n)
            self._item_order = [int(x) for x in perm]
        elif self._config.item_order == "by_category":
            self._item_order = sorted(range(n), key=lambda i: instance.items[i].category)
        else:  # ascending_qualified (default)
            self._item_order = sorted(
                range(n),
                key=lambda i: len(instance.items[i].qualified_supplier_ids),
            )

        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray[Any, Any]], float, bool, bool, dict[str, Any]]:
        assert self._instance is not None
        assert self._partition is not None

        n = len(self._instance.items)
        item_idx = self._item_order[self._step_idx]

        # Clamp invalid actions to nearest valid action
        n_current_lots = self._partition.n_lots
        max_valid = min(n_current_lots, K_MAX - 1)  # can open new lot up to K_MAX-1
        action = min(action, max_valid)

        # Assign current item
        if action < n_current_lots:
            # Assign to existing lot — find the lot_id by index
            lot_ids_sorted = sorted(self._partition.lot_ids)
            lot_id = lot_ids_sorted[action]
        else:
            # Open new lot
            lot_id = self._partition.next_lot_id()

        self._partition = self._partition.assign(item_idx, lot_id)
        self._step_idx += 1

        terminated = self._step_idx >= n
        truncated = False

        reward = 0.0
        info: dict[str, Any] = {}

        if terminated:
            reward, info = self._compute_terminal_reward()

        obs = self._get_obs() if not terminated else self._get_terminal_obs()

        return obs, reward, terminated, truncated, info

    def _compute_terminal_reward(self) -> tuple[float, dict[str, Any]]:
        assert self._instance is not None
        assert self._partition is not None
        assert self.np_random is not None

        # Derive simulator RNG from env's RNG
        sim_seed = int(self.np_random.integers(0, 2**31))
        sim_rng = np.random.default_rng(sim_seed)

        sim_result = self._simulator.simulate(self._partition, self._instance, sim_rng)

        # Build solver inputs
        bids: dict[int, dict[int, float]] = {}
        entrants: dict[int, frozenset[int]] = {}
        volumes: dict[int, float] = {}
        capacities = {s.id: s.capacity for s in self._instance.suppliers}

        n_unserviced = 0
        for lot_id in sorted(self._partition.lot_ids):
            lot_bids = sim_result.bids.get(lot_id, {})
            lot_entrants = sim_result.entrants.get(lot_id, frozenset())

            if not lot_entrants:
                n_unserviced += 1
                continue

            bids[lot_id] = lot_bids
            entrants[lot_id] = lot_entrants
            item_idxs = self._partition.lot_items(lot_id)
            volumes[lot_id] = sum(self._instance.items[i].volume for i in item_idxs)

        # Compute value only for SERVICED items (not unserviced lots)
        serviced_item_idxs: set[int] = set()
        for lot_id in bids:
            serviced_item_idxs.update(self._partition.lot_items(lot_id))
        serviced_value = sum(self._instance.items[i].base_value for i in serviced_item_idxs)
        mean_item_value = sum(item.base_value for item in self._instance.items) / len(
            self._instance.items
        )

        if not bids:
            # All lots unserviced — large penalty
            penalty = -10.0 * mean_item_value * n_unserviced
            return float(penalty), {"status": "ALL_UNSERVICED", "n_unserviced": n_unserviced}

        # Build volume tiers from config
        cfg = self._config.supplier_model
        vol_tiers: tuple[tuple[float, float], ...] = ()
        if cfg.coupling_strength > 0:
            vol_tiers = tuple(
                zip(cfg.volume_tier_thresholds, cfg.volume_tier_discounts, strict=False)
            )

        solver_result = self._solver.solve(
            bids=bids,
            entrants=entrants,
            volumes=volumes,
            capacities=capacities,
            package_bids=sim_result.package_bids,
            volume_tiers=vol_tiers,
        )

        if solver_result.status == "INFEASIBLE":
            # Only penalize lots that were sent to solver (had entrants)
            penalty = -10.0 * mean_item_value * (len(bids) + n_unserviced)
            return float(penalty), {"status": "INFEASIBLE", "n_unserviced": n_unserviced}

        # Reward = value of serviced items - cost paid - penalty for unserviced
        reward = serviced_value - solver_result.total_cost
        if n_unserviced > 0:
            reward -= 10.0 * mean_item_value * n_unserviced

        # Lot overhead: penalizes lot proliferation (gated by coupling)
        lot_overhead = cfg.lot_overhead_base * cfg.coupling_strength
        lot_overhead_total = 0.0
        if lot_overhead > 0:
            lot_overhead_total = lot_overhead * self._partition.n_lots
            reward -= lot_overhead_total

        info = {
            "status": solver_result.status,
            "total_cost": solver_result.total_cost,
            "n_lots": self._partition.n_lots,
            "n_unserviced": n_unserviced,
            "lot_overhead_total": lot_overhead_total,
            "optimality_gap": solver_result.optimality_gap,
            "solve_time_s": solver_result.solve_time_s,
            "n_package_bids": len(sim_result.package_bids),
            "n_package_awards": len(solver_result.package_awards),
        }
        return float(reward), info

    def _get_obs(self) -> dict[str, np.ndarray[Any, Any]]:
        assert self._instance is not None
        assert self._partition is not None

        n = len(self._instance.items)
        n_cats = max(1, len(self._category_map))

        # Current item features
        if self._step_idx < n:
            item_idx = self._item_order[self._step_idx]
            item = self._instance.items[item_idx]
            current_item = np.array(
                [
                    self._category_map.get(item.category, 0) / n_cats,
                    item.base_value,
                    item.x,
                    item.y,
                    item.volume,
                    float(len(item.qualified_supplier_ids)),
                ],
                dtype=np.float64,
            )
        else:
            current_item = np.zeros(D_ITEM, dtype=np.float64)

        # Lot summaries
        lot_summaries = np.zeros((K_MAX, D_LOT), dtype=np.float64)
        lot_ids_sorted = sorted(self._partition.lot_ids)
        for idx, lot_id in enumerate(lot_ids_sorted):
            if idx >= K_MAX:
                break
            item_idxs = self._partition.lot_items(lot_id)
            if not item_idxs:
                continue
            items = [self._instance.items[i] for i in item_idxs]
            xs = [it.x for it in items]
            ys = [it.y for it in items]
            vols = [it.volume for it in items]
            n_quals = [float(len(it.qualified_supplier_ids)) for it in items]
            n_cats_in_lot = float(len({it.category for it in items}))
            # Supplier overlap: fraction of qualified suppliers shared with other lots
            qualified_sets = [it.qualified_supplier_ids for it in items]
            lot_qualified = frozenset().union(*qualified_sets) if qualified_sets else frozenset()
            overlap = 0.0
            if lot_ids_sorted and len(lot_ids_sorted) > 1:
                other_count = 0
                for other_idx, other_lid in enumerate(lot_ids_sorted):
                    if other_idx == idx or other_idx >= K_MAX:
                        continue
                    other_items_idxs = self._partition.lot_items(other_lid)
                    other_qual = (
                        frozenset().union(
                            *(
                                self._instance.items[i].qualified_supplier_ids
                                for i in other_items_idxs
                            )
                        )
                        if other_items_idxs
                        else frozenset()
                    )
                    if lot_qualified or other_qual:
                        union_size = len(lot_qualified | other_qual)
                        if union_size > 0:
                            overlap += len(lot_qualified & other_qual) / union_size
                    other_count += 1
                if other_count > 0:
                    overlap /= other_count

            lot_summaries[idx] = [
                float(len(items)),
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                sum(vols),
                sum(n_quals) / len(n_quals),
                n_cats_in_lot,
                float(len(lot_qualified)),  # pkg_eligible proxy
                overlap,  # supplier overlap with other lots
            ]

        # Global context
        coupling = self._config.supplier_model.coupling_strength
        global_context = np.array(
            [
                float(n - self._step_idx),
                float(self._partition.n_lots),
                float(n),
                float(coupling),
                0.0,  # n_package_bids placeholder (unknown until terminal)
            ],
            dtype=np.float64,
        )

        # Action mask
        action_mask = np.zeros(K_MAX + 1, dtype=np.int8)
        n_current = self._partition.n_lots
        # Existing lots
        for i in range(min(n_current, K_MAX)):
            action_mask[i] = 1
        # New lot (if room)
        if n_current < K_MAX:
            action_mask[n_current] = 1

        return {
            "current_item": current_item,
            "lot_summaries": lot_summaries,
            "global_context": global_context,
            "action_mask": action_mask,
        }

    def _get_terminal_obs(self) -> dict[str, np.ndarray[Any, Any]]:
        """Return a valid observation for the terminal state."""
        return {
            "current_item": np.zeros(D_ITEM, dtype=np.float64),
            "lot_summaries": np.zeros((K_MAX, D_LOT), dtype=np.float64),
            "global_context": np.zeros(D_GLOBAL, dtype=np.float64),
            "action_mask": np.zeros(K_MAX + 1, dtype=np.int8),
        }

    def render(self) -> str | None:  # type: ignore[override]
        if self.render_mode != "ansi":
            return None
        assert self._partition is not None
        assert self._instance is not None
        n = len(self._instance.items)
        lines = [
            f"Step {self._step_idx}/{n}",
            f"Lots: {self._partition.n_lots}",
        ]
        for lot_id in sorted(self._partition.lot_ids):
            items = self._partition.lot_items(lot_id)
            lines.append(f"  Lot {lot_id}: {len(items)} items {items}")
        unassigned = sum(1 for a in self._partition.assignments if a == -1)
        lines.append(f"Unassigned: {unassigned}")
        return "\n".join(lines)
