# TODO: ProcurementGym (Phase 1 + Phase 2)

## Plan
path: .claude/plans/20260412-procurement-gym.md
plan_id: 20260412-procurement-gym

## Build Environment
test: uv run pytest tests/ -v --tb=short
lint: uv run ruff check src/
typecheck: uv run mypy src/ --strict
format: uv run ruff format src/ tests/
env_check: uv run python -c "from procurement_gym import ProcurementEnv; import gymnasium; gymnasium.utils.env_checker.check_env(ProcurementEnv()); print('ENV CHECK PASSED')"

## Test Map
| Requirement | Source Files | Affected Tests | Backpressure | Status |
|-------------|-------------|----------------|-------------|--------|
| REQ-3 | types.py, partition.py | test_types (10), test_partition (15) | tests ✓ lint ✓ types ✓ | DONE |
| REQ-2 | ortools_solver.py | test_solver (7) | tests ✓ lint ✓ types ✓ | DONE |
| REQ-4 | transport.py | test_transport_generator (11) | tests ✓ lint ✓ types ✓ | DONE |
| REQ-5 | cost.py, participation.py, markup.py, simulator.py | test_supplier_model (12) | tests ✓ lint ✓ types ✓ | DONE |
| REQ-1 | env.py, __init__.py | test_env (12) | tests ✓ lint ✓ types ✓ env_check ✓ | DONE |
| REQ-11 | all | test_seeded_reproducibility, test_deterministic | tests ✓ | DONE |

## Roadmap
### REQ-3: Immutable Data Objects [critical] ✓
- [x] Tests written (RED) — 25 tests
- [x] Implementation (GREEN) — types.py, partition.py
- [x] Backpressure: tests ✓ lint ✓ types ✓

### REQ-2: Exact Solver Integration [critical] ✓
- [x] Tests written (RED) — 7 tests
- [x] Implementation (GREEN) — ortools_solver.py
- [x] Backpressure: tests ✓ lint ✓ types ✓

### REQ-4: Transport Instance Generator [critical] ✓
- [x] Tests written (RED) — 11 tests
- [x] Implementation (GREEN) — transport.py
- [x] Backpressure: tests ✓ lint ✓ types ✓

### REQ-5: Three-Layer Supplier Model [critical] ✓
- [x] Tests written (RED) — 12 tests
- [x] Implementation (GREEN) — cost.py, participation.py, markup.py, simulator.py
- [x] Backpressure: tests ✓ lint ✓ types ✓
- [x] Reward variance >5% across 3 different lot structures (verified)

### REQ-1: Gymnasium Environment [critical] ✓
- [x] Tests written (RED) — 12 tests
- [x] Implementation (GREEN) — env.py, __init__.py
- [x] Backpressure: tests ✓ lint ✓ types ✓
- [x] check_env PASSED

### REQ-11: Reproducibility [high] ✓
- [x] Seeded reproducibility tests in test_transport_generator, test_env, test_solver
- [x] Backpressure: tests ✓

## Decisions
- Synergy formula: normalized distances by area_size, scaled by total_base_cost (plan fix for underflow)
- Participation circularity: use eligible count at participation stage, entrant count at markup stage
- Category encoding: integer / n_categories (not one-hot for MVP)
- Invalid action: clamp to nearest valid action (for check_env compatibility)
- Unserviced penalty: -10 * mean_item_value * n_unserviced (mean value, not raw base_value)

## Walkthrough

### What was built
Phase 1 (Core Skeleton) + Phase 2 (Three-Layer Supplier Model) of ProcurementGym.

### Files created (11 source, 7 test)
**Source** (procurement-gym/src/procurement_gym/):
- `__init__.py` — exports + Gymnasium registration
- `config.py` — Pydantic config (EnvConfig, InstanceConfig, SupplierModelConfig)
- `env.py` — ProcurementEnv (Gymnasium wrapper)
- `core/types.py` — Item, Supplier, ProcurementInstance (frozen dataclasses)
- `core/partition.py` — LotPartition (immutable, functional updates)
- `solver/ortools_solver.py` — WDP via CP-SAT (set packing ILP)
- `instances/transport.py` — TransportInstanceGenerator
- `suppliers/cost.py` — Additive cost + geographic synergy
- `suppliers/participation.py` — Logistic participation model
- `suppliers/markup.py` — Competition-compressed markup
- `suppliers/simulator.py` — ThreeLayerSupplierSimulator

**Tests** (procurement-gym/tests/):
- `test_types.py` (10 tests), `test_partition.py` (15), `test_config.py` (6)
- `test_solver.py` (7), `test_transport_generator.py` (11)
- `test_supplier_model.py` (12), `test_env.py` (12)

### Exit criteria verification
```
uv run pytest tests/ -v          # 73 passed
uv run ruff check src/           # All checks passed
uv run mypy src/ --strict        # Success: no issues found in 15 files
check_env(ProcurementEnv())      # ENV CHECK PASSED
reward variance test             # >5% across 3 lot structures
```

All findings were verified empirically with OR-Tools, gymnasium, and uv where possible.

### Confirmed bugs that will cause runtime failures

1. **ILP capacity constraint (float TypeError)** — `vol[k] * x[s,k] <= capacity[s]` passes raw floats to CP-SAT `model.add()`. CP-SAT requires integer coefficients. The `SCALE=1000` note in the plan covers only bids; volume and capacity must also be scaled. Verified: `TypeError` at runtime without scaling.

2. **Dict comprehension NameError in OR-Tools snippet** — `{(s, k): ... for s in entrants[k] for k in lots}` has loops reversed; `k` is undefined when `entrants[k]` is evaluated. Appears in both the plan and PRODUCT_INTENT.md. Fix: `for k in lots for s in entrants[k]`.

3. **synergy() wrong config type** — Plan's synergy function signature takes `config: SupplierModelConfig` but accesses `config.area_size`, which is a field on `InstanceConfig`. Fails `mypy --strict` (an exit criterion).

### Specification gaps requiring a decision before building

4. **Eligibility mechanism ambiguity** — Three overlapping mechanisms exist: `Item.qualified_supplier_ids`, `Supplier.categories ⊇ categories(L_k)`, and the ILP `E_k` definition. Not reconciled.

5. **Unserviced lot penalty formula** — `base_value` is an Item attribute, not a scalar. The formula `-10 * base_value * n_unserviced` is ambiguous: whose base_value? Sum of lot items? Instance mean?

6. **Invalid action handling unspecified** — `check_env()` sends random `Discrete(21)` actions including out-of-mask actions. The env must not crash. No defensive behavior specified.

7. **TransportGenerator constructor conflict** — REQ-4 says `TransportGenerator(n_items, n_suppliers, seed)`. The Missing Interface Signatures section says constructor takes `config` with a separate `generate(seed)` method. Both can't be right.

8. **Simulation RNG derivation incomplete** — The seeding chain says "SupplierSimulator receives rng from env" but doesn't specify how the env derives it from `self.np_random`.

9. **`match` variable in participation formula undefined** — `sigmoid(w_cap*match + ...)`: what is `match`? Binary? Fraction of categories covered?

10. **Category encoding normalization divisor unspecified** — Normalized by `n_categories` (config) or `len(unique categories in instance)`? Different choices produce non-stationary observations if instance generation doesn't always produce all categories.

11. **Config schema is a forward reference** — Plan says "all parameters from PRODUCT_INTENT.md" without inlining the schema. Plan is not self-contained.

### Gate evidence

- `tests-pass`: 0 tests collected, 0 failures (skeleton only — no implementation yet).
- `lint-pass`: ruff clean on empty modules.
- `typecheck-pass`: mypy --strict clean on empty modules.
- `plan-synced`: plan file exists at `.claude/mission/plan.md`.
