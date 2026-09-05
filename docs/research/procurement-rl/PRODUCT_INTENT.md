---
plan_id: 20260412-procurement-gym
created: 2026-04-12T22:00:00-07:00
status: draft
---

# Product Intent: ProcurementGym

## The Job

**Who**: Researchers building the first open RL environment for procurement lot structure optimization.

**Before**: The procurement AI field has no public verifiable environment. The most impactful buyer decision (lot structuring) is formally unsolved for N>2 items and has zero ML/RL applied. Industry (Keelvar, Coupa) uses manual/heuristic lot design. Academia studies winner determination (already solved by CPLEX) and mechanism design (seller-side only). The gap between industry practice and academic research is enormous.

**After**: ProcurementGym exists as an open-source Gymnasium-compatible environment where an RL agent learns to partition procurement items into lots. The environment uses an exact solver (OR-Tools/CPLEX) as the inner oracle, producing verifiable rewards. A trained agent demonstrates 2-8% cost reduction over standard heuristics on transportation procurement. Results are published at EC (Economics and Computation).

**The hiring job**: "Help me discover whether RL can learn non-obvious lot structuring principles that outperform human heuristics, and package this as a reproducible benchmark the field can build on."

## The Announcement

ProcurementGym: the first RL environment where an AI agent learns to design procurement auctions, not just solve them. While CPLEX solves the allocation problem in seconds, nobody has automated the upstream question — how to structure what gets auctioned. Our agent discovers lot structures that reduce buyer costs by 2-8% over expert heuristics, validated on real EU procurement data. Open-source, Gymnasium-compatible, built for reproducibility.

## The Appetite

**Complexity budget**: 8 weeks of engineering across 5 phases. Each phase has a hard exit criterion. No scope creep beyond what's needed for an EC paper.

**Solution shape**: A Python package (`procurement_gym`) installable via pip, with a Gymnasium env, configurable supplier simulation, OR-Tools solver integration, baselines, and evaluation harness. Plus a training script and paper-ready experiment configs.

## Out of Scope

| Exclusion | Why |
|-----------|-----|
| Commercial product / SaaS | Deferred — solve the research problem first |
| Multi-round dynamic auctions | The lot partition problem alone is novel enough for one paper |
| LLM-based agents | PPO/SAC RL agents only. LLM agents are a follow-up |
| Real-time deployment | Research environment, not production system |
| Scoring formula optimization | Lot structure only. Scoring formula is a separate L1 variable |
| Multi-attribute quality dimensions | Price-only for MVP. Quality adds dimensions without changing the core contribution |
| Neural WDP solver | CPLEX/OR-Tools handles L2. We're solving L1. |

## Principles

- **Always verifiability over speed.** Every reward must be traceable to an exact solver solution. No approximate rewards.
- **Always immutability.** All data objects (Item, Supplier, LotPartition, ProcurementInstance) are frozen. No mutation. New objects for every state change.
- **Always calibratability over realism.** The supplier model must be fittable from real data, even if simplified. A wrong-but-calibratable model beats a "realistic"-but-opaque one.
- **Never reward shaping.** The agent discovers structure from raw buyer welfare. Shaping encodes human assumptions about what matters.
- **When in doubt, choose the simpler formulation** and verify it works before adding complexity.

## Decision Precedence

When /execute faces an ambiguous choice, apply in this order:
1. **Invariants**: CPLEX/OR-Tools must verify every allocation. Frozen dataclasses. Gymnasium API compliance.
2. **Guardrails**: No mutation of shared state. No hardcoded parameters (everything configurable). No `print()` — use `logging`.
3. **Explicit acceptance criteria** (REQ-N WHEN/SHALL statements below)
4. **Principles** (above)
5. **Source Map** reference implementations
6. **Simpler / more reversible option**

## Boundaries

- **Success**: RL agent trained in ProcurementGym beats geographic clustering baseline by >2% on buyer welfare, across 3+ random seeds, on N>=50 item instances. Toy case (N=2) recovers known theoretical optimum. GPV robustness ablation shows <1% degradation at ±20% parameter shift.
- **Invariants**: (I1) All allocations verified by exact solver with optimality certificate. (I2) All data objects immutable (frozen dataclass). (I3) Environment passes `gymnasium.utils.env_checker.check_env()`. (I4) Every experiment reproducible with seed.
- **Guardrails**: (G1) Solver timeout 60s, returns incumbent + suboptimality flag. (G2) Max K_max=20 lots. (G3) Max N=500 items. (G4) All randomness via `np.random.Generator` with explicit seed.
- **Stop rules**: (S1) If OR-Tools cannot solve instances at N>200 within timeout, switch to CPLEX (academic license). (S2) If RL agent doesn't beat random after 1M steps, diagnose reward signal before adding shaping. (S3) If GPV robustness ablation shows >5% degradation, the circularity problem is real — escalate to rethink the supplier model.

---

## Domain Knowledge (CRITICAL — agents must internalize this)

### What is Procurement?

A **buyer** (e.g., FedEx, Siemens, a government agency) needs to purchase goods/services from **suppliers**. The buyer runs a **sourcing event** (auction/RFP) where suppliers submit **bids**. The buyer then **awards** contracts to the best bidders.

### The Three-Layer Decomposition

**Layer 1 — Problem Formulation (LOT DESIGN)**: The buyer decides HOW to structure the auction. The most impactful decision is the **lot structure** — which items get bundled together into lots that suppliers bid on as a package. Example: a logistics buyer sourcing 100 shipping lanes might bundle them into 10 lots of 10 lanes each, or 50 lots of 2, or keep all 100 as individual lots.

**Why lot structure matters**: It determines:
- **Competition**: Larger lots → fewer suppliers can bid (need capability for all items) → less competition → higher prices. Smaller lots → more suppliers per lot → more competition → lower prices. BUT too-small lots → suppliers can't offer volume discounts → higher per-unit prices.
- **Synergies**: Items with shared cost drivers (e.g., nearby shipping lanes) are cheaper to serve together. Bundling them captures synergies.
- **The tradeoff**: competition breadth vs. volume efficiency. This is the core tension the RL agent must learn to navigate.

**Layer 2 — Winner Determination (WDP)**: Given bids on lots, find the cost-minimizing allocation. This is a weighted set packing problem (NP-hard in general). CPLEX/OR-Tools solve it exactly in seconds for typical procurement instances. **This layer is already solved — we don't touch it.**

**Layer 3 — Strategic Interaction**: Suppliers choose bid amounts strategically. They know their own costs but not competitors'. More competition → smaller markups. This is modeled via the supplier simulation.

### Key Economics Concepts

**GPV Estimation** (Guerre, Perrigne, Vuong 2000): In a first-price sealed-bid auction with N bidders, each bidder's cost c maps to their bid b via:
```
b = c * (1 + markup)
markup = alpha * (1/N_competitors)^beta
```
Given observed bids, you can INVERT this to estimate the cost distribution. This is the bridge between observed auction data and the cost model in our simulator.

**Bell Number**: The number of ways to partition a set of N items into non-empty, non-overlapping subsets. Bell(10) = 115,975. Bell(50) ~ 10^47. This is the size of the lot design action space. Too large for enumeration, structured enough for RL.

**Weighted Set Packing**: Given a universe of items and a collection of subsets (bids) with weights (prices), find the maximum-weight collection of non-overlapping subsets. This is the WDP formulation. Equivalent to maximum weight independent set on the bid conflict graph.

---

## Mathematical Formulation

### The Lot Partition Problem

**Given**:
- Items I = {1, ..., N} with features x_i (category, volume, location, qualified suppliers)
- Suppliers S = {1, ..., M} with capabilities cap_s ⊆ Categories, costs c_s(i) per item, capacity C_s
- Synergy function syn(L, s) capturing cost economies when items in lot L are served together

**Find**: A partition π = {L_1, ..., L_K} of I into K lots (K is a decision variable) that maximizes buyer welfare.

**The pipeline** (for evaluating any partition π):

```
1. For each lot L_k in π:
   a. Determine eligible suppliers: E_k = {s : cap_s ⊇ categories(L_k) AND capacity_s ≥ volume(L_k)}
   b. For each eligible supplier s ∈ E_k:
      - Compute cost: c_s(L_k) = Σ_{i∈L_k} c_s(i) + syn(L_k, s) + ε_s
      - Compute participation: p_s(L_k) = sigmoid(w_cap*match + w_size*log|L_k| + w_comp*|E_k| + b_s)
      - Sample participation: enters_s ~ Bernoulli(p_s(L_k))
      - If enters: compute bid: bid_s(L_k) = c_s(L_k) * (1 + α_s / |entrants_k|^β_s)
      
2. Solve WDP via ILP:
   minimize  Σ_k Σ_s bid_s(L_k) * x_{sk}
   subject to:
     Σ_s x_{sk} = 1          ∀k (each lot awarded to exactly one supplier)
     Σ_k volume(L_k)*x_{sk} ≤ C_s  ∀s (capacity constraints)
     x_{sk} ∈ {0,1}
     x_{sk} = 0 if s ∉ entrants_k
     
3. Reward = Σ_i value(i) - Σ_k Σ_s bid_s(L_k) * x*_{sk}
   where x* is the optimal ILP solution
```

### The RL Formulation

- **State** at step t: features of item t, plus summary of current partial partition
- **Action** at step t: assign item t to lot k ∈ {1, ..., K_current} or open new lot K_current+1
- **Transition**: partition updated with item t assigned
- **Reward**: 0 at steps 1..N-1, buyer_welfare at step N (sparse, terminal)
- **Episode**: N steps, one per item

---

## Requirements (ordered by priority)

| ID | Name | Acceptance Criteria | Priority |
|----|------|-------------------|----------|
| REQ-1 | Gymnasium Environment | WHEN `gymnasium.utils.env_checker.check_env(ProcurementEnv())` is called, system SHALL pass all checks including reset/step/render/close lifecycle | critical |
| REQ-2 | Exact Solver Integration | WHEN WDP is formulated for any valid lot partition, system SHALL return provably optimal allocation via OR-Tools CP-SAT with optimality gap ≤ 0.01% | critical |
| REQ-3 | Immutable Data Objects | WHEN any Item, Supplier, LotPartition, or ProcurementInstance is created, system SHALL be a frozen dataclass with no mutable fields | critical |
| REQ-4 | Transport Instance Generator | WHEN `TransportGenerator(n_items=N, n_suppliers=M, seed=S)` is called, system SHALL produce a valid ProcurementInstance with geographic coordinates, distance-based costs, and carrier capability sets | critical |
| REQ-5 | Three-Layer Supplier Model | WHEN a lot partition is given to SupplierSimulator, system SHALL compute costs (additive + synergy), participation (logistic), and bids (markup compressed by competition) independently for each supplier-lot pair | critical |
| REQ-6 | Baseline Implementations | WHEN baselines are run on same instances, system SHALL implement: random partition, single-item (each item own lot), geographic clustering (k-means on coordinates), equal-size, and exhaustive (N≤15 only) | high |
| REQ-7 | PPO Training Pipeline | WHEN `python train.py --env ProcurementEnv --algo ppo --seed S` is run, system SHALL train a PPO agent via stable-baselines3 with tensorboard logging and checkpoint saving | high |
| REQ-8 | Evaluation Harness | WHEN `python evaluate.py --policy P --instances I` is run, system SHALL report: mean buyer welfare, std, improvement over each baseline, competition per lot, and optimality gap vs exhaustive (small N) | high |
| REQ-9 | Toy Validation | WHEN N=2 items with symmetric suppliers, trained agent SHALL recover the known theoretical optimum (bundle if synergy > competition loss, split otherwise) with >95% accuracy | high |
| REQ-10 | GPV Robustness | WHEN trained policy is evaluated with cost parameters shifted ±20% from training distribution, buyer welfare SHALL degrade <1% (acceptable) or <5% (investigate) | high |
| REQ-11 | Reproducibility | WHEN any experiment is run with the same seed, system SHALL produce identical results (deterministic solver + seeded RNG) | high |
| REQ-12 | Data Calibration Pipeline | WHEN OpenTender CSV data is provided, system SHALL extract lot-bidder statistics and fit supplier model parameters (cost distribution, participation logistic, markup GPV) | medium |
| REQ-13 | Configurable Difficulty | WHEN difficulty parameter is set (easy/medium/hard), system SHALL generate instances with N∈{10,20,50}, M∈{5,10,20}, varying synergy strength and supplier heterogeneity | medium |
| REQ-14 | Retrospective Evaluation | WHEN a real historical tender (lot structure + bids + awards) is provided, system SHALL compute counterfactual buyer welfare under the agent's recommended lot structure vs. the actual lot structure | medium |

## Key Decisions (with WHY)

| Decision | Choice | Why | Alternatives Considered |
|----------|--------|-----|------------------------|
| Action space | Sequential assignment (autoregressive) | Per-step action space is small (≤K+1). Proven pattern from pointer networks for TSP. PPO handles this naturally. | Full partition vector (too large), hierarchical (commits K early), graph-based (clustering step non-differentiable) |
| Observation | Fixed-size aggregation per lot | Simple, works for MVP. Avoids variable-length sequences. | Transformer attention (more expressive, premature for MVP) |
| Solver | OR-Tools CP-SAT (primary), CPLEX upgrade path | OR-Tools is free, open-source, pip-installable. CPLEX needs academic license. Both solve WDP in seconds. | Gurobi (commercial), custom B&B (unnecessary), neural solver (defeats the point) |
| Supplier model | Three independent layers (cost/participation/markup) | Each layer calibratable from different data sources. Separation of concerns. GPV works on the markup layer independently. | End-to-end neural simulator (not calibratable), full BNE model (intractable for combinatorial), fixed markup (too simple) |
| Reward | Terminal, sparse, pure buyer welfare | No human assumptions baked in. Sparse over N steps is fine for PPO (episodes are short). | Shaped per-step rewards (encodes heuristics), multi-objective (premature) |
| RL algorithm | PPO via stable-baselines3 | Well-understood, works for discrete actions, good baselines. SB3 is battle-tested. | SAC (continuous actions, wrong for discrete), DQN (no function approx advantage here), custom (unnecessary) |
| Item ordering | Canonical: sorted by ascending qualified supplier count | Hardest-to-source items first. Provides inductive bias. Alternative: random (forces order-invariance, harder). | Random, by category, by volume |
| Domain | Transportation/freight procurement | Cost structure is well-understood (distance + fuel + equipment). Public data exists (OpenTender, ProZorro). Large industrial relevance. | Direct materials (complex BOM), services (opaque costs), commodities (too simple) |

## Approach

### What to Extract/Adopt

1. **Gymnasium env pattern**: Copy from RL4CO's `RL4COEnvBase` for the constructive (autoregressive) environment pattern. DeepWiki `ai4co/rl4co` for architecture understanding. Adapt the step/reset/render lifecycle.
2. **OR-Tools WDP solver**: Use `ortools.sat.python.cp_model` directly. The WDP is a standard set packing ILP. No external framework needed.
3. **CATS instance generator**: Study `kevinlb1/CATS` for realistic combinatorial auction instance distributions. Adapt the "regions" distribution (most similar to transportation).
4. **stable-baselines3 PPO**: Standard training loop. `MlpPolicy` for the aggregated observation.
5. **GPV estimation**: Implement from the 2000 paper formula directly. It's ~50 lines of numpy (kernel density estimate + equilibrium inversion).

### What to Build from Scratch

1. **LotPartition data structure** — no existing library handles partitions as immutable functional objects
2. **TransportInstanceGenerator** — generates geographic lanes with distance-based costs, carrier capability sets
3. **Three-layer supplier simulator** — cost/participation/markup models
4. **Baseline partition strategies** — geographic clustering, equal-size, etc.
5. **Evaluation harness** — metrics computation, comparison framework

## Source Map (CRITICAL — /execute copies before rewriting)

| Requirement | Reference Source | Repo/File | What to Copy | What to Adapt |
|-------------|----------------|-----------|-------------|---------------|
| REQ-1 | RL4CO constructive env | `github.com/ai4co/rl4co` `rl4co/envs/common/base.py` | Step/reset lifecycle, observation space construction, action masking pattern | Replace routing problem with partition problem. Use standard Gymnasium not TorchRL. |
| REQ-2 | OR-Tools CP-SAT examples | `github.com/google/or-tools` `ortools/sat/samples/assignment_sat.py` | CP-SAT model construction, variable creation, constraint adding, solving | Formulate as set packing instead of assignment. Add capacity constraints. |
| REQ-4 | CATS regions distribution | `github.com/kevinlb1/CATS` `regions.c` | Geographic point generation, distance-based bid value computation | Convert from C to Python. Use for item location generation, adapt cost model. |
| REQ-7 | SB3 custom env training | `stable-baselines3` docs | PPO training loop, callback pattern, tensorboard integration | Wire up ProcurementEnv as custom env. Add evaluation callback with baseline comparison. |
| REQ-12 | GPV estimator | Guerre, Perrigne, Vuong (Econometrica 2000) Eq. 3-5 | Kernel density estimation of bid distribution, equilibrium inversion formula | Implement in numpy. ~50 lines. |

## Build Environment

- **Language/Runtime**: Python 3.12+
- **Package manager**: uv
- **Project config**: pyproject.toml (PEP 621)
- **Test command**: `uv run pytest tests/ -v --tb=short`
- **Lint command**: `uv run ruff check src/`
- **Format command**: `uv run ruff format src/ tests/`
- **Type check command**: `uv run mypy src/ --strict`
- **Env check command**: `uv run python -c "from procurement_gym import ProcurementEnv; import gymnasium; gymnasium.utils.env_checker.check_env(ProcurementEnv())"`
- **Train command**: `uv run python scripts/train.py --seed 42 --total-timesteps 500000`
- **Evaluate command**: `uv run python scripts/evaluate.py --policy results/ppo_best --n-eval 100`
- **Integration test**: `uv run pytest tests/integration/ -v`

### Dependencies

```toml
[project]
name = "procurement-gym"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "gymnasium>=1.0",
    "numpy>=2.0",
    "ortools>=9.10",
    "scipy>=1.14",
]

[project.optional-dependencies]
train = ["stable-baselines3>=2.4", "tensorboard>=2.18"]
data = ["pandas>=2.2", "requests>=2.32"]
dev = ["pytest>=8.3", "mypy>=1.13", "ruff>=0.8"]
```

### Project Layout

```
procurement-gym/
├── pyproject.toml
├── README.md
├── src/
│   └── procurement_gym/
│       ├── __init__.py           # Exports ProcurementEnv + registration
│       ├── env.py                # ProcurementEnv (Gymnasium)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── types.py          # Item, Supplier, ProcurementInstance
│       │   └── partition.py      # LotPartition
│       ├── suppliers/
│       │   ├── __init__.py
│       │   ├── base.py           # SupplierModel Protocol
│       │   ├── cost.py           # AdditiveWithSynergy cost model
│       │   ├── participation.py  # LogisticParticipation model
│       │   ├── markup.py         # CompetitionCompressedMarkup model
│       │   ├── simulator.py      # ThreeLayerSupplierSimulator (composes above)
│       │   └── gpv.py            # GPVEstimator
│       ├── solver/
│       │   ├── __init__.py
│       │   ├── base.py           # WDPSolver Protocol
│       │   └── ortools_solver.py # ORToolsWDPSolver
│       ├── instances/
│       │   ├── __init__.py
│       │   ├── base.py           # InstanceGenerator Protocol
│       │   └── transport.py      # TransportInstanceGenerator
│       ├── baselines/
│       │   ├── __init__.py
│       │   ├── base.py           # BaselinePolicy Protocol
│       │   ├── random_partition.py
│       │   ├── single_item.py
│       │   ├── geographic.py
│       │   ├── equal_size.py
│       │   └── exhaustive.py
│       └── evaluation/
│           ├── __init__.py
│           ├── metrics.py
│           └── benchmark.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── calibrate.py             # Fit supplier model from OpenTender data
├── tests/
│   ├── __init__.py
│   ├── test_types.py
│   ├── test_partition.py
│   ├── test_solver.py
│   ├── test_supplier_model.py
│   ├── test_transport_generator.py
│   ├── test_baselines.py
│   ├── test_env.py
│   ├── test_metrics.py
│   └── integration/
│       ├── __init__.py
│       ├── test_episode.py       # Full episode: reset → N steps → reward
│       └── test_training.py      # 1000-step PPO trains without crash
├── configs/
│   ├── default.yaml
│   ├── toy_2item.yaml
│   ├── small_20item.yaml
│   ├── medium_50item.yaml
│   └── large_100item.yaml
└── docs/
    └── research/                 # Research artifacts (already exist)
```

---

## Configuration Schema (Pydantic — the source of truth)

```python
from pydantic import BaseModel, Field

class SupplierModelConfig(BaseModel):
    synergy_weight: float = Field(0.1, description="Weight for geographic synergy savings")
    participation_w_cap: float = Field(2.0, description="Logistic weight for capability match")
    participation_w_size: float = Field(0.5, description="Logistic weight for log(lot_size)")
    participation_w_comp: float = Field(-0.3, description="Logistic weight for expected competition (negative = less entry when crowded)")
    markup_alpha_mean: float = Field(0.15, description="Mean markup intercept across suppliers")
    markup_alpha_std: float = Field(0.05, description="Std of markup intercept")
    markup_beta_mean: float = Field(0.5, description="Mean competition elasticity")
    markup_beta_std: float = Field(0.1, description="Std of competition elasticity")
    cost_noise_std: float = Field(0.05, description="Std of idiosyncratic cost noise (fraction of base cost)")

class InstanceConfig(BaseModel):
    n_items: int = Field(20, ge=2, le=500, description="Number of procurement items")
    n_suppliers: int = Field(10, ge=2, le=100, description="Number of suppliers")
    n_categories: int = Field(3, ge=1, le=20, description="Number of item categories")
    area_size: float = Field(100.0, description="Geographic area side length (km)")
    min_supplier_categories: int = Field(1, description="Min categories per supplier")
    max_supplier_categories: int = Field(2, description="Max categories per supplier")
    base_value_mean: float = Field(1000.0, description="Mean item value")
    base_value_std: float = Field(200.0, description="Std item value")

class EnvConfig(BaseModel):
    instance: InstanceConfig = Field(default_factory=InstanceConfig)
    supplier_model: SupplierModelConfig = Field(default_factory=SupplierModelConfig)
    k_max: int = Field(20, description="Maximum number of lots")
    solver_timeout_s: float = Field(60.0, description="Solver timeout in seconds")
    item_order: str = Field("ascending_qualified", description="Item presentation order: ascending_qualified | random | by_category")
    seed: int = Field(42, description="Random seed for reproducibility")
```

Agents load config via: `EnvConfig.model_validate(yaml.safe_load(open("configs/default.yaml")))`.
Config YAML files are serialized from this schema. Every parameter has a default — agents never guess values.

## Exact ILP Formulation (WDP — what OR-Tools must implement)

The Winner Determination Problem as solved by `ortools_solver.py`:

**Sets**:
- K = {1, ..., n_lots}: lots in the partition
- S = {1, ..., n_suppliers}: suppliers
- E_k ⊆ S: set of suppliers who entered lot k (after participation sampling)

**Decision Variables**:
- x_{sk} ∈ {0, 1}: 1 if supplier s is awarded lot k

**Parameters**:
- bid_{sk}: supplier s's bid for lot k (computed by supplier simulator)
- C_s: capacity of supplier s
- vol_k: total volume of lot k = Σ_{i∈L_k} volume(i)

**Formulation**:
```
minimize   Σ_k Σ_{s∈E_k} bid_{sk} · x_{sk}

subject to:
  (1) Σ_{s∈E_k} x_{sk} = 1           ∀k ∈ K        [each lot awarded to exactly one supplier]
  (2) Σ_k vol_k · x_{sk} ≤ C_s        ∀s ∈ S        [supplier capacity]
  (3) x_{sk} = 0                       ∀k, s ∉ E_k   [only entrants can win]
  (4) x_{sk} ∈ {0, 1}                  ∀k, s
```

**OR-Tools CP-SAT implementation pattern**:
```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()

# Scale bids to integers (CP-SAT requires int objectives)
SCALE = 1000
scaled_bids = {(s, k): int(bid[s][k] * SCALE) for s in entrants[k] for k in lots}

# Variables
x = {}
for k in lots:
    for s in entrants[k]:
        x[s, k] = model.new_bool_var(f"x_{s}_{k}")

# Constraint (1): each lot awarded to exactly one supplier
for k in lots:
    model.add_exactly_one(x[s, k] for s in entrants[k])

# Constraint (2): supplier capacity
for s in suppliers:
    model.add(
        sum(vol[k] * x[s, k] for k in lots if (s, k) in x) <= capacity[s]
    )

# Objective: minimize total cost
model.minimize(sum(scaled_bids[s, k] * x[s, k] for (s, k) in x))

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = timeout_s
status = solver.solve(model)

# Extract allocation
allocation = {}
if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    for (s, k) in x:
        if solver.value(x[s, k]) == 1:
            allocation[k] = s
```

**Edge case**: If a lot has zero entrants (no supplier participates), the lot is unserviced. The environment penalizes this with a large negative reward term: `unserviced_penalty = -10 * base_value * n_unserviced_lots`.

## Logging Specification

Use Python `logging` module + JSONL file output. No wandb (keep dependencies minimal).

```python
# In train.py
import logging, json

logger = logging.getLogger("procurement_gym")
handler = logging.FileHandler(f"results/{run_name}/train.jsonl")
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# Log each episode
logger.info(json.dumps({
    "episode": ep,
    "reward": float(reward),
    "n_lots": partition.n_lots,
    "avg_competition": float(avg_bidders_per_lot),
    "unserviced_lots": int(n_unserviced),
    "solver_time_s": float(solver_time),
}))
```

SB3 tensorboard logging is automatic via `tensorboard_log` parameter. No additional config needed.

---

## Phase Specifications

### Phase 1: Core Skeleton (Week 1-2)

**Goal**: A working Gymnasium environment that passes `check_env()` with a trivial supplier model and solver.

**Deliverables**:
1. `src/procurement_gym/core/types.py` — frozen dataclasses
2. `src/procurement_gym/core/partition.py` — LotPartition with immutable assign()
3. `src/procurement_gym/instances/transport.py` — generates N items as geographic points, M suppliers with capability sets
4. `src/procurement_gym/solver/ortools_solver.py` — WDP as set packing ILP
5. `src/procurement_gym/suppliers/simulator.py` — simplified (cost + fixed markup, no participation model yet)
6. `src/procurement_gym/env.py` — Gymnasium wrapper with sequential assignment action space
7. Tests for all of the above

**Exit criteria**:
```bash
uv run pytest tests/test_types.py tests/test_partition.py tests/test_solver.py tests/test_env.py -v  # all pass
uv run python -c "from procurement_gym import ProcurementEnv; import gymnasium; gymnasium.utils.env_checker.check_env(ProcurementEnv())"  # passes
uv run ruff check src/  # clean
uv run mypy src/ --strict  # clean
```

**Key implementation details for agents**:

`types.py` must define:
```python
@dataclass(frozen=True)
class Item:
    id: int
    category: str
    base_value: float
    x: float  # geographic longitude
    y: float  # geographic latitude
    volume: float
    qualified_supplier_ids: frozenset[int]

@dataclass(frozen=True) 
class Supplier:
    id: int
    categories: frozenset[str]
    capacity: float
    base_x: float  # hub location
    base_y: float
    cost_per_unit_distance: float
    fixed_cost: float
    markup_alpha: float  # GPV parameters
    markup_beta: float

@dataclass(frozen=True)
class ProcurementInstance:
    items: tuple[Item, ...]
    suppliers: tuple[Supplier, ...]
    seed: int
```

`partition.py` must define:
```python
@dataclass(frozen=True)
class LotPartition:
    assignments: tuple[int, ...]  # length N, assignments[i] = lot_id for item i, -1 = unassigned
    n_items: int
    
    @staticmethod
    def empty(n_items: int) -> "LotPartition": ...
    
    def assign(self, item_idx: int, lot_id: int) -> "LotPartition":
        """Return NEW partition with item assigned. Never mutates self."""
        ...
    
    def next_lot_id(self) -> int:
        """Return the next available lot ID (= max existing + 1, or 0 if empty)."""
        ...
    
    def lot_items(self, lot_id: int) -> tuple[int, ...]:
        """Return item indices assigned to this lot."""
        ...
    
    def is_complete(self) -> bool:
        """True if all items assigned."""
        ...
    
    @property
    def n_lots(self) -> int: ...
    
    @property
    def lot_ids(self) -> frozenset[int]: ...
```

`env.py` observation and action spaces:
```python
# Observation space (fixed-size)
K_MAX = 20  # max lots
D_ITEM = 6  # item feature dimension (category_encoded, base_value, x, y, volume, n_qualified)
D_LOT = 5   # lot summary dimension (n_items, mean_x, mean_y, total_volume, n_qualified_mean)
D_GLOBAL = 3  # (items_remaining, current_n_lots, total_items)

obs_space = gymnasium.spaces.Dict({
    "current_item": gymnasium.spaces.Box(-np.inf, np.inf, shape=(D_ITEM,)),
    "lot_summaries": gymnasium.spaces.Box(-np.inf, np.inf, shape=(K_MAX, D_LOT)),
    "global_context": gymnasium.spaces.Box(-np.inf, np.inf, shape=(D_GLOBAL,)),
    "action_mask": gymnasium.spaces.MultiBinary(K_MAX + 1),  # valid lot assignments
})

# Action space: assign current item to lot k (0..K_current) or new lot (K_current+1)
action_space = gymnasium.spaces.Discrete(K_MAX + 1)
```

### Phase 2: Supplier Model (Week 2-3)

**Goal**: Replace simplified supplier model with the three-layer model. Verify that reward varies meaningfully with lot structure.

**Deliverables**:
1. `suppliers/cost.py` — additive cost + synergy (geographic proximity)
2. `suppliers/participation.py` — logistic participation model
3. `suppliers/markup.py` — competition-compressed markup
4. `suppliers/simulator.py` — composes all three layers
5. Tests with parameter sensitivity: show that different lot structures produce different rewards

**Exit criteria**:
```bash
uv run pytest tests/test_supplier_model.py -v  # all pass
# Sensitivity test: same instance, 3 different lot structures, rewards differ by >5%
uv run python -c "
from procurement_gym import ProcurementEnv
env = ProcurementEnv(config='configs/small_20item.yaml')
# Run 3 episodes with different fixed partition strategies
# Assert rewards are different
"
```

**Key implementation details**:

Synergy function (geographic proximity):
```python
def synergy(lot_items: Sequence[Item], supplier: Supplier) -> float:
    """Negative cost = savings when items are geographically close.
    
    Items near each other and near the supplier's hub cost less to serve together.
    """
    if len(lot_items) <= 1:
        return 0.0
    # Pairwise distance between items in the lot
    coords = np.array([(item.x, item.y) for item in lot_items])
    pairwise_dist = pdist(coords)
    mean_pairwise = np.mean(pairwise_dist)
    # Distance from supplier hub to lot centroid
    centroid = coords.mean(axis=0)
    hub_dist = np.sqrt((centroid[0] - supplier.base_x)**2 + (centroid[1] - supplier.base_y)**2)
    # Synergy is negative cost (savings) — closer items = more savings
    return -SYNERGY_WEIGHT * (1.0 / (1.0 + mean_pairwise)) * (1.0 / (1.0 + hub_dist))
```

### Phase 3: Baselines + RL Training (Week 3-5)

**Goal**: Implement all baselines. Train PPO agent. Agent beats random baseline.

**Deliverables**:
1. All baseline implementations in `baselines/`
2. `scripts/train.py` — PPO training with SB3
3. `scripts/evaluate.py` — evaluation against all baselines
4. `evaluation/metrics.py` — buyer welfare, competition per lot, optimality gap
5. Training curves and baseline comparison plots

**Exit criteria**:
```bash
uv run pytest tests/test_baselines.py -v
uv run python scripts/train.py --config configs/small_20item.yaml --total-timesteps 500000 --seed 42
uv run python scripts/evaluate.py --policy results/ppo_small_20/best_model --config configs/small_20item.yaml --n-eval 100
# Agent mean welfare > random mean welfare (p < 0.05, paired t-test)
```

**Geographic clustering baseline** (the primary benchmark):
```python
def geographic_partition(instance: ProcurementInstance, n_lots: int) -> LotPartition:
    """K-means on item (x, y) coordinates."""
    coords = np.array([(item.x, item.y) for item in instance.items])
    kmeans = KMeans(n_clusters=n_lots, random_state=0).fit(coords)
    assignments = tuple(int(label) for label in kmeans.labels_)
    return LotPartition(assignments=assignments, n_items=len(instance.items))
```

### Phase 4: Data Calibration (Week 5-7)

**Goal**: Fit supplier model from real EU procurement data. Generate calibrated instances. Agent beats geographic clustering by >2%.

**Deliverables**:
1. `scripts/calibrate.py` — download + parse OpenTender data, fit model parameters
2. `instances/calibrated.py` — CalibratedInstanceGenerator
3. `suppliers/gpv.py` — GPV estimator
4. Trained agent on calibrated instances

**Data access commands**:
```bash
# OpenTender bulk download (Germany, transport sector CPV 60000000)
curl -o opentender_de.json "https://data.open-contracting.org/api/v1/search?country=DE&cpv=60000000&limit=10000"

# ProZorro transport tenders (CPV 60000000)  
curl -o prozorro_transport.json "https://public.api.openprocurement.org/api/2.5/tenders?mode=real_&opt_fields=lots,bids&classificationID=CPV-60000000"

# TED CSV subset
curl -o ted_csv.zip "https://data.europa.eu/api/hub/store/data/ted-csv-data-bulk-download.zip"
```

**Exit criteria**:
```bash
uv run python scripts/calibrate.py --data data/opentender_de.json --output configs/calibrated_de.yaml
uv run python scripts/train.py --config configs/calibrated_de.yaml --total-timesteps 1000000 --seed 42
uv run python scripts/evaluate.py --policy results/ppo_calibrated/best_model --config configs/calibrated_de.yaml --n-eval 100
# Agent welfare > geographic_clustering welfare by >2% (p < 0.05)
```

### Phase 5: Validation + Paper (Week 7-8)

**Goal**: Run all validation experiments. Produce paper-ready results.

**Deliverables**:
1. Toy validation (N=2, recover theory)
2. GPV robustness ablation (±20% parameter shift)
3. Scaling experiments (N=20, 50, 100)
4. Retrospective on one real EU transport tender
5. Paper draft sections: environment description, experiments, results

**Exit criteria**:
```bash
# Toy: N=2 symmetric, agent chooses optimal partition >95% of time
uv run python scripts/validate_toy.py --n-eval 1000

# GPV robustness: <5% welfare degradation at ±20%
uv run python scripts/validate_gpv_robustness.py --shift 0.2

# Scaling: agent beats geographic clustering at all scales
for n in 20 50 100; do
  uv run python scripts/evaluate.py --config configs/medium_${n}item.yaml --n-eval 50
done
```

---

## Knowledge Map

| Source | ID | Core Contribution | Verified? | Implication |
|--------|-----|------------------|-----------|-------------|
| Grimm et al. 2006 | Ch.7 Handbook of Procurement | Lot structure framework (qualitative) | Yes — read | Our work is the first computational approach |
| Subramaniam-Venkatesh 2009 | Marketing Science | 2-item bundle/split theory | Yes — read | Toy validation must match their results |
| GPV 2000 | Econometrica | Structural cost estimation from bids | Yes — formula verified | Foundation of our calibration pipeline |
| Sandholm 2002 | AAMAS | WDP inapproximability | Yes — read | Confirms WDP is hard in general but easy in practice |
| CABOB, Sandholm 2005 | Management Science | Specialized WDP solver | Yes — read | Confirms CPLEX handles our instances |
| RegretNet, Dutting 2019 | arXiv:1706.03459 | Neural mechanism design | Yes — read | Zero procurement application — our gap |
| PlanB&B 2025 | arXiv:2511.09219 | MCTS + learned B&B | Yes — read | Solves L2, not L1 — different problem |
| ProcureGym 2026 | arXiv:2603.23880 | Multi-agent procurement env | Yes — read | Narrow (pharma NVBP), single-item — our env is broader |
| You et al. 2026 | arXiv:2601.13489 | RegretNet IC is 70x worse | Yes — read | Validates our decision to use exact solver, not neural |
| Conitzer-Sandholm 2002 | AAMAS | Automated mechanism design | Yes — read | Optimizes within mechanism — we optimize problem formulation |
| RL4CO 2025 | KDD | 27 CO environments | Yes — DeepWiki | Architecture reference for constructive env pattern |
| OR-Gym 2020 | CMU/Dow | OR RL environments | Yes — DeepWiki | No auction/procurement — confirms our gap |
| OpenTender/DIGIWHIST | data.open-contracting.org | 35-country lot-level procurement data | Yes — verified access | Primary calibration source |
| ProZorro | public.api.openprocurement.org | Individual bid amounts per lot | Yes — verified access | Gold source for GPV estimation (has losing bids) |
| CATS | github.com/kevinlb1/CATS | Synthetic CA instance generator | Yes — repo exists | Reference for instance generation distributions |
| Stackelberg POMDP 2022 | arXiv:2210.03852 | RL for economic rule design | Yes — read | Closest prior art — continuous followers, not MILP |

## The Why Behind Everything

Procurement is a $13 trillion global market where the highest-leverage buyer decision — how to structure what gets auctioned — is made manually using spreadsheets and intuition. The academic community has spent decades optimizing the allocation step (Layer 2: WDP) which CPLEX already solves in seconds, while ignoring the formulation step (Layer 1: lot design) which has Bell(N) possible configurations and no algorithm.

We build ProcurementGym because:
1. **The problem is real**: Keelvar's multi-billion dollar business depends on lot optimization that today requires human experts
2. **The problem is formally open**: No theory beyond 2-item cases, no algorithms, no benchmarks
3. **The problem is RL-shaped**: Bell(N) action space too large to enumerate, structured enough to learn, verifiable via exact solver
4. **The field needs this**: No public procurement RL environment exists despite massive industrial relevance

The environment design follows one meta-principle: **use CPLEX/OR-Tools as the physics engine, not the agent.** The solver provides ground truth. The agent learns upstream design. This separation ensures every reward is verifiable and every result is reproducible.

The choice of transportation procurement as the initial domain is driven by three factors: (1) cost structures are well-understood and observable (distance + fuel + equipment), (2) public lot-level data exists (OpenTender, ProZorro), and (3) the competition-volume tradeoff is particularly clear (geographic lot bundling is the industry default, leaving room for RL to discover non-obvious improvements).
