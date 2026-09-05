# ProcurementGym: Architecture Specification

## Overview

An RL environment where an agent learns to partition N procurement items into K lots to maximize buyer welfare. CPLEX/OR-Tools solves the inner WDP exactly. Suppliers are simulated with GPV-calibrated cost/bid models.

## 1. Action Space: Sequential Assignment

The agent assigns items to lots **one at a time** (autoregressive):
- At step t: observe item t's features + current partial partition
- Action: assign item t to an existing lot OR open a new lot
- Action space per step: at most K_current + 1 (small, discrete)
- Episode length: N steps (one per item)

Items presented in canonical order (sorted by number of qualified suppliers, descending). This provides inductive bias — hardest-to-source items first.

## 2. State Representation: Fixed-Size Aggregation

```
observation = {
    "current_item": (d_item,)      # features of item being assigned
    "lot_summaries": (K_max, d_lot) # per-lot aggregates (size, mean cost, n_suppliers, category dist)
    "global": (d_global,)           # items remaining, current lot count, market context
}
```

Start with aggregation. Upgrade to transformer attention over items/lots if needed.

## 3. Supplier Simulation: Three-Layer Model

Each layer independently calibratable:

### Cost Model
```python
c_s(lot) = sum(base_cost_s(i) for i in lot) + synergy(lot, s) + noise
```
- `base_cost_s(i)`: supplier s's cost for item i
- `synergy`: scope economies (e.g., geographic proximity of transport lanes)
- `noise`: idiosyncratic cost variation

### Participation Model
```python
P(s enters lot) = sigmoid(w_cap * capability_match 
                         + w_size * log(|lot|) 
                         + w_comp * expected_competition 
                         + bias_s)
```

### Markup Model (GPV-calibratable)
```python
markup_s(lot) = alpha_s * (1 / n_competitors(lot))^beta_s
bid_s(lot) = cost_s(lot) * (1 + markup_s(lot))
```
- alpha, beta estimated from historical bid data via GPV structural estimation
- More competition compresses markup — the core mechanism RL should discover

## 4. Solver Integration

- **Primary**: OR-Tools CP-SAT (open-source, free)
- **Upgrade**: CPLEX/Gurobi for production instances
- **Interface**: Protocol (dependency injection)
- **Timeout**: 60s default, returns best incumbent if timeout
- **WDP formulation**: minimize total award cost subject to lot coverage constraints

## 5. Reward: Pure Buyer Welfare

```python
reward = sum(value(item) for item in allocated) - sum(price_paid)
```

No auxiliary shaping. Sparse reward over N steps is fine for PPO (this is not a long-horizon problem).

## 6. Project Structure

```
procurement_gym/
├── env.py                     # ProcurementEnv (Gymnasium API)
├── core/
│   ├── types.py               # Item, Supplier, ProcurementInstance (frozen dataclasses)
│   ├── partition.py           # LotPartition (immutable, functional)
│   └── instance.py            # ProcurementInstance factory
├── suppliers/
│   ├── base.py                # SupplierModel (Protocol)
│   ├── cost.py                # CostModel
│   ├── participation.py       # ParticipationModel  
│   ├── markup.py              # MarkupModel
│   └── gpv.py                 # GPVEstimator
├── solver/
│   ├── base.py                # WDPSolver (Protocol)
│   └── ortools_solver.py      # ORToolsSolver
├── instances/
│   ├── transport.py           # TransportInstanceGenerator
│   └── calibrated.py          # CalibratedGenerator (from real data)
├── baselines/
│   ├── random_partition.py
│   ├── single_item.py
│   ├── geographic.py
│   └── exhaustive.py          # Brute-force (small N)
└── evaluation/
    ├── metrics.py             # Welfare, competition, optimality gap
    └── benchmark.py           # Benchmark runner
```

## 7. Data Sources for Calibration

### Primary (lot-level bid data)

| Source | Coverage | Key Fields | Access |
|--------|----------|-----------|--------|
| **OpenTender/DIGIWHIST** | 35 EU countries, 2006+ | Lot structure, CPV codes, bidder count, award values | `data.open-contracting.org` bulk CSV/JSON |
| **ProZorro (Ukraine)** | 2016+, millions of tenders | Individual bid amounts per lot (including losers!) | `public.api.openprocurement.org` |
| **TED CSV** | EU-wide, 800K notices/year | Lot IDs, CPV codes, offer counts, award values | `data.europa.eu/data/datasets/ted-csv` |

### Supporting

| Source | Use |
|--------|-----|
| GPPD (72M contracts, Mendeley) | Cross-country calibration |
| US FPDS (100M actions, Figshare) | Supplier behavior patterns (no lot structure) |
| CATS (GitHub: kevinlb1/CATS) | Synthetic combinatorial auction instances |

### Calibration Pipeline

```
OpenTender/ProZorro historical data
  → Extract: items per lot, bidder counts per lot, award prices
  → GPV estimation: infer cost distributions from bid data
  → Fit participation model: logistic regression on entry data
  → Fit synergy model: cost regression on awarded contracts
  → Generate: unlimited synthetic instances from fitted distributions
```

## 8. Build Order

### Phase 1: Skeleton (Week 1-2)
- `core/types.py` — frozen dataclasses for Item, Supplier, ProcurementInstance
- `core/partition.py` — immutable LotPartition with assign() operation
- `instances/transport.py` — synthetic transport lane generator
- `solver/ortools_solver.py` — WDP solver
- `env.py` — Gymnasium wrapper
- **Exit criterion**: `gymnasium.utils.env_checker.check_env(env)` passes

### Phase 2: Supplier Model (Week 2-3)
- Three-layer supplier simulation (cost + participation + markup)
- Parameter sensitivity tests
- **Exit criterion**: RL agent (random policy) completes episodes, reward varies with lot structure

### Phase 3: Baselines + Training (Week 3-5)
- Implement all baselines (random, single-item, geographic, exhaustive)
- Train PPO agent (stable-baselines3)
- **Exit criterion**: RL agent beats random baseline

### Phase 4: Calibration (Week 5-7)
- Download OpenTender + ProZorro data
- Fit GPV cost model
- Fit participation + synergy models
- Generate calibrated instances
- **Exit criterion**: RL agent trained on calibrated instances beats geographic clustering by >2%

### Phase 5: Validation (Week 7-8)
- Toy case: 2-item symmetric → must recover known optimum
- GPV robustness: ±20% parameter shift
- Retrospective on one real EU transport tender
- **Exit criterion**: paper-ready results
