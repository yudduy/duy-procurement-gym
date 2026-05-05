# ProcurementGym

A Gymnasium-compatible RL environment for procurement **lot-structure optimization** — the buyer-side decision of how to partition a set of items into lots before running a sealed-bid auction. The environment is a verification harness, not a simulation: supplier participation, costs, and markups are sampled from a calibratable three-layer behavioral model, and the resulting Winner Determination Problem ([WDP](https://en.wikipedia.org/wiki/Winner_determination_problem)) is solved **exactly** via [OR-Tools CP-SAT](https://developers.google.com/optimization/cp/cp_solver). Reward = realized buyer welfare under the optimal allocation. Built as a CS224R (Stanford Deep RL) course-project environment but designed to outlive the course as standalone infrastructure for procurement-RL or for combinatorial-auction verification work.

Running updates at [docs/LOG.md](docs/LOG.md).

## Background and motivation

Procurement-as-RL sits at the intersection of two literatures that don't usually talk to each other:

- **Combinatorial auctions and the WDP.** Allocating heterogeneous items to bidders with bundle-valuations is the canonical NP-hard set-packing problem. CATS (de Vries & Vohra 2003) and follow-ups settled the algorithmic ground for *winner determination given bids*. CP-SAT and modern MIP solvers handle realistic instances exactly.
- **RL with verifiable rewards (RLVR).** When an environment provides exact, programmatic reward (DeepSeek-R1, AlphaProof, GSM-Verifier-style training), the agent learns from ground truth instead of a learned critic. This sidesteps reward-hacking and makes credit assignment clean.

The empty cell, which neither literature occupies cleanly: **the buyer's lot-structure decision is itself a sequential combinatorial choice that selects which downstream WDP gets solved.** Lot design changes which suppliers can participate, which packages become legal, and which capacity constraints bind. The current procurement-RL literature mostly bolts a learned policy onto a hand-coded simulator; the WDP is approximated, not solved.

ProcurementGym takes the RLVR position seriously: the auction outcome is computed by an exact solver, not a neural critic. The agent's only optimization target is realized buyer welfare under the *true* optimal allocation given simulated bids. The behavioral model is parametric and adversarially testable — a coupling-strength knob slides from fully stochastic to fully deterministic verification (the AlphaProof regime).

For the long-form thesis, design rationale, validation record, and open questions, see [docs/LOG.md](docs/LOG.md).

## Experimental Setup

The agent assigns N items to lots one at a time. After all items are assigned, the environment runs the three-layer supplier model, generates package bids, and hands the result to CP-SAT. The terminal reward is the optimal buyer welfare; per-step rewards are zero (sparse).

```
                    ┌───────────────────────────────┐
  current_item ──▶  │  Agent / Policy               │ ─▶ action ∈ {0..K_existing, new_lot}
  lot_summaries ──▶ │  (assigns one item per step)  │
  global_context ──▶│                               │
  action_mask  ───▶ │                               │
                    └───────────────┬───────────────┘
                                    │   step()
                                    ▼
                    ┌───────────────────────────────┐
                    │  LotPartition (immutable)     │
                    │  assignments: tuple[int, ...] │
                    └───────────────┬───────────────┘
                                    │   on terminal step
                                    ▼
       ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐
       │ ThreeLayerSim    │    │ Package bids     │    │ ORToolsWDPSolver     │
       │   eligibility    │ ─▶ │   coupling-gated │ ─▶ │   CP-SAT, exact      │ ─▶ allocation, cost
       │   participation  │    │   synergy filter │    │   capacity constraints│
       │   cost + markup  │    │                  │    │   volume tier rebates │
       └──────────────────┘    └──────────────────┘    └──────────┬───────────┘
                                                                  │
                                                                  ▼
                                          reward = Σ serviced_value
                                                 − total_cost
                                                 − unserviced_penalty
                                                 − lot_overhead · n_lots
```

The **observation space** is a `Dict` with `current_item` (6-dim), `lot_summaries` (K_MAX × 8), `global_context` (5-dim), and an `action_mask` over the K_MAX+1 discrete actions. The **action space** is `Discrete(K_MAX+1)`: assign to one of the existing lots, or open a new lot. K_MAX = 20 by default.

The **three-layer supplier model** is the calibration surface:

1. **Cost** — distance-based per-item plus fixed cost per lot, with geographic synergy (pairwise item proximity * centroid-to-supplier proximity, normalized by area). Economy-of-scale discount kicks in for large lots, gated by `coupling_strength`.
2. **Participation** — logistic in `(capability_match, log lot_size, n_eligible, geo_proximity, supplier_bias)`. At `coupling_strength=1.0` with `noise_reduction=1.0`, participation collapses to a deterministic threshold (the AlphaZero/DeepSeek regime — every run on the same partition produces the same bids).
3. **Markup** — competition compression: `markup = α · (1/n_competitors)^β`. Calibratable to GPV (Guerre, Perrigne, Vuong 2000) first-price-auction markups.

`coupling_strength ∈ [0, 1]` is the single knob that toggles the regime: at 0, lots are independent and the WDP factorizes; at 1, package bids enable cross-lot synergies, capacity binds, and the problem is genuinely combinatorial. Volume-tier rebates and package-bid generation are gated by this parameter.

### What the verifier guarantees

The `ProcurementVerifier` (TRL `GRPOTrainer`-compatible reward function) wraps `evaluate_partition` so an LLM completion can be scored end-to-end:

1. Parse `<partition>[...]</partition>` from the completion (three-stage fallback: tag → JSON → bracketed-int).
2. Validate completeness (every item assigned, no negative lot IDs).
3. Run the simulator + CP-SAT solver at a fixed seed.
4. Return realized buyer welfare, or a per-item parse-failure penalty.

Determinism is non-negotiable in this path. With `coupling_strength=1.0, noise_reduction=1.0`, every step is reproducible: same partition → same supplier biases → same costs → same CP-SAT objective.

## Repository layout

```
src/procurement_gym/
  config.py                  # Pydantic schema: EnvConfig, InstanceConfig, SupplierModelConfig
  env.py                     # ProcurementEnv (Gymnasium); sequential lot assignment; terminal WDP reward
  evaluation.py              # standalone evaluate_partition + multi-seed statistics
  __init__.py                # gymnasium.register("Procurement-v0", ...)

  core/
    types.py                 # frozen dataclasses: Item, Supplier, PackageBid, ProcurementInstance
    partition.py             # immutable LotPartition (functional .assign() returns a new partition)

  instances/
    transport.py             # synthetic transport-procurement instance generator (geographic, capability-gated)

  suppliers/
    cost.py                  # additive base cost + geographic synergy + noise (deterministic at coupling=1)
    participation.py         # logistic entry model
    markup.py                # competition-compressed markup α·(1/n_comp)^β
    packages.py              # cross-lot package bid generation (coupling-gated, synergy-filtered)
    simulator.py             # composes the three layers into a full SimulationResult

  solver/
    ortools_solver.py        # ORToolsWDPSolver — set-packing ILP via CP-SAT, package bids, capacity, volume tiers

  verifier/
    serializer.py            # ProcurementInstance → LLM prompt; LotPartition → tagged text
    parser.py                # LLM text → LotPartition (tag → JSON → regex fallback)
    reward.py                # ProcurementVerifier — TRL GRPOTrainer-compatible reward_fn

  baselines/
    strategies.py            # 5 baselines: random, single-item, category, geographic (k-means), equal-size

scripts/
  setup_training.sh          # B200 setup (no-pip-torch, CUDA env, package install)
  train_sft.py               # LoRA SFT on search-optimized partitions (TRL SFTTrainer)
  train_grpo.py              # GRPO with ProcurementVerifier as reward (TRL GRPOTrainer, LoRA)
  eval_model.py              # held-out eval vs category / geographic / equal-size baselines

sft_data.jsonl               # search-optimized SFT data (~1.6 MB)
pyproject.toml               # uv build, Python 3.12, mypy strict, ruff
```

## Setup

```bash
# 1. Clone
git clone https://github.com/yudduy/duy-procurement-gym.git
cd duy-procurement-gym

# 2. Install (uv recommended; pip works)
uv sync                              # or: pip install -e ".[dev]"
```

## Using the environment

```python
import gymnasium
import procurement_gym  # registers Procurement-v0

env = gymnasium.make("Procurement-v0")
obs, info = env.reset(seed=0)

done = False
while not done:
    action = env.action_space.sample()        # replace with your policy
    valid = obs["action_mask"]
    action = int(action) if valid[action] else int(valid.argmax())
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(f"buyer welfare = {reward:.1f}, lots used = {info['n_lots']}, status = {info['status']}")
```

For a deterministic verification regime (the RLVR setting):

```python
from procurement_gym.config import EnvConfig, SupplierModelConfig

cfg = EnvConfig(supplier_model=SupplierModelConfig(coupling_strength=1.0, noise_reduction=1.0))
env = gymnasium.make("Procurement-v0", config=cfg)
```

For LLM-driven training, `ProcurementVerifier.reward_fn` plugs straight into `trl.GRPOTrainer`:

```python
from procurement_gym.config import InstanceConfig, SupplierModelConfig
from procurement_gym.instances.transport import TransportInstanceGenerator
from procurement_gym.verifier.reward import ProcurementVerifier

sc = SupplierModelConfig(coupling_strength=1.0)
ic = InstanceConfig(n_items=20, n_suppliers=10, n_categories=3)
instance = TransportInstanceGenerator(ic, sc).generate(seed=42)
verifier = ProcurementVerifier(instance, sc)
# reward_fn(completions, **kwargs) -> list[float], TRL-compatible
```

## Training pipeline

```bash
# 1. SFT on search-optimized lot designs (LoRA, Qwen2.5-7B by default)
python scripts/train_sft.py Qwen/Qwen2.5-7B-Instruct sft_data.jsonl checkpoints/sft

# 2. GRPO with the procurement verifier as reward
python scripts/train_grpo.py checkpoints/sft/final checkpoints/grpo 1000

# 3. Held-out evaluation vs category / geographic / equal-size baselines
python scripts/eval_model.py checkpoints/grpo/final 20 700000
```

`scripts/setup_training.sh` is the B200-aware remote setup (skips `pip install torch` so the SM_100 build survives, sets `TRITON_PTXAS_PATH`, attempts `fa4` and falls back to eager).

## References

The work in this repo builds on:

- **OR-Tools CP-SAT**: Perron & Furnon, *OR-Tools v9*. Google. [github.com/google/or-tools](https://github.com/google/or-tools).
- **Combinatorial auctions and the WDP**: de Vries & Vohra (2003). *Combinatorial auctions: A survey.* INFORMS Journal on Computing 15(3): 284–309.
- **CATS (instance generator)**: Leyton-Brown, Pearson & Shoham (2000). *Towards a universal test suite for combinatorial auction algorithms.* EC '00.
- **First-price auction calibration**: Guerre, Perrigne & Vuong (2000). *Optimal nonparametric estimation of first-price auctions.* Econometrica 68(3): 525–574.
- **RLVR / verifier-based RL**: DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:[2501.12948](https://arxiv.org/abs/2501.12948).
- **TRL / GRPO**: von Werra et al. *TRL: Transformer Reinforcement Learning.* [github.com/huggingface/trl](https://github.com/huggingface/trl).
- **Gymnasium**: Towers et al. (2023). *Gymnasium.* [github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

CS224R (Stanford Deep RL, Spring 2026) is the course this environment was first built for; the gym itself is course-independent.

## License

MIT.
