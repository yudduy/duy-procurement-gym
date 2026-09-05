# Procurement RLVR: Architecture & Research Findings

> Last updated: 2026-04-13. Source: deep research across 30+ papers, DeepWiki analysis of AZR codebase, multi-model deliberation.

## Why RLVR Works (the mechanism, not the headline)

**RLVR does not teach new reasoning. It activates reasoning the base model already has.**

Three independent findings converge:

| Paper | Finding |
|-------|---------|
| 1-shot RLVR (2504.20571) | ONE training example → 36%→73.6% math accuracy. Distribution shaping, not learning. |
| Invisible Leash (2507.14843, Stanford/NVIDIA) | RLVR mathematically cannot create probability mass at zero-probability solutions. Net discovery rate <0.04. |
| Spurious Rewards (2506.10947) | Even RANDOM rewards improve performance on models with domain pretraining. RLVR amplifies what's there. |

**Implication**: If the base model has no procurement reasoning from pretraining, pure RLVR will fail. Must SFT first to install the domain prior.

### Five Necessary Conditions for RLVR

| Condition | Code/Math | Procurement |
|-----------|-----------|-------------|
| Verification cheaper than solving | O(ms) | O(seconds) — OK |
| Compositionality (multi-step) | Proofs/programs build on parts | Lots compose into auctions with cross-lot coupling |
| Difficulty spectrum | Easy→hard problems | `coupling_strength` 0.0→1.0, N=5→500 |
| Structured solution space (reasoning correlates with outcome) | Correct logic → correct answer | Good lot design → measurably better buyer welfare |
| **Latent competence in base model** | Massive code/math pretraining | **Weak — requires SFT to install procurement prior** |

### Three Procurement-Specific Challenges (the research contribution)

1. **Stochastic verification** — supplier participation is random, adding noise to rewards. Fix: drop std normalization from GRPO (Stanford uncalibrated reasoning paper, 2508.11800). Average over simulation rollouts.
2. **Continuous reward** — buyer welfare is a float, not binary pass/fail. Fix: continuous soft rewards work and even outperform binary (Crossing the Reward Bridge, Tencent, 2503.23829).
3. **Strategic opponents** — suppliers respond to the agent's lot design decisions. **Genuinely novel — no prior RLVR work addresses environments with strategic agents.** This IS the contribution.

---

## The SOTA Pipeline (everyone does the same thing)

Nobody does pure RLVR from scratch. DeepSeek-R1 has FOUR stages:

| Stage | What | Data Source |
|-------|------|-------------|
| 1. Cold Start SFT | Stabilize early RL | Model's own RL trajectories OR teacher model distillation |
| 2. Reasoning RL (GRPO) | RLVR against verifier — reasoning emerges here | Verifier provides pass/fail or continuous reward |
| 3. Rejection Sampling SFT | Generate 800K trajectories, filter correct ones, SFT again | Model generates, verifier filters |
| 4. Final RL | Mixed rewards (rule-based + model-based) | Combination |

**The SFT data is NOT human-written.** It is model-generated, verifier-filtered. The model tries things, the verifier says pass/fail, you keep the passes.

**Exception**: DeepSeek-R1-Zero (pure RLVR, no SFT) DOES work for reasoning benchmarks. But it produces unreadable output and fails on general tasks. For a domain-specific tool, R1-Zero style might work.

### Key Reference Papers

| Paper | Domain | SFT Examples | Result |
|-------|--------|-------------|--------|
| DeepSeek-R1 (2501.12948) | Math/Code | ~800K (thousands for cold start) | SOTA reasoning |
| Recon (2506.00577, Duke/UPenn) | Economics | 868 economic reasoning examples | Nash equilibrium reasoning emerges, transfers to multi-agent games |
| CO Paper (2509.16865, TU Eindhoven) | TSP/Bin Packing | 500K solver-generated solutions | 100% feasibility, 1-8% optimality gap on 7 NP-hard problems |
| Absolute Zero (2505.03335) | Math via code | Zero external data (self-play) | Beats models trained on 10K+ human examples |

---

## Our Pipeline: Procurement RLVR

### Phase 1: SFT Data Generation (install procurement reasoning)

```
1. GENERATE: TransportInstanceGenerator creates 1000+ instances
   (items with geography, categories, volumes + suppliers with capabilities)

2. SOLVE: For each instance, evaluate multiple lot partition strategies:
   - Random partitions (100x)
   - Geographic clustering (k=2..10)
   - Category grouping
   - Equal-size splits
   - Greedy search
   → Evaluate each with SupplierSimulator + ORToolsWDPSolver
   → Record: partition, buyer welfare, which packages triggered, solve stats

3. RANK: Pick the best partition per instance. Record score breakdown:
   - "Package discount from supplier 3 triggered (lots 1+2 bundled)"
   - "Competition increased: 4 bidders vs 2 in category-grouped baseline"

4. TRACE: Feed instance + best partition + score breakdown to LLM:
   "Given this procurement instance and this optimal lot structure,
    explain step-by-step WHY this structure achieves the best buyer welfare."
   → LLM generates reasoning trace
   → Filter: only keep traces where re-running the partition confirms the score

5. SFT: Fine-tune Qwen2.5-3B on ~1000 (instance, reasoning, partition) examples
```

### Phase 2: RLVR Training (sharpen procurement reasoning)

```
1. GRPO with ProcurementVerifier as reward function
2. Curriculum: coupling_strength 0.0 → 0.3 → 0.5 → 0.7 → 1.0
3. Advance difficulty when model's average reward beats category-grouping baseline
4. No std normalization (stochastic verifier)
5. KL penalty to prevent distribution collapse
```

### Phase 3: Evaluation

```
- Compare: RLVR model vs SFT-only vs PPO vs all baselines
- Analyze chain-of-thought for genuine procurement reasoning
- Test on unseen instance distributions (generalization)
```

---

## Absolute Zero Reasoner (AZR) Analysis

> Based on DeepWiki deep dive of `LeapLabTHU/Absolute-Zero-Reasoner`

### What AZR Does (the self-play magic)

AZR uses a single model as both **proposer** and **solver** in a self-play loop:

```
PROPOSE phase:
  Model generates problems (Python programs)
  → Executor validates (runs code, checks determinism)
  → Valid problems stored in DatasetManager pool
  → Proposer rewarded for "learnable" problems (intrinsic rewards: complexity, diversity)

SOLVE phase:
  Sample problems from the pool
  → Model predicts inputs/outputs/functions
  → Executor checks correctness
  → Solver rewarded for accuracy

Both phases use PPO/GRPO to update the SAME model.
```

The key innovation: the model learns to generate problems at the **edge of its own competence** — hard enough to learn from, easy enough to sometimes solve. This creates automatic curriculum without manual difficulty tuning.

### How AZR Maps to Procurement

| AZR (Code) | Procurement Equivalent |
|-------------|----------------------|
| PROPOSE: Generate Python programs | PROPOSE: Generate procurement instances (or adversarial lot structures) |
| SOLVE: Predict inputs/outputs | SOLVE: Design lot partitions for given instances |
| VERIFY: Python executor runs code | VERIFY: Supplier simulator + OR-Tools solves WDP |
| REWARD: accuracy + intrinsic (complexity, diversity) | REWARD: buyer welfare + intrinsic (competition diversity, package utilization) |
| DIFFICULTY: Composite functions | DIFFICULTY: coupling_strength curriculum + composite instances |

### Fork Assessment

AZR is **tightly coupled to Python code execution**. No plugin system. The `PythonExecutor` and `SandboxfusionExecutor` are hardcoded throughout.

**~10-15 files need modification** to adapt for procurement:

| Component | File(s) | Change Required |
|-----------|---------|-----------------|
| Executor | `python_executor.py`, `sandboxfusion_executor.py` | Replace with `ProcurementExecutor` |
| Reward Manager | `rewards/reward_managers.py` | Replace `CodeIORewardManager` with `ProcurementRewardManager` |
| Code Parsing | `rewards/code_reward.py` | Replace with lot partition parsing |
| Problem Types | `configs/azr_ppo_trainer.yaml` | `code_i/o/e/f` → `propose_instance/solve_partition` |
| Intrinsic Rewards | `rewards/reward_managers.py` | Halstead/AST → competition/synergy metrics |
| Validation | `utils/code_utils/checks.py` | Code validity → partition feasibility |
| Training Scripts | `scripts/selfplay/*.sh` | Update executor and problem type configs |
| Seed Data | `data_construction/constructor.py` | Bootstrap with procurement instances |
| Type System | `utils/code_utils/` | Remove Python-specific type inference |

**Estimated effort**: ~2 weeks for a working fork. The core RL loop (veRL + Ray + GRPO) stays intact.

### Fork vs TRL Decision

| | Fork AZR | TRL GRPOTrainer |
|---|---------|----------------|
| **Setup time** | ~2 weeks | ~2 days |
| **Self-play** | Built-in (propose+solve) | Manual (solve only) |
| **Difficulty scaling** | Automatic (edge-of-competence) | Manual curriculum |
| **Infrastructure** | Ray + vLLM + veRL (heavy) | Single GPU, HF ecosystem |
| **Risk** | Integration complexity | Simpler but no self-play |

**Recommendation**: Start with TRL to prove RLVR works for procurement at all. If it works, fork AZR for self-play. If it doesn't work with TRL, it won't work with AZR either — the bottleneck is the verification signal, not the training infrastructure.

---

## What We've Built (Current State)

### ProcurementGym: 1,365 lines source, 89/89 tests green

| Module | Lines | Status |
|--------|-------|--------|
| `core/types.py` | 55 | Frozen dataclasses: Item, Supplier, PackageBid, ProcurementInstance |
| `core/partition.py` | 63 | Immutable LotPartition |
| `config.py` | 77 | Pydantic: EnvConfig, InstanceConfig, SupplierModelConfig |
| `instances/transport.py` | 126 | TransportInstanceGenerator (geographic, seeded) |
| `suppliers/cost.py` | 72 | Fixed + distance + synergy + noise |
| `suppliers/participation.py` | 42 | Logistic participation model |
| `suppliers/markup.py` | 22 | Competition-compressed markup (GPV-ready) |
| `suppliers/packages.py` | 112 | Cross-lot package bid generation |
| `suppliers/simulator.py` | 140 | ThreeLayerSupplierSimulator |
| `solver/ortools_solver.py` | 262 | CP-SAT WDP with package bids + volume tiers |
| `env.py` | 368 | ProcurementEnv (Gymnasium) |

### Cross-Lot Coupling (what makes the problem genuinely hard)

- **Package bids**: Suppliers offer discounts for lot bundles → lot boundaries affect ALL other lots
- **Volume tiers**: Non-linear pricing at thresholds → non-convex rewards
- **Difficulty curriculum**: `coupling_strength` 0.0 (trivial, "group by category" dominates) → 1.0 (NP-hard, no heuristic dominates)

### What's Missing

- Baselines (random, geographic, category, equal-size, single-item)
- Evaluation harness (metrics, comparison scripts)
- RLVR verifier (text interface to the gym)
- SFT data generation pipeline
- Training scripts (SFT, GRPO, PPO)
- Config YAML files for difficulty levels

---

## Knowledge Map

| Source | ID | Contribution | Verified | Implication |
|--------|----|-------------|----------|-------------|
| DeepSeek-R1 | 2501.12948 | 4-stage pipeline, GRPO, cold-start SFT from model's own trajectories | Paper read | Template for our pipeline |
| 1-shot RLVR | 2504.20571 | 1 example unlocks reasoning — distribution shaping | Paper read | Small SFT datasets may suffice |
| Invisible Leash | 2507.14843 | RLVR can't create new capabilities, only sharpen | Paper read | Must SFT first for procurement prior |
| Recon | 2506.00577 | 868 econ examples → economic reasoning via RLVR | Paper read | Proof that strategic reasoning is RLVR-compatible |
| CO Paper | 2509.16865 | SFT+GRPO on 7 NP-hard problems → near-optimal | Paper read | Proof that optimization reasoning works |
| Uncalibrated Reasoning | 2508.11800 | Std normalization breaks GRPO for stochastic rewards | Paper read | MUST disable std norm for procurement |
| Crossing the Reward Bridge | 2503.23829 | Continuous soft rewards work, outperform binary | Paper read | Our continuous welfare reward is fine |
| Edge of Competence | 2602.14872 | Train at difficulty frontier for grokking | Paper read | Curriculum: coupling_strength 0→1 |
| Absolute Zero | 2505.03335 | Self-play reasoning, zero external data | Paper + code read | Self-play concept for procurement |
| Spurious Rewards | 2506.10947 | Random rewards work on models with domain priors | Paper read | RLVR amplifies pretraining, doesn't create |
| TRL GRPOTrainer | HF library | Production RLVR, custom reward callables | Docs verified | Fastest path to prototype |
| OpenRLHF | GitHub | Distributed RLVR, Ray+vLLM, 3.1x faster than TRL | Repo verified | Scale-up option |
| AZR (LeapLabTHU) | GitHub | Self-play RLVR, veRL-based, auto curriculum | DeepWiki deep dive | Fork target for self-play upgrade |
| Keelvar | Product | Industry SOTA: combinatorial optimization for sourcing | Product research | Benchmark target |
| Grimm et al. 2006 | Handbook Ch.7 | Lot structure theory (qualitative, no algorithm) | Read | We're first to compute it |
| GPV 2000 | Econometrica | Structural cost estimation from bid data | Formula verified | Future calibration pipeline |
