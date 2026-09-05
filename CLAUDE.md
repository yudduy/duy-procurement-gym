# ProcurementGym

## What This Is

An RLVR (RL with Verifiable Rewards) environment for procurement lot structure optimization. An LLM learns to design procurement auctions — partitioning items into lots — verified by an exact combinatorial solver (OR-Tools CP-SAT).

**The insight**: CPLEX/OR-Tools solves the allocation problem (WDP) in seconds. Nobody has automated the upstream question — how to structure WHAT gets auctioned. Bell(50) ~ 10^47 possible lot partitions. Too large to enumerate, structured enough to learn.

## Architecture

```
LLM proposes lot partition
  → Supplier Simulator (cost + participation + markup + package bids)
    → OR-Tools CP-SAT (exact WDP solver — the verifier)
      → Buyer Welfare = value - cost (the reward)
```

Two interfaces share the same core:
- **ProcurementEnv** (Gymnasium) — numpy obs/action, for PPO baselines
- **ProcurementVerifier** (planned) — text in → float reward, for LLM RLVR

## Cross-Lot Coupling (what makes it hard)

At `coupling_strength=0`, "group by category" trivially dominates. At `coupling_strength>0`:
- **Package bids**: Suppliers offer discounts for winning lot bundles → lot boundaries affect ALL lots
- **Volume tiers**: Non-linear pricing at thresholds → non-convex rewards
- This makes lot design genuinely NP-hard and gives RL something to learn.

## Current State (April 2026)

**140/140 tests green.** Gym + baselines + evaluation + coupling fix v2 complete.

Built: frozen dataclasses, LotPartition, TransportInstanceGenerator, three-layer supplier model,
OR-Tools CP-SAT solver, Gymnasium env, 5 baseline strategies, standalone evaluator, coupling mechanism.

**Not yet built**: RLVR verifier (text→float), SFT data pipeline, training scripts.

## Commands

```bash
uv run pytest tests/ -v --tb=short                    # all tests (140)
uv run python scripts/adversarial_coupling_test.py     # adversarial suite
uv run python scripts/adversarial_parallel.py          # parallel version (16 workers)
uv run python scripts/validate_reward_signal.py        # full validation
```

## Signal Quality Diagnostics (MUST READ)

Three literature-grounded metrics determine if the environment is RLVR-ready:

### Youden's Index J (from RLVεR, 2601.04411)
J = TPR - FPR. Measures: does the verifier correctly rank "good" vs "bad" partitions?
- J > 0.5: Strong signal. RLVR will learn.
- J > 0.2: Moderate. Learning possible but slow.
- J ≈ 0: Not learnable. Noise drowns signal. More compute won't help.
- **Current measurement: J = 0.025 (WEAK).** The verifier barely distinguishes good from bad.

### Effective Prompt Ratio (from RLVE, 2511.07317)
EPR = fraction of instances where different partitions produce distinguishable rewards.
- EPR > 0.8: Good. Most instances provide learning signal.
- EPR 0.5-0.8: Moderate. Half the compute is wasted.
- EPR < 0.5: Poor. Most instances are noise.
- **Current measurement: EPR = 0.55 (MODERATE).**

### p(1-p) Bernoulli Variance (from LILO, 2502.12272)
p = probability the best heuristic wins a given instance. RLVR gradient ∝ p(1-p).
- p ≈ 0.5: Maximum gradient. Best for learning.
- p ≈ 1.0: Zero gradient. One heuristic always wins.
- **Current: p ≈ 0.53 (geographic) → p(1-p) ≈ 0.25. GOOD for gradient, but J is too low.**

### The Core Insight (April 13 2026)
Every system that groks (AlphaZero, DeepSeek-R1, AZR) has a DETERMINISTIC verifier. J=1.0.
Stochastic verification → J≈0 → no learning signal → can't scale.

**Fix applied**: at coupling=1.0 with noise_reduction=1.0, the supplier model is fully deterministic:
- Participation: threshold-based (prob ≥ 0.5 → bid), no random sampling
- Cost: no noise term, pure function of geography
- Markup: already deterministic given supplier params

**After deterministic fix**:
- J = 0.533 (geographic vs random) — STRONG. 77% correct ranking.
- Determinism: confirmed. Same input → same output regardless of seed.
- Geographic wins 67%, equal_size 20%, random 10%, category 3%.

### What Scaling Requires (from AlphaZero/AZR/RLVE)
1. **Deterministic verification** ← DONE (noise_reduction=1.0)
2. **Procedural instance generation** ← DONE (TransportInstanceGenerator)
3. **Adaptive difficulty** ← NOT YET. Need to auto-increase coupling/n_items as agent masters current level.
4. **No plateau** ← Need RLVE-style EPR tracking + ACCEL-style regret-based curriculum

## Cross-Lot Coupling (v2 — Signal Over Noise)

At `coupling_strength=0`, "group by category" trivially dominates. At `coupling_strength>0`, six mechanisms create tension:

| Mechanism | Effect | Gated by coupling |
|-----------|--------|-------------------|
| Participation boost (+2.0 bias) | 73% bid rate vs 27% | Yes |
| Deterministic verification | J=0.533 (was 0.025). Same input→same output. | Yes (noise_reduction=1.0) |
| Capacity scarcity (1.5x) | Solver must spread work | Yes |
| Economy of scale (15%) | Bulk discount for large lots | Yes |
| Lot overhead (100/lot) | Penalizes many small lots | Yes |
| Geo proximity damping | Prevents single-item exploit | Always |

Volume tiers DISABLED (solver rebate bug: max_supplier_bid includes non-winning bids → negative costs).
Package bids remain but are secondary to capacity binding as coupling mechanism.

## Key Research Findings

### RLVR Environment Design (Literature Synthesis, 30+ papers)
- RLVR activates existing reasoning, doesn't create new (Invisible Leash, 2507.14843)
- p(1-p) principle: only train on instances where model succeeds 30-70% of the time (LILO, 2502.12272)
- Stochastic verifiers need J > 0 to learn. J ≤ 0 → anti-learning (RLVεR, 2601.04411)
- Adaptive difficulty via "effective prompt ratio" tracking (RLVE, 2511.07317)
- Process-level verification gates noisy rewards (Trade-R1, 2601.03948)
- Regret-based curriculum with level editing for emergent complexity (ACCEL, 2203.01302)
- Must SFT first to install domain prior (GURU, 2506.14965; Recon, 2506.00577)

### Procurement-Specific
- Lot design is a second-order effect (2-8% savings) after specification and PQQ
- Real coupling comes from binding capacity constraints, not packages
- The right litmus test: "Can a human expert predict which partition is better?"
- LLM-as-heuristic-designer (ReEvo) avoids the action space problem entirely

## Code Conventions

- **Immutability**: All data objects are frozen dataclasses. Never mutate.
- **No print()**: Use `logging` module.
- **Type annotations**: All function signatures typed. `mypy --strict` must pass.
- **Config-driven**: All parameters via Pydantic config. No hardcoded values.
- **Test-first**: Write failing test, then implement.
- **Diagnostic-first**: Measure J, EPR, p(1-p) before any training run.

## Plan

Plan: `../.claude/plans/imperative-finding-music.md` (coupling fix v2).
Product Intent: `docs/research/procurement-rl/PRODUCT_INTENT.md`.
