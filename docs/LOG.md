# Genealogy of the work

A traceable log of the thinking, design, and validation that produced this codebase.

---

## 1. Thesis

Procurement is usually framed as an *allocation* problem: given items and bids, who wins what? The harder, less-studied question is the *structuring* problem that comes first — given items, suppliers, and a behavioral model of how suppliers respond to lot composition, **how should the buyer partition the items into lots before opening bidding?**

Lot design is a sequential combinatorial decision that *selects which downstream auction gets run*:

- It changes which suppliers are eligible (capability gating).
- It changes which suppliers participate (entry costs, lot-size effects, geographic proximity).
- It changes which package bids become legal (cross-lot synergies).
- It changes which capacity constraints bind (volume tiering, supplier-level caps).

This is exactly the surface where a learned policy could plausibly help — the combinatorial structure is too rich for closed-form optimization, the search space is too large for brute force, and the downstream solve is well-conditioned. But to learn it cleanly, the environment must give an *exact* reward signal. A learned reward critic introduces the same reward-hacking failure modes the RLVR literature was built to escape.

> The gym hypothesis: **lot-structure optimization is a clean RLVR target** — a sequential combinatorial decision with an exact, programmatic reward (CP-SAT-solved buyer welfare) over a calibratable behavioral simulator.

This codebase is the **environment**, not the agent. The contribution is a verification harness that other people can drop trained policies into.

---

## 2. Design — the verification harness

Two design decisions are load-bearing.

### 2.1 The reward is solver output, not simulator output

A common shortcut in procurement-RL prototypes is to score a partition by a hand-coded heuristic — total bid sum, average markup, or a learned welfare estimate. We do not do this.

After supplier behavior is sampled, the environment hands the resulting bids, capacities, and package offers to **OR-Tools CP-SAT** as a set-packing ILP:

```
minimize   Σ_(s,k) bid[k][s] · x[s,k]  +  Σ_p price[p] · y[p]
subject to ∀k: Σ_s x[s,k] + Σ_{p∋k} y[p] = 1            (each lot covered exactly once)
           ∀s: Σ_k volume[k] · x[s,k] + Σ_{p:supplier(p)=s} pkg_volume[p] · y[p] ≤ capacity[s]
           x[s,k], y[p] ∈ {0, 1}
```

Volume-tier rebates are added as additional indicator-gated linear terms (see `_volume_tier_rebates` in `solver/ortools_solver.py`). The reward is the optimal objective value translated back to buyer welfare:

```
reward = Σ serviced_value − total_cost − unserviced_penalty − coupling_strength · lot_overhead_base · n_lots
```

CP-SAT either returns OPTIMAL with a certified objective, FEASIBLE with a finite optimality gap (which we expose in `info`), or INFEASIBLE (which we treat as a structured penalty proportional to lot count). At the K=20 lot bound and N=20 item default, every problem we've run solves to OPTIMAL in milliseconds.

This is what makes the environment RLVR-compatible: the reward is not approximated.

### 2.2 The `coupling_strength` knob — from stochastic to deterministic verification

Most procurement simulators are point-in-design-space: one set of behavioral assumptions, one regime. We exposed a single scalar `coupling_strength ∈ [0, 1]` that interpolates the regime continuously:

| Knob value     | Regime                                                         |
|----------------|----------------------------------------------------------------|
| 0.0            | Lots independent. No package bids. No volume tiers. Stochastic supplier participation and markup. |
| 0.5            | Mixed regime. Some package bids generated. Capacity tightens proportionally. |
| 1.0 + `noise_reduction=1.0` | Deterministic verification. Participation collapses to a threshold (no random draw). Markup and cost noise zeroed. Same partition → same reward, exactly. |

The deterministic regime is what makes this gym usable for **verifier-based RL** in the AlphaProof / DeepSeek-R1 sense: the agent's reward is reproducible across rollouts on the same instance. The stochastic regime is what makes it a useful **noisy-reward RL benchmark**: variance across simulator seeds is real, agents must average over draws.

Both regimes are first-class. The agent doesn't know which one it's in — the observation space is identical.

### 2.3 Immutability throughout

`LotPartition` is a frozen dataclass. `assign(item_idx, lot_id)` returns a new partition. Items, suppliers, instances are frozen. No in-place mutation anywhere in the env step path. This was a deliberate choice to make rollouts trivially parallelizable and to avoid the spooky-state bugs that bite gym implementations as they grow.

### 2.4 Three-layer behavioral model

Cost, participation, and markup are decomposed cleanly:

- **Cost** (`suppliers/cost.py`) — additive per-item distance × supplier rate, plus fixed cost per lot, plus geographic synergy (negative — closer items, lower cost), plus per-lot Gaussian noise. Economy-of-scale gated by `coupling_strength`.
- **Participation** (`suppliers/participation.py`) — logistic in `(capability_match, log lot_size, n_eligible, geographic_proximity, supplier_bias)`. The `n_eligible` term gives competition self-attenuation: lots with many eligible suppliers see lower individual entry rates.
- **Markup** (`suppliers/markup.py`) — `α · (1/n_competitors)^β`. Competition compresses markup. α and β are calibratable to GPV (Guerre, Perrigne, Vuong 2000) markup estimates.

Splitting these into three modules means each is independently testable, and a future calibration to a real procurement dataset can target the layer with the largest residual without touching the others.

### 2.5 Cross-lot package bids

When `coupling_strength > 0`, `suppliers/packages.py` enumerates per-supplier subsets of lots they bid on (size 2..`max_package_size`) and offers a synergy-discounted bundle price for each subset that clears the synergy threshold. The discount is deterministic given synergy and `coupling_strength`. CP-SAT handles bundle awards via a `y[p]` indicator; the set-packing constraint guarantees no double-coverage.

This is the cross-lot coupling that makes the lot-design problem genuinely combinatorial. Without it, the WDP factorizes per lot and lot design is trivial.

---

## 3. Validation record

The validation discipline here is **environment-quality**, not agent-quality. The goal of this section is to document what the harness has been verified to produce, not to claim a learned policy beats a baseline (that's downstream work — see §5).

### 3.1 Static checks

- **mypy strict.** The `[tool.mypy] strict = true` flag is on, with overrides only for `ortools.*` and `scipy.*` (which lack stubs). The codebase typechecks clean under `mypy >= 1.20`.
- **ruff.** Line-length 100, py312 target. Clean.
- **Frozen-dataclass invariants.** Every domain object (`Item`, `Supplier`, `PackageBid`, `ProcurementInstance`, `LotPartition`) is `@dataclass(frozen=True)`. `assign()` raises on out-of-range indices, double-assignment, or negative lot IDs.

### 3.2 Solver validation

The CP-SAT formulation has been spot-checked end-to-end:

- **Determinism at coupling=1.** With `coupling_strength=1.0, noise_reduction=1.0`, repeated `evaluate_partition(...)` on the same `(partition, instance, seed)` returns bit-identical rewards. This is what `evaluation.evaluate_multi_seed` exercises.
- **OPTIMAL status under default config.** At N=20 items, K_MAX=20 lots, the default 60-second timeout is never approached on Apple Silicon dev hardware — solves are millisecond-scale.
- **Package-bid double-coverage.** The set-packing constraint `add_exactly_one(sources)` is asserted at award-extraction time (line 159, `solver/ortools_solver.py`): a lot covered by a package cannot also be covered by a single bid.
- **Capacity feasibility.** The `add(sum(volume_terms) ≤ capacity)` constraint uses integer scaling at SCALE=1000 to avoid float-comparison artifacts.

### 3.3 Five baselines as an acceptance suite

`baselines/strategies.py` ships five lot-partition strategies:

1. `random_partition(n_lots)` — uniform random assignment with all lots used.
2. `single_item_partition()` — one lot per item (worst case for lot overhead, best case for supplier specificity).
3. `category_partition()` — group by item category.
4. `geographic_partition(n_lots)` — k-means on item (x, y).
5. `equal_size_partition(n_lots)` — round-robin into equal-sized lots.

These are not just baselines for an agent to beat — they are **regression tests on the reward landscape**. Across instance seeds:

- Single-item partitions should incur the highest lot overhead and (under `coupling_strength > 0`) lose the most volume-tier rebate.
- Category partitions should be feasible (no capability mismatch) and capture eligibility cleanly.
- Geographic partitions should outperform random under high synergy weight.
- Random partitions should produce a wide reward distribution — useful as a noise baseline for SNR estimation.

A reward landscape that doesn't separate these strategies on average across seeds would indicate a bug in the simulator, the solver, or the reward composition. The current parameter defaults produce the expected separation.

### 3.4 Verifier round-trip

`verifier/serializer.py` + `verifier/parser.py` form a closed loop:

```
ProcurementInstance ─serialize_instance─▶ prompt_text
LotPartition       ─serialize_partition─▶ "<partition>[...]</partition>"
"<partition>[...]</partition>" ─parse_partition─▶ LotPartition
```

The parser has three-stage fallback (tag → JSON → bracketed-int) to handle realistic LLM completion noise: trailing commentary, malformed JSON, missing tags. `validate_partition` rejects length mismatches, negative lot IDs, and incomplete partitions before the partition reaches the solver. Parse-or-validate failure routes to a per-item penalty in `ProcurementVerifier.reward_fn`, never to the solver.

### 3.5 Reproducibility surface

Every randomness source threads through an explicit seed:

- `EnvConfig.seed` → `gymnasium.Env.reset(seed=...)` → `self.np_random` (Gymnasium's standard PRNG).
- The env derives `instance_seed` from `np_random` for the instance generator, and `sim_seed` for the supplier simulator. No global random state is ever consulted.
- `cp_model.CpSolver().parameters.random_seed = 0` for deterministic CP-SAT branching.

This means a `(env_seed, partition)` pair fully determines the reward at `coupling_strength=1.0, noise_reduction=1.0`. Multi-seed evaluation (`evaluate_multi_seed`) exists for the stochastic regime.

---

## 4. What the environment does *not* commit to

These are deliberate non-commitments — the gym leaves them to the user.

- **No fixed instance distribution.** `TransportInstanceGenerator` is one generator. Procurement domains beyond transport (services, IT, capex) would need their own generator; the `ProcurementInstance` interface is generic.
- **No fixed reward shaping schedule.** The `unserviced_penalty = 10 · mean_item_value` and `lot_overhead_base = 100.0 · coupling_strength` defaults are starting points. Both are exposed via config; both should be tuned per use case.
- **No fixed item presentation order.** `ascending_qualified` (default), `random`, and `by_category` are supported. The choice is policy-relevant and should be ablated, not assumed.
- **No claim about which RL algorithm wins.** SFT on search-optimized partitions, GRPO with the verifier reward, PPO on the env step interface, or AlphaZero-style MCTS all type-check against this gym. We have not run a head-to-head.

---

## 5. What's open

The environment is production-ready. The downstream questions are about which agents to put inside it.

### 5.1 Algorithms to test against the gym

- **GRPO (TRL `GRPOTrainer`)** — `scripts/train_grpo.py` is wired up. Uses `ProcurementVerifier` as the reward function; the verifier scores complete partitions, so this trains on full-trajectory rollouts. Open question: does GRPO with sparse terminal reward learn the lot-structure decision, or does it need credit-assignment scaffolding?
- **SFT bootstrap then GRPO** — `scripts/train_sft.py` + `scripts/train_grpo.py`. Search-optimized SFT data ships with the repo (`sft_data.jsonl`, ~1.6 MB). The standard RLVR recipe.
- **PPO / TRPO on the env step interface** — `Procurement-v0` is registered with Gymnasium. Sequential per-step rewards are zero, terminal reward is the welfare. Open question: does dense reward shaping (per-step "potential" reward = current-partition welfare estimate) improve sample efficiency, or does it leak information from the simulator?
- **AlphaZero-style MCTS** — partition assembly is a tree where each node is a partial partition. The reward is exact (CP-SAT) but expensive. Open question: how many simulator+solve calls per move are tractable, and does MCTS beat the LLM-based approaches at fixed compute?
- **Constructive heuristics + learned residual** — start from the geographic or category baseline, learn a policy that proposes local moves (split-lot, merge-lot, swap-item). This is the OR-style route.

### 5.2 Calibration to real data

The three-layer model is parametric for a reason. Calibrating `markup_alpha`, `markup_beta`, and the participation logistic weights to a real procurement dataset (e.g., a freight-procurement RFP outcome record) would make the gym's behavioral model empirically grounded. The hooks are there; the data is the rate-limiter.

### 5.3 Adversarial robustness suite

Adversarial questions worth running formally before any agent claim:

- **Reward-hacking probes.** Does the agent learn to open spurious lots that game lot overhead, or to under-cluster items to suppress competition? The `unserviced_penalty / lot_overhead_base / coupling_strength` triple is the relevant param surface.
- **Distribution shift across `coupling_strength`.** A policy trained at `coupling_strength=0` (no package bids) likely degrades sharply at `coupling_strength=1`. By how much?
- **Capacity-binding regimes.** As `capacity_scarcity` rises, the WDP becomes infeasibility-prone. Does the agent learn to leave headroom, or does it ride the cliff?

### 5.4 Scaling

K_MAX=20 lots and N=20 items is the development default. The CP-SAT formulation handles 100s of lots in single-digit seconds; the env observation space encodes K_MAX statically. Scaling N and K is a config change plus an observation reshape, not a redesign.

---

## 6. Cited literature

The work in this repo builds on:

- **OR-Tools CP-SAT**: Perron, L. & Furnon, V. *OR-Tools v9.* Google. [github.com/google/or-tools](https://github.com/google/or-tools).
- **Combinatorial auctions / WDP**: de Vries, S. & Vohra, R. (2003). *Combinatorial auctions: A survey.* INFORMS Journal on Computing 15(3): 284–309.
- **CATS instance generators**: Leyton-Brown, K., Pearson, M. & Shoham, Y. (2000). *Towards a universal test suite for combinatorial auction algorithms.* EC '00.
- **First-price auction calibration**: Guerre, E., Perrigne, I. & Vuong, Q. (2000). *Optimal nonparametric estimation of first-price auctions.* Econometrica 68(3): 525–574.
- **RLVR / verifier-based RL**: DeepSeek-AI (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* arXiv:[2501.12948](https://arxiv.org/abs/2501.12948).
- **GRPO**: Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* arXiv:[2402.03300](https://arxiv.org/abs/2402.03300).
- **TRL**: von Werra, L. et al. *TRL: Transformer Reinforcement Learning.* [github.com/huggingface/trl](https://github.com/huggingface/trl).
- **Gymnasium**: Towers, M. et al. (2023). *Gymnasium.* [github.com/Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium).

Adjacent / contextual:

- **AlphaProof** — DeepMind (2024). *AI achieves silver-medal standard solving International Mathematical Olympiad problems.* (verifier-driven RL, formal-proof setting).
- **rl4co** — Berto, F. et al. (2024). *RL4CO: An Extensive Reinforcement Learning for Combinatorial Optimization Benchmark.* arXiv:[2306.17100](https://arxiv.org/abs/2306.17100).
- **CS224R**: Stanford Deep Reinforcement Learning (Spring 2026). The course this environment was first scoped for.
