# Understanding: AI/RL for Procurement (Iteration 2 — Post-Adversarial)

## The Primitives

Procurement optimization decomposes into **three nested decision layers**:

| Layer | Problem | Structure | Solved? |
|-------|---------|-----------|---------|
| L1: Problem Formulation | Partition items into lots, choose scoring formula, set constraints | Combinatorial design (Bell(N) partitions) | **NO — unsolved for N>2** |
| L2: Winner Determination | Given bids + constraints, find optimal allocation | NP-hard ILP (weighted set packing) | **YES — CPLEX in seconds for typical instances** |
| L3: Strategic Interaction | Suppliers choosing what to bid | Multi-agent, imperfect information | Partially (single-item: Myerson; combinatorial: open) |

**CRITICAL FINDING: The bottleneck is L1, not L2.** CPLEX solves typical procurement WDP (50-500 lots, 5-50 suppliers) in seconds. Neural WDP solvers don't add value at this scale. The genuine unsolved problem is UPSTREAM: how to partition N items into lots such that the resulting auction maximizes buyer welfare.

**Mathematical structure of L1:** The lot partition problem has Bell(N) possible partitions (Bell(50) ~ 10^47). Each partition determines: (a) which suppliers can bid on which lots (capability filtering), (b) competition intensity per lot, (c) cost synergies within lots, (d) WDP structure and difficulty. The relationship between partition and buyer welfare goes through the entire auction pipeline: partition → supplier bidding strategy → WDP allocation → welfare.

## The Assumption Chain

| # | Assumption | Status | Evidence |
|---|-----------|--------|----------|
| 1 | L1 (lot design) has more value than L2 (WDP solving) | **Supported** | CPLEX solves L2 in seconds; Grimm et al. (2006) + practitioners confirm lot design is the binding constraint |
| 2 | Lot structure design is genuinely unsolved | **Confirmed** | Only 2-item binary results exist (Subramaniam-Venkatesh 2009, Maurer-Herz 2014). No algorithm for N>2. Zero ML/RL applied. |
| 3 | Bell(N) is the right scale for RL | **Plausible** | Too large to enumerate (10^47 for N=50), but structured by item similarity → RL should find structure |
| 4 | Cost distributions can be estimated from bid data | **Partially true** | GPV (2000) works for single-lot procurement (well-founded). Combinatorial estimation remains hard. |
| 5 | Competition structure transfers within categories | **Assumed, needs testing** | The principle "more bidders per lot → lower prices" is universal, but the optimal lot structure depends on category-specific supplier markets |
| 6 | RL outer + MILP inner is novel | **Confirmed** | Bilevel check: closest = Stackelberg POMDP (continuous followers) + Neur2BiLO (supervised, not RL). The specific combination doesn't exist. |

## What the Field Gets Right

1. **CPLEX/Gurobi for WDP** — commercial solvers handle real procurement instances efficiently. No need to replace them.
2. **Grimm et al. (2006)** — correctly identified lot structure as the key buyer decision in procurement design.
3. **GPV (2000)** — structural estimation from bid data provides the sim-to-real bridge for simple formats.
4. **RegretNet/differentiable economics** — neural mechanism design IS possible, just hasn't been applied to procurement or to lot structure.
5. **PlanB&B (2025)** — learned evaluation + search for B&B is the right architecture for large-scale L2 problems.

## What the Field Gets Wrong (or Doesn't Question)

1. **Lot structure is treated as exogenous.** Both the economics literature (takes lots as given, optimizes mechanism) and the OR literature (takes bids as given, solves WDP) ignore the most impactful design variable. The joint optimization of lot structure + mechanism design is untouched.

2. **"Bundling in auctions" is studied seller-side.** The economics literature on bundling (Palfrey 1983, Subramaniam-Venkatesh 2009) focuses on seller revenue maximization. Procurement (buyer-side, cost minimization) has fundamentally different IR constraints and competitive dynamics.

3. **Combinatorial auctions bypass the lot problem.** Ausubel & Milgrom's solution — let bidders express preferences via package bids — sidesteps lot design but creates an intractable preference elicitation problem. For large N, the cognitive burden on bidders makes this impractical.

4. **The NCO/neural CO community ignores set packing.** WDP ≡ weighted set packing is one of Karp's 21 NP-complete problems with trillion-dollar industrial relevance, yet it's absent from all NCO benchmarks (TSP, VRP, bin packing dominate).

5. **No public procurement RL environment exists.** OR-Gym has inventory. RL4CO has routing. Nobody has built a Gymnasium/PettingZoo environment for procurement mechanism design.

## Current Model: The Revised Conjecture

### Statement

**For procurement categories where per-item cost structures are estimable (transportation, commodities, standardized services), an RL agent that learns lot structure designs can outperform human heuristics by optimizing the competition-volume tradeoff — verified via GPV-calibrated supplier simulation and CPLEX-exact allocation.**

### The Verifiable Environment Architecture

```
STATE: Market conditions (item features, supplier capabilities, historical prices)
  ↓
RL AGENT: Outputs lot partition π = {L₁, L₂, ..., Lₖ} of N items
  ↓
SUPPLIER SIMULATION: Each supplier s bids on feasible lots
  - Cost model: c_s(L) = Σᵢ∈L cost_s(i) + synergy(L) + noise
  - Bid strategy: b_s(L) = c_s(L) × markup (calibrated from GPV on historical data)
  ↓
CPLEX: Solves WDP exactly → allocation A*
  ↓
REWARD: BuyerWelfare(A*) = Σᵢ value(i) - Σₛ payment(s)  [verifiable scalar]
  ↓
RL UPDATE: PPO/SAC on the mechanism design policy
```

**Why this is verifiable:**
- Cost model calibrated from real bid data via GPV structural estimation
- WDP solution is provably optimal (CPLEX with optimality certificate)
- Buyer welfare is a deterministic function of the allocation
- Sim-to-real validation: train in simulation, evaluate on held-out historical events

### Why Lot Design is the Right "Game" for RL

1. **Action space sweet spot**: Bell(N) for N=50 is ~10^47. Too large to enumerate. Too structured to be intractable (nearby partitions have correlated outcomes because similar items have similar supplier markets).

2. **Verifiable reward**: Unlike general mechanism design (where optimal depends on unknown equilibria), lot design welfare is computable given a cost model + CPLEX.

3. **Structural invariance**: The competition-volume tradeoff IS a transferable principle:
   - Items with few qualified suppliers → bundle into larger lots (create "package deal" competition)  
   - Items with many qualified suppliers → keep separate (maximize direct competition)
   - Items with cost complementarities → bundle (capture synergies)
   - The RL agent should discover these principles and optimize the tradeoff

4. **Practical impact**: Keelvar, the market leader in sourcing optimization, explicitly provides lot structuring tools. This is a multi-billion dollar industrial problem.

### How This Differs from Prior Art

| Prior Work | What it does | How we differ |
|-----------|-------------|---------------|
| Conitzer-Sandholm AMD (2002) | Optimizes mechanism parameters (allocation + payment rules) | We optimize the PROBLEM FORMULATION (lot structure) — upstream |
| RegretNet (2019) | Neural auction mechanism design | We use CPLEX for allocation; optimize lot design, not allocation rules |
| Stackelberg POMDP (2022) | RL for economic rule design with learning followers | Our inner problem is combinatorial (MILP), theirs is continuous |
| Grimm et al. (2006) | Qualitative framework for lot structure trade-offs | We provide the first computational approach |
| PlanB&B (2025) | Neural-guided B&B for MILP solving | We use CPLEX (already solved); focus on problem formulation |

### What "Grokking" Means Here (scoped claim)

NOT: "The model learns universal procurement principles."
YES: "Within a procurement category (e.g., transportation), the model learns the mapping from (item features × supplier market structure) → optimal lot partition, and this mapping has learnable regularities that transfer across instances within the category."

The invariant is the **competition function**: how lot structure maps to effective competition (number of qualified bidders per lot weighted by their cost competitiveness). This function has category-specific parameters but a universal structure.

## Weakest Point (Post Round-2 Review)

**GPV circularity under endogenous participation.** The lot structure itself changes which suppliers enter, which changes the cost distribution GPV estimates, which changes the simulated bids. The reward signal depends on a distribution the agent's actions perturb. This is NOT the sim-to-real gap (already addressed by scoping to observable-cost categories) — it's an internal consistency problem.

**Resolution**: Model supplier participation as endogenous:
1. Participation model: P(supplier s enters lot L) = f(capability_match, lot_size, expected_competition)
2. Train against a DISTRIBUTION of GPV parameters (robustness/domain randomization)
3. Ablation: run trained policy against ±20% shifted GPV parameters — if robust, the circularity is manageable

**Additional calibration from Round 2 review:**
- Bell(N) overclaims — with practical constraints (lot size ∈ [2,8], categorical coherence), effective space is ~10^8-10^12. Still RL-appropriate but be honest.
- In mature procurement categories, 1-2% cost reduction is a big deal. Hypothesis should be 2-8% over geographic clustering, not 5-15%.
- Must show RL recovers known theoretical optima in toy settings (2-item, symmetric suppliers) before claiming it discovers new structure.
- EC (Economics and Computation) is the right venue, not NeurIPS. The contribution is problem formulation + environment, not ML novelty.

## Concrete Experiment Design

**Domain**: Transportation procurement (freight lanes)
- Items = origin-destination lanes (N=20-100)
- Suppliers = carriers (M=10-30)
- Cost structure: distance + fuel + equipment + market premium (observable from BLS indices)

**Environment**: ProcurementGym
- Agent partitions N lanes into K lots (K is chosen by agent)
- Carriers bid on feasible lots (capability-filtered, GPV-calibrated markup)
- CPLEX solves WDP → allocation → reward

**Baselines**: Random, single-item, geographic clustering, equal-sized, exhaustive (small N)

**Hypothesis (pre-registered)**: RL discovers that lanes with few qualified carriers should be bundled with high-competition lanes (cross-subsidy effect), while lanes with many carriers should be kept separate. Expected improvement: 2-8% total award cost reduction over geographic clustering baseline (1-2% is significant in mature categories).

**Validation experiments (from Round 2 review):**
1. Toy case: 2-item, symmetric suppliers — RL must recover known theoretical optimum
2. GPV robustness: train policy, evaluate with ±20% shifted cost parameters
3. Retrospective: apply learned lot structure to one real historical tender (EU transport data)

**Key metric**: Total award cost reduction + competition per lot + optimality gap vs. exhaustive (small N)

## Venue Assessment

**EC (Economics and Computation) is the right venue.** EC reviewers understand procurement mechanism design, will engage with the GPV circularity constructively, and value problem formulation contributions. NeurIPS would penalize the straightforward RL architecture (the hard part is the environment, not the agent). A strong EC result will be read by the right audience.
