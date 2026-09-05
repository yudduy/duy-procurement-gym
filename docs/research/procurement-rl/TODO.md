# TODO: Procurement/Supply Chain RL Research

## Goal
Discover something fundamental at the intersection of AI/RL and procurement that could advance the field. The AlphaZero analogy: find the right game formulation where self-play + verifiable outcomes enables superhuman procurement agents.

## Sessions
- **Session 1** (2026-04-12): Initial deep landscape + conjecture formation

## Current Phase
COMPLETE — Converged on ProcurementGym spec. Ready for implementation.

## Sub-questions (iteration 1)
1. What is the exact mathematical structure of Winner Determination Problem (WDP) in combinatorial procurement auctions?
2. Has anyone applied neural/RL methods to WDP or combinatorial auctions?
3. What verifiable environments exist for combinatorial optimization + RL (OR-Gym, etc.)?
4. What does the self-play / AlphaZero analogy actually look like for procurement?
5. What's the connection between mechanism design (incentive compatibility) and RL in auctions?
6. What are the hardest open problems in procurement optimization that resist classical OR methods?

## Decisions
- [2026-04-12] Starting with 6 parallel sub-agent investigations
- [2026-04-12] Focusing on the WDP/combinatorial auction angle as most promising for verifiable environment
- [2026-04-12] CPLEX solves WDP in seconds → L2 is NOT the bottleneck → pivot to L1 (mechanism design)
- [2026-04-12] Adversarial review identified 3 fatal attacks: sim-to-real circularity, AMD prior art, weak grokking
- [2026-04-12] Revised conjecture: LOT STRUCTURE DESIGN is the genuinely novel lever (upstream of AMD)
- [2026-04-12] Gemini deliberation confirms B+ direction (RL for L1 with CPLEX inner solver)
- [2026-04-12] Bilevel check confirms: RL outer + exact MILP inner + procurement params = novel COMBINATION
- [2026-04-12] Launched targeted checks: lot structure literature + structural estimation feasibility

## Key Attacks to Resolve
1. **Sim-to-real circularity**: Need structural estimation (GPV 2000) from bid data to build realistic simulator
2. **AMD prior art**: Lot structure design is UPSTREAM of AMD — AMD optimizes within mechanism, we optimize the problem formulation
3. **Grokking weak**: Don't claim general transfer. Claim within-category transfer (e.g., across transport lane configs). Invariant = competition structure
4. **Action space sweet spot**: Lot structure for N items has Bell(N) possible partitions. For N=50, Bell(50)~10^47. Too large to enumerate, structured for RL

## Resolved Concerns
- [RESOLVED] Lot structure design confirmed unstudied for N>2 (Grimm 2006 is qualitative only, zero ML/RL)
- [RESOLVED] Structural estimation feasible for single-lot (GPV 2000), scope to transport/commodities
- [RESOLVED] Bilevel novelty confirmed: RL outer + MILP inner + procurement params = novel combination
- [PARTIALLY RESOLVED] Sim-to-real: GPV bridge works but endogenous participation creates circularity. Fix: participation-aware model + domain randomization.

## Remaining Concerns
- Bell(N) overclaims: with practical constraints (lot size [2,8], categorical coherence), effective space ~10^8-10^12
- Improvement hypothesis needs anchoring: 2-8% over geographic clustering (1-2% is big in mature categories)
- Must show RL recovers known optima in toy 2-item cases to validate environment

## Session Walkthrough

### Research Process (10 sub-agents, 2 adversarial rounds, 1 deliberation)

**Iteration 1: Landscape**
- 6 parallel agents investigated: WDP math, neural auctions, OR environments, AlphaZero analogy, mechanism design + RL, hard problems
- Key finding: CPLEX solves WDP in seconds → L2 is NOT the bottleneck
- Three-layer decomposition: L1 (mechanism design) > L2 (WDP) > L3 (strategic interaction)
- Initial conjecture: RL for mechanism design with CPLEX inner solver

**Iteration 1.5: Adversarial Review**
- 1 adversarial agent + 1 multi-model deliberation + 1 bilevel literature check
- 3 fatal attacks identified: sim-to-real circularity, AMD prior art, weak grokking
- Key pivot: lot structure design (upstream of AMD) is the genuinely novel lever

**Iteration 2: Targeted Investigation**
- 2 agents: lot structure literature + structural estimation feasibility
- Confirmed: lot design unsolved for N>2, zero ML/RL, Bell(N) partitions
- GPV structural estimation bridges sim-to-real for simple formats

**Iteration 2.5: Final Review**
- Round 2 adversarial review identified remaining fix: endogenous participation model
- Venue assessment: EC (Economics and Computation), not NeurIPS
- Expected EC score: 6-7 (likely accept) with participation fix + toy validation + retrospective

**Implementation Spec**
- 2 agents: data sources + gym architecture
- Architecture: sequential assignment, 3-layer supplier model, OR-Tools solver, Gymnasium API
- Data: OpenTender/DIGIWHIST + ProZorro (lot-bid-award triangle)
- Build order: 5 phases, 8 weeks

### Modified Files
- `docs/research/procurement-rl/UNDERSTANDING.md` — main research artifact (2 iterations)
- `docs/research/procurement-rl/KNOWLEDGE-MAP.md` — bibliography (30+ papers)
- `docs/research/procurement-rl/TODO.md` — this file
- `docs/research/procurement-rl/PROCUREMENT_GYM_SPEC.md` — implementation spec

### Next Steps
1. Set up Python project scaffold with Gymnasium dependency
2. Implement Phase 1 skeleton (types, partition, transport generator, OR-Tools solver, env)
3. Download OpenTender data for 2-3 EU countries (start with Germany or Netherlands)
4. Implement baselines (geographic clustering, random, single-item)
5. Train PPO agent and compare to baselines
