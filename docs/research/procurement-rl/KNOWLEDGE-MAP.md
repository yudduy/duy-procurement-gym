# Knowledge Map: AI/RL for Procurement

## Core Papers

| Paper | Year | Key Contribution | Relevance |
|-------|------|-----------------|-----------|
| RegretNet (Dutting et al.) arXiv:1706.03459 | 2019 | Neural auction mechanism design via soft IC constraint | Foundation of differentiable economics; seller-side only |
| Gasse et al. arXiv:1906.01629 | 2019 | GCN for B&B variable selection in MILP | Foundation of learn-to-branch; directly applicable to WDP |
| GNN for Multi-Unit CA arXiv:2009.13697 | 2020 | GNN predicts WDP winners, near-optimal with speedup | Only neural WDP solver paper |
| Auction as Two-Player Game arXiv:2006.05684 | 2021 | Reformulates IC as game, stabilizes RegretNet training | Key RegretNet improvement |
| GNN for Energy CAs arXiv:2307.13470 | 2023 | Heterogeneous tripartite graph for WDP, 76% F1, <5% gap | Best applied GNN-WDP result |
| PlanB&B arXiv:2511.09219 | 2025 | MCTS + learned world model for B&B = "AlphaZero for B&B" | Most direct AlphaZero-to-procurement transfer |
| CANet/CAFormer arXiv:2501.19219 | 2025 | RegretNet extended to combinatorial auctions | First neural mechanism design for CAs |
| Principal-Agent RL arXiv:2407.18074 | 2024 | RL discovers contracts in MDPs, Parkes group | Closest to "RL discovers procurement rules" |
| DeepMind Sequential CAs arXiv:2407.08022 | 2024 | RL for sequential auction mechanism design | RL for mechanism rules (not bids) |
| MARL for Iterative CAs arXiv:2402.19420 | 2024 | MARL bidding strategies in spectrum auctions, UBC | Multi-agent auction RL |
| MCTS for Spectrum Auctions arXiv:2307.11428 | 2023 | MCTS for bidding in SAA | Closest to "AlphaZero for auctions" (bidder side) |
| You et al. IC Underestimation arXiv:2601.13489 | 2026 | RegretNet true regret 70x higher than reported | Breaks field's flagship metric |
| GemNet arXiv:2406.07428 | 2024 | Menu-based, exactly strategy-proof multi-bidder | Exact IC by construction |
| LLM Mechanism Design arXiv:2502.12203 | 2025 | LLMs evolve interpretable auction code, rediscovers Myerson | Alternative to neural approach |
| Deep Incentive Design arXiv:2603.07705 | 2026 | Differentiable equilibrium blocks for contract design | Generalizes across game sizes |

## Benchmarks & Environments

| Environment | Type | Auction/Procurement? | API | Status |
|-------------|------|---------------------|-----|--------|
| OR-Gym | Single-agent CO | No (inventory/knapsack) | Gym (old) | Unmaintained |
| RL4CO | Single-agent CO | No (routing/scheduling) | TorchRL | Active, 27 envs |
| Jumanji | Single-agent CO | No (routing/packing) | JAX custom | Active |
| Ecole | ML-inside-MILP | Has CA instances for B&B | Gym-like | Active |
| OpenSpiel Clock Auction | Multi-agent auction | Forward spectrum auction | OpenSpiel | Research fork |
| SafeOR-Gym | Constrained CO | No (energy/supply chain) | Gymnasium+CMDP | 2025 |
| EconEvals | LLM economic eval | Has "Procurement" task | Benchmark only | 2025 |
| ProcureGym | Multi-agent procurement | Pharma NVBP bidding | Custom | 2026, narrow |
| **GAP: Procurement env** | **Multi-agent reverse CA** | **MISSING** | **PettingZoo?** | **Does not exist** |

## Key Surveys

| Survey | Coverage |
|--------|----------|
| Bengio et al. arXiv:1811.06128 | ML for CO taxonomy (config, alongside, as optimization) |
| ML for MIP arXiv:2203.02878 | Learn to branch, cut, select nodes |
| An 2026 arXiv:2602.03003 | Open problems in automated mechanism design |
| NCO Tutorial arXiv:2505.xxxxx (ScienceDirect 2025) | Neural combinatorial optimization comprehensive |
