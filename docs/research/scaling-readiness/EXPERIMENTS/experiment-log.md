# Experiment Log

## E-1: High-coupling determinism and sanity
- **Date**: 2026-04-15
- **Conjecture**: C-1
- **Gate**: RUN
- **Hypothesis**: After the simulator fix, rewards at `coupling=1.0` should be deterministic and total cost should never be negative.
- **Method**: `uv run python scripts/verify_scaling_readiness.py`
- **Result**: `max_reward_diff_across_seeds=0.0`, `negative_total_cost_evals=0/50`
- **Analysis**: Correctness gate passes.
- **Decision**: DOUBLE DOWN

## E-2: Strategy signal and package activity
- **Date**: 2026-04-15
- **Conjecture**: C-1
- **Gate**: RUN
- **Hypothesis**: If the mechanism is non-degenerate, no trivial baseline should dominate completely and packages should matter in a meaningful fraction of evaluations.
- **Method**: Same audit, 30 instances at `coupling=1.0`
- **Result**: `category` wins `22/30`; package awards in `60.7%` of evaluations; mean relative top1-top2 gap `3.65%`
- **Analysis**: Signal exists and the coupling mechanism is active.
- **Decision**: DOUBLE DOWN

## E-3: Runtime curve under fragmentation
- **Date**: 2026-04-15
- **Conjecture**: C-1
- **Gate**: RUN
- **Hypothesis**: If verifier runtime remains bounded, high-lot outputs should not explode package counts or solver wall time.
- **Method**: Same audit, `n_lots ∈ {2,3,5,10,20}`
- **Result**: Initial run failed (`p95_wall_s=3.73s` at 20 lots). After adding `max_packages_per_supplier`, rerun passed with `p95_wall_s=0.093s` and max package bids `640`.
- **Analysis**: Runtime was a real blocker, but the bounded package language removed it.
- **Decision**: DOUBLE DOWN

## E-4: Search uplift probe
- **Date**: 2026-04-15
- **Conjecture**: C-1
- **Gate**: RUN
- **Hypothesis**: If the environment is worth scaling, a cheap local search should still find gains beyond heuristics.
- **Method**: Built-in search probe in `scripts/verify_scaling_readiness.py`
- **Result**: Search beats the best non-single-item heuristic on `87.5%` of sampled instances and by `>2%` on `25%`.
- **Analysis**: The reward landscape is not purely trivial heuristic imitation.
- **Decision**: DOUBLE DOWN

## E-5: Production bridge existence check
- **Date**: 2026-04-15
- **Conjecture**: C-1
- **Gate**: RUN
- **Hypothesis**: If production readiness is plausible, the calibration and retrospective-eval artifacts named in the spec should exist.
- **Method**: File-system check inside `scripts/verify_scaling_readiness.py`
- **Result**: `suppliers/gpv.py`, `instances/calibrated.py`, and `scripts/calibrate.py` are missing
- **Analysis**: Production readiness fails independently of simulator quality.
- **Decision**: ABANDON production-readiness claim for the current repo state
