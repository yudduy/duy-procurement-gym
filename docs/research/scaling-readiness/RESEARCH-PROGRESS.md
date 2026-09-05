# Research: Is The Procurement RL Substrate Ready To Scale?
> Status: COMPLETE
> Active Conjecture: C-1 | Confidence: 92
> Iteration: 1/1 | Last updated: 2026-04-15

## The Question
The real question is not whether the code runs. It is whether the verifier and simulator are now trustworthy and operationally tractable enough to scale RL training, and whether that would say anything about real procurement decisions.

## Current Understanding
The corrected oracle is now economically coherent at `coupling=1.0`: rewards are deterministic across seeds, no negative total-cost cases were found, and package awards are materially active. A bounded per-supplier package language removed the verifier runtime explosion without killing the coupling mechanism. The substrate is now ready for large-scale simulated RL training. The remaining blocker is the missing sim-to-real bridge promised in the spec.

## Key Results
### Result 1: Oracle correctness recovered
- **Experiment**: `uv run python scripts/verify_scaling_readiness.py`
- **Expected**: If the eligibility and markup fixes were first-order, high-coupling rewards should become deterministic and economically sane.
- **Observed**: `max_reward_diff_across_seeds=0.0`, `negative_total_cost_evals=0/50`.
- **Implication**: The substrate is no longer failing for the original reason. Correctness is now strong enough to audit seriously.

### Result 2: Reward signal exists and is not degenerate
- **Experiment**: Same audit, 30-instance baseline sweep at `coupling=1.0`.
- **Observed**: `category` wins `22/30`, `single_item` wins `0/30`, package awards occur in `60.7%` of evaluations, and mean top1-top2 relative gap is `3.82%`.
- **Implication**: There is learnable structure. The verifier is not degenerate into “single-item always wins.”

### Result 3: Runtime blocker was removed
- **Experiment**: Same audit after adding `max_packages_per_supplier`.
- **Observed**: p95 wall time is `0.0024s` at 3 lots, `0.023s` at 10 lots, and `0.093s` at 20 lots. Max package bids dropped to `640`.
- **Implication**: The verifier is now operationally tractable for large-scale simulated RL training.

### Result 4: Search still finds upside beyond heuristics
- **Experiment**: Lightweight local-search probe built into `scripts/verify_scaling_readiness.py`
- **Observed**: Search beats the best non-single-item heuristic on `87.5%` of sampled instances and beats it by `>2%` on `25%`. Median absolute gain is `206`.
- **Implication**: The environment is not trivialized into a single static baseline. There is still optimization headroom worth training on.

### Result 5: Production bridge is absent
- **Experiment**: File-system audit in `scripts/verify_scaling_readiness.py`.
- **Observed**: `suppliers/gpv.py`, `instances/calibrated.py`, and `scripts/calibrate.py` are all missing.
- **Implication**: Production-readiness cannot be claimed. The simulator is still uncalibrated.

## Cross-Verification Log
| Check | Primary | Gemini | Agreement | Signal |
|-------|---------|--------|-----------|--------|
| Scale-readiness verdict | Simulated training ready; production not ready | Simulated training ready; production not ready | Yes | Missing calibration is now the main blocker |

## Decision Trail
| Iter | Decision | Rationale | Next Action |
|------|----------|-----------|-------------|
| 1 | DIG DEEPER | Old diagnostics were stale and one adversarial metric was buggy | Fix harness, rerun substrate audit |
| 2 | REVISE | Correctness passed, runtime failed | Bound package language and rerun audit |
| 3 | COMPLETE | Simulated-training gates pass; calibration still fails | Scale simulated training only |

## Summary
- **Status**: REVISED
- **Confidence**: 92
- **Answer**: The substrate is now ready for large-scale simulated RL training, but still not ready for production claims because the calibration and retrospective-evaluation layer is missing.
- **Experiments run**: 5
- **Conjectures tested**: 1 (confirmed: 0, killed: 0, revised: 1)
- **Key mistake**: Treating one blocker as the whole story. Correctness, throughput, learnability, and calibration are separate gates.
