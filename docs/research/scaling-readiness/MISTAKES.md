# Mistakes

## M-1: Trusted stale diagnostics
- **What happened**: Existing validation/adversarial JSON files were treated as current state.
- **Why**: They predated the simulator fix and one adversarial metric had a counting bug.
- **Pattern**: Audit harness drift.
- **Fix**: Add regression test for the adversarial percentage helper and rerun a fresh readiness audit.

## M-2: Collapsed distinct readiness questions
- **What happened**: Correctness, training throughput, and production readiness were treated as one binary question.
- **Why**: The repo had a single “reward signal” framing.
- **Pattern**: Category error in evaluation.
- **Fix**: Split the audit into determinism/sanity, signal/mechanism, runtime, and calibration.

## M-3: Confused heuristic dominance with triviality
- **What happened**: A strong category baseline was almost treated as proof that the environment was not worth scaling.
- **Why**: Win rate alone hides whether local search can still extract meaningful gains.
- **Pattern**: Over-reading baseline dominance.
- **Fix**: Add a lightweight search-uplift probe to the readiness audit.

## Recurring Patterns
| Pattern | Count | Mitigation |
|---------|-------|------------|
| Stale measurement artifacts | 1 | Prefer reruns over cached JSON |
| Weak audit metrics | 1 | Add unit-tested helpers and explicit gates |
| Premature triviality claim | 1 | Check search uplift before concluding the landscape is flat |
