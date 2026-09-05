# Conjectures

## Active
### C-1: Correct Oracle Implies Scaling Readiness
- **Statement**: Once the simulator/verifier is economically correct, the substrate is ready to scale RL training toward production.
- **Kill criterion**: Any of the following fails: deterministic/economically sane rewards, operationally tractable verifier under high-lot exploration, or implemented sim-to-real calibration bridge.
- **Confidence**: 25
- **Evidence for**:
- Correctness at `coupling=1.0` now passes.
- Package coupling is materially active.
- Runtime cap reduced worst-case verifier p95 to `0.093s`.
- Lightweight search still beats heuristics on a nontrivial subset of instances.
- **Evidence against**:
- Calibration artifacts required by spec are missing.
- **Experiments pending**:
- None for this conjecture. It was revised into separate correctness, runtime, learnability, and calibration gates.
- **Status**: REVISED → runtime/calibration split

## Confirmed

## Killed

## Revised
### C-1 → Split verdict
- **Revision**: Oracle correctness, runtime, and lightweight learnability probes now pass. Production readiness remains blocked by missing calibration.
