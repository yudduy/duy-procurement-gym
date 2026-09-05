# TODO

## Now
- Implement the calibration bridge promised in the spec:
  - `src/procurement_gym/suppliers/gpv.py`
  - `src/procurement_gym/instances/calibrated.py`
  - `scripts/calibrate.py`
- Add retrospective replay on at least one historical tender.

## Next
- Scale simulated RL training with the current substrate.
- Keep rerunning `scripts/verify_scaling_readiness.py` after major model/simulator changes.

## Later
- After calibration exists, add GPV robustness and retrospective replay to the readiness audit.
