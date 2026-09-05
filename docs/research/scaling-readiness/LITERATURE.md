# Literature

## Grounding
- The repo’s own specs make GPV calibration and retrospective replay part of the definition of success:
  - `docs/research/procurement-rl/PRODUCT_INTENT.md`
  - `docs/research/procurement-rl/PROCUREMENT_GYM_SPEC.md`
- `UNDERSTANDING.md` explicitly frames GPV calibration plus held-out retrospective evaluation as the sim-to-real bridge.

## Practitioner signal
- Combinatorial procurement is valuable precisely because bundles matter, but supplier bid expressiveness is limited in practice. That makes bounded bundle languages more realistic than exhaustive enumeration.

## Cross-model critique
- Gemini critique on 2026-04-15 first identified runtime as the main training blocker.
- After the bounded package-language fix and search-uplift probe, Gemini agreed with the revised conclusion:
  - Simulated training readiness is justified.
  - Production readiness is still blocked by missing calibration/replay artifacts.
