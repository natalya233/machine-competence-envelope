# Changelog

## v1.1.0 — reviewer-response revision

Theory
- Added **Theorem 2 (ε-dormant approximate separation)** with proof and numerical
  verification (`src/theory/verify_epsilon_dormant.py`): prediction-side gaps grow O(ε)
  from zero while the explanation fidelity gap stays large.
- Unified the explanation side as a single **structural explanation functional**
  A(f; P_ref); conditioned the detection lower bound on a **fixed certification procedure**;
  added a **black-box / API** feasibility corollary.
- Made the Fig. 1c verification fully traceable (injected dummy coordinate, W=8,
  tolerance |Δ| < 1e-12, fixed seeds).

New analyses (`src/analysis/`)
- `reviewer_addendum.py`: monitor ablation (+bootstrap CIs), **clean-certification
  poisoning** with cue-dormancy/localisation audit, calibration-scales-with-n,
  interaction bootstrap CI.
- `reviewer_addendum2.py`: weighted-conformal comparator; multimodal temperature-scaling baseline.
- `reviewer_addendum3.py`: **SHAP-harmonised** explanation drift across all five architectures
  (common explanation family; resolves the Fig. 4 explanation-primitive concern).

Manuscript / figures
- Repositioned around the separation theorem; softened over-general claims; reframed the
  bridge case study as *deferral under low evidential reliability*; added a resilient-AI
  control-loop paragraph and flagship Nature/Science references.
- De-AI text pass (em-dashes reduced; inline emphasis bold → italic).
- Figure fixes: removed on-image figure titles; resolved legend/label overlaps in
  Figs 2, 4, 5, 6, 8, 9, 10; contrast and readability improvements.

## v1.0.1 — initial public release
- Separation theorem (exact), competence-envelope experiments, label-free monitor,
  adversarial poisoning, multimodal fusion, resource degradation, foundation-model runner,
  full manuscript builders.
