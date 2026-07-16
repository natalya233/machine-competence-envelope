# Changelog

## v1.2.0 — real-LLM separation (Mode B) + information-theoretic core
- **Mode B — separation on a genuine LLM.** LoRA-adapted Qwen2.5-1.5B with true token-level
  integrated-gradient attributions (src/analysis/mode_b/): conformal coverage degrades under drift
  (0.886->0.811); a data-poisoning backdoor drives attack success 0.39->1.00 while clean-data
  coverage holds 0.88-0.90 -- Theorem 1 on a real language model, not a probe. Honest third-instance
  finding: a clean-data explanation *audit* is as blind to the backdoor as a clean-data prediction
  certificate (input-independent |coef| exposes the cue; input-dependent IG on clean inputs does not).
  New Results section + Fig. 6; figures renumbered (6->7 ... 10->11).
- **Information-theoretic formulation in Methods/SI:** prediction-side information I_P, structural
  information I_S, and Lemma 1 (non-identifiability): I_P->I_S is not injective; Theorem 1 is its
  quantitative form.
- Abstract -> Nature summary paragraph. Claims updated to include large-language-model family.

## v1.1.0 -- reviewer-response revision
- Theorem 2 (epsilon-dormant) + proof + verification; unified structural functional; fixed
  certification procedure; Fig 1c traceability; black-box corollary. Monitor ablation (+CIs),
  clean-certify poisoning + cue audit, calibration-scales-with-n, interaction bootstrap CI,
  SHAP-harmonised drift across 5 architectures, weighted-conformal. Repositioning, de-AI pass,
  figure overlap fixes, flagship references.

## v1.0.1 -- initial public release
- Separation theorem, competence-envelope experiments, monitor, poisoning, multimodal, resource
  degradation, foundation-model runner.
