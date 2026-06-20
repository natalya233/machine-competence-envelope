# Machine competence under compound stress — reproducible code

Open code for the *Nature Machine Intelligence* Perspective **"Machine competence under compound stress: certifying when AI and its explanations can be trusted"** (N. Shakhovska, I. Izonin, S. A. Mitoulis).

Every figure and every reported number is produced by the scripts here from public datasets and open-source libraries only. No proprietary data or models are used. Released under the MIT licence.

## What it computes

A **competence envelope** is the region of operating conditions in which a (model, explainer) pair has *both* bounded prediction reliability (conformal coverage) and bounded explanation fidelity (faithfulness + stability). These scripts measure that envelope, its contraction under compound stress, and test whether the contraction law transfers across domains.

## Requirements

Python 3.12 and the packages in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The datasets (Wisconsin Diagnostic Breast Cancer; scikit-learn handwritten digits) ship with scikit-learn, so **no network access or data download is needed**.

## Reproduce everything

```bash
bash run_all.sh
```

This runs the three experiments (writing result grids to `*.npz`) and renders the three figures to `img/`. Total runtime is a few minutes on a laptop CPU. Random seeds are fixed (integers 1–8), so results are deterministic.

## Files

| Script | Produces | Paper item |
|---|---|---|
| `01_envelope_wdbc.py` | `exp_results.npz` — coverage / stability / accuracy / silent-error over drift × scarcity (WDBC, gradient-boosted trees + SHAP) | Fig. 1 |
| `02_contraction_wdbc.py` | `contraction.npz` — joint certified margin *g* and contraction-law fit over scarcity × contamination (WDBC) | Fig. 2a,b |
| `03_deep_transfer_digits.py` | `deep_transfer.npz` — deep MLP on digits 3-vs-8 with SmoothGrad explanation, conformal stability certificate (no network constant), and contraction-law fit for the transfer test | Fig. 2c,d |
| `04_fig1_envelope.py` | `img/fig1_envelope.png` | Fig. 1 |
| `05_fig2_compound_transfer.py` | `img/fig2_compound_transfer.png` | Fig. 2 |
| `06_fig3_infrastructure.py` | `img/fig3_infrastructure.png` (schematic) | Fig. 3 |

Full method specifications (operating-condition construction, conformal protocol, the randomized-smoothing stability certificate, and the bootstrap) are in the paper's Supplementary Information.

## Expected key results

These print to stdout when you run the experiments; reviewers can check them directly.

**`01_envelope_wdbc.py` (WDBC, drift × scarcity)**
- Shapley efficiency: `max |sum(phi) + base − margin| ≈ 1e-14` (faithfulness exact for trees).
- Conformal coverage degrades along **drift** (≈ 0.90 → 0.77), almost flat in scarcity.
- Explanation stability degrades along **scarcity** (≈ 0.07 → 0.37), almost flat in drift.
- The two certificates therefore bind on **different** axes (separable).

**`02_contraction_wdbc.py` (WDBC, scarcity × contamination)**
- Prediction-error interaction coefficient **c = 0.27, 95% CI [0.18, 0.36]** (bootstrap over seeds) → **super-additive** (significant).
- Explanation-stability interaction: not significant (additive).
- Joint certified margin *g* is positive only in the low-stress corner.

**`03_deep_transfer_digits.py` (digits 3-vs-8 MLP, scarcity × contamination)**
- SmoothGrad explanation stability degrades under contamination (≈ 0.05 → 0.16) and leaves the certified band — the explanation certificate works on a **deep** model, with no network Lipschitz constant.
- Prediction-error interaction coefficient **c = −0.02, 95% CI [−0.23, 0.19]** → **additive** (not significant).
- **Transfer test:** WDBC c = 0.27 vs digits c = −0.02 → the contraction law's interaction is **domain-specific, not invariant** (the strong form of hypothesis H2 is not supported).

Small numerical differences (±0.01) across platforms/BLAS are expected; the qualitative conclusions are stable.

## Citation

If you use this code, please cite the Perspective (see the manuscript for the full reference) and this repository.

## Licence

MIT — see `LICENSE`.
