# The Competence Envelope: certifying when AI and its explanations can be trusted

Reference implementation and reproducibility code for the manuscript

> **Prediction certification cannot replace explanation certification: a competence envelope for trustworthy AI under compound stress**
> Nataliya Shakhovska, Ivan Izonin (Lviv Polytechnic National University); Stergios A. Mitoulis (University of Birmingham).

The paper proves a **separation theorem**: monitoring what a model *predicts* is, within the class of certification-time prediction-law functionals, provably insufficient to certify whether it can be trusted. A reliable model and a compromised one can be made identical to every prediction-side certificate (coverage, accuracy, calibration — to machine precision) yet differ arbitrarily in the fidelity of their *explanations* and in behaviour under shift. The two certificates are organised by a **competence envelope**: the region of operating conditions over which predictions and explanations can jointly be trusted.

This repository reproduces every figure, theorem verification, and reported number from fixed seeds.

---

## Repository layout

```
.
├── src/
│   ├── theory/        # numerical verification of the theorems
│   │   ├── verify_separation.py       # Theorem 1 (exact separation): max|f−f'|=0, KS=0, fidelity separates
│   │   ├── verify_epsilon_dormant.py  # Theorem 2 (ε-dormant approximate separation)
│   │   └── verify_theorem.py          # Proposition 1 (existence, star-shape, two binding cones)
│   ├── analysis/      # empirical envelope, monitor, adversarial, multimodal, resource, breadth
│   │   ├── analyze*.py
│   │   ├── reviewer_addendum.py       # monitor ablation, clean-certify poisoning + cue audit,
│   │   │                              # calibration-scales-with-n, interaction bootstrap CI
│   │   ├── reviewer_addendum2.py      # weighted-conformal comparator; multimodal temp-scaling baseline
│   │   └── reviewer_addendum3.py      # SHAP-harmonised explanation drift across 5 architectures
│   ├── figures/       # figure generators (make_*.py) -> PNGs
│   └── manuscript/    # docx builders (Node.js + docx-js): article, SI, cover letter
├── data/              # place public datasets here (see data/README.md)
├── results/           # JSON/NPY outputs (generated)
├── figures/           # generated figure PNGs
├── requirements.txt
├── run_all.sh         # end-to-end reproduction
└── LICENSE
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Verify the theorems (no data needed — synthetic constructions)
python src/theory/verify_separation.py
python src/theory/verify_epsilon_dormant.py
python src/theory/verify_theorem.py

# 2. Reproduce the empirical analyses (needs data/ — see below)
export DATA_DIR=./data
python src/analysis/reviewer_addendum.py     # monitor ablation, clean-certify poisoning, etc.
python src/analysis/reviewer_addendum2.py     # weighted conformal; multimodal baselines

# 3. Regenerate figures
python src/analysis/mode_b/  # real-LLM (LoRA + integrated gradients) runner + README_ModeB.md
src/figures/make_figs.py               # Fig 2–3 (envelope, cross-domain)
python src/analysis/mode_b/  # real-LLM (LoRA + integrated gradients) runner + README_ModeB.md
src/figures/make_fig_theorem.py        # Fig 1 (separation theorem)
# ... (see run_all.sh for the full list)
```

All scripts read data from `$DATA_DIR` (default `./data`) and write intermediate `.npy`/`.json`
and figure `.png` files to the working directory. Run from the repository root.

## Reproducing the theorem (no data required)

The core result is verified on controlled constructions and requires no external data:

| Script | Verifies | Key output |
|---|---|---|
| `verify_separation.py` | Theorem 1 (exact) | max\|f−f'\| = 0; coverage/accuracy/ECE Δ = 0; KS = 0; fidelity 1.00→0.19; deployment acc 0.82→0.41 |
| `verify_epsilon_dormant.py` | Theorem 2 (ε-dormant) | prediction gaps grow O(ε) from 0 (coverage gap ≤ 0.005 at ε = 0.1); structural fidelity gap ≈ 0.4 throughout |
| `verify_theorem.py` | Proposition 1 | envelope non-empty; 0 star-shape violations / 121 cells; coverage binds first on drift, fidelity first on scarcity |

## Data

The empirical study uses **public** data only. See [`data/README.md`](data/README.md) for sources and
expected filenames. In brief:

- **Telegram (information integrity)** — public war-reporting channel posts (`telegram_2026.csv`).
- **Reddit climate discourse** — Harvard Dataverse `doi:10.7910/DVN/NL06IX` (`data/climate/Climate_Dataset/...`).
- **Clinical tabular** — Wisconsin Diagnostic Breast Cancer (bundled with scikit-learn).
- **CMU-MOSI** — public multimodal sentiment features (`mosi_features.npz`).
- **Eight text benchmarks** (breadth study) and **transformer backbones** — via the optional
  foundation-model runner (`src/analysis` + Hugging Face; needs network access).

## Foundation-model runner (optional)

Reproducing Fig 5 (foundation-model backbones) and the cross-dataset breadth study requires
Hugging Face `transformers`/`datasets` and network access. The runner (`run_transformer_envelope.py`,
`run_hf_datasets.py`) computes frozen mean-pooled embeddings for DistilBERT-multilingual, XLM-R and
MiniLM and re-runs the identical envelope protocol; outputs drop into the same JSON format the
figure scripts consume.

## Rebuilding the manuscript (optional)

The `.docx` article, Supplementary Information and cover letter are built with Node.js and `docx`:

```bash
npm install -g docx
NODE_PATH=$(npm root -g) node src/manuscript/build2.js          # article (expects figures/*.png)
NODE_PATH=$(npm root -g) node src/manuscript/build_si.js        # supplementary information
NODE_PATH=$(npm root -g) node src/manuscript/build_coverletter.js
```

## Citation

See [`CITATION.cff`](CITATION.cff). If you use this code, please cite the manuscript.

## License

MIT — see [`LICENSE`](LICENSE).
