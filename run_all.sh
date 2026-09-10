#!/usr/bin/env bash
# Reproduce theorem verification, empirical analyses and figures.
# Usage:  DATA_DIR=./data bash run_all.sh
set -e
export DATA_DIR="${DATA_DIR:-./data}"
echo "== 1. Theorem verification (no data required) =="
python src/theory/verify_separation.py
python src/theory/verify_epsilon_dormant.py
python src/theory/verify_theorem.py

echo "== 2. Empirical analyses (require \$DATA_DIR=$DATA_DIR) =="
python src/analysis/reviewer_addendum.py
python src/analysis/reviewer_addendum2.py
python src/analysis/reviewer_addendum3.py
# core envelope / monitor / adversarial / multimodal / resource analyses:
for f in src/analysis/analyze*.py; do
  echo "-- $f"; python "$f" || echo "   (skipped: $f — check data availability)"
done

echo "== 3. Figures =="
python src/figures/make_fig_theorem.py
python src/figures/make_figs.py
python src/figures/make_figs2.py
python src/figures/make_fig3.py
python src/figures/make_fig_resdeg.py
python src/figures/make_fig_multiarch.py
python src/figures/make_fig_transformer.py || echo "   (Fig 5 needs transformer_results.json from the HF runner)"
python src/figures/make_pillar3.py

echo "== done. Figures (*.png) and results (*.json) are in the working directory. =="
