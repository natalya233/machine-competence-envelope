#!/usr/bin/env bash
# Reproduce every number and figure in the paper, in order.
set -e
python 01_envelope_wdbc.py
python 02_contraction_wdbc.py
python 03_deep_transfer_digits.py
python 04_fig1_envelope.py
python 05_fig2_compound_transfer.py
python 06_fig3_infrastructure.py
echo "Done. Figures in ./img/ ; result grids in ./*.npz"
