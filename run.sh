#!/usr/bin/env bash
# Run each task end-to-end (eval -> train -> eval). Results are written to
# results/<task>/<task>.txt. Defaults in arguments.py are now set for a working
# run (--embed-dim 768, --n-epochs 10, etc.), so the flags below are just the
# knobs most worth tuning. Run tasks individually by commenting out the others.

# Task 1: baseline BERT + classifier head, cross-entropy loss.
# Tuned hyperparameters (lr 2e-5, drop 0.1, hidden-dim 150, max-len 32) are the
# defaults in arguments.py, so no extra flags are needed.
python main.py --task baseline

# Task 2: custom fine-tuning technique (e.g. reinitialize the last N encoder layers).
python main.py --task custom --reinit_n_layers 2

# Task 3: contrastive learning (SupContrast / SimCLR). Larger batches usually help.
python main.py --task supcon --batch-size 32
