#!/bin/bash
# Fetch the released (full-corpus) pretrained checkpoints into models/pretrained/
# so you can run inference / evaluation without training. Use them via the
# --eval_model flag, e.g.:
#   ./run.sh --stage 3 --eval_model models/pretrained/cnn_model.ckpt
#   ./run_mae_vit.sh --stage 3 --eval_model models/pretrained/mae_vit_model_best_dev.ckpt
#
# Two sources are supported:
#   1. LOCAL : set MARMOSET_MODELS_DIR to a folder holding the two files (copied).
#   2. URL   : set MARMOSET_MODELS_URL to a base URL hosting the two files
#              (e.g. a GitHub Release download URL or a Hugging Face resolve URL).
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

DST_DIR="models/pretrained"
FILES=(cnn_model.ckpt mae_vit_model_best_dev.ckpt)

mkdir -p "$DST_DIR"
for f in "${FILES[@]}"; do
    dst="$DST_DIR/$f"
    if [ -n "${MARMOSET_MODELS_DIR:-}" ]; then
        echo "Copying ${MARMOSET_MODELS_DIR}/${f} -> ${dst}"
        cp "${MARMOSET_MODELS_DIR}/${f}" "$dst"
    elif [ -n "${MARMOSET_MODELS_URL:-}" ]; then
        echo "Downloading ${MARMOSET_MODELS_URL}/${f} -> ${dst}"
        curl -L --fail -o "$dst" "${MARMOSET_MODELS_URL}/${f}"
    else
        echo "ERROR: set MARMOSET_MODELS_DIR (local folder) or MARMOSET_MODELS_URL (download base)." >&2
        echo "See models/MODELS.md." >&2
        exit 1
    fi
done
echo "Done -> $DST_DIR/"
ls -lh "$DST_DIR"
