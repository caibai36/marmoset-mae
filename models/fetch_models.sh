#!/bin/bash
# Download the released (full-corpus) pretrained checkpoints into models/pretrained/
# so you can run inference / evaluation without training. Use them via --eval_model:
#   ./run.sh         --stage 3 --eval_model models/pretrained/cnn_model.ckpt
#   ./run_mae_vit.sh --stage 3 --eval_model models/pretrained/mae_vit_model_best_dev.ckpt
#
# By default it pulls from this repo's v1.0 GitHub Release. Override the source:
#   MARMOSET_MODELS_URL=<base url>  (download base; files appended)
#   MARMOSET_MODELS_DIR=<folder>    (copy from a local folder instead)
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

BASE_URL="${MARMOSET_MODELS_URL:-https://github.com/caibai36/marmoset-mae/releases/download/v1.0}"
DST_DIR="models/pretrained"
FILES=(cnn_model.ckpt mae_vit_model_best_dev.ckpt)

mkdir -p "$DST_DIR"
for f in "${FILES[@]}"; do
    dst="$DST_DIR/$f"
    if [ -s "$dst" ]; then
        echo "Already present: $dst"
    elif [ -n "${MARMOSET_MODELS_DIR:-}" ]; then
        echo "Copying ${MARMOSET_MODELS_DIR}/${f} -> ${dst}"
        cp "${MARMOSET_MODELS_DIR}/${f}" "$dst"
    else
        echo "Downloading ${BASE_URL}/${f} -> ${dst}"
        curl -L --fail -o "$dst" "${BASE_URL}/${f}"
    fi
done
echo "Done -> $DST_DIR/"
ls -lh "$DST_DIR"
