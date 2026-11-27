# Pretrained models

These are the **full-corpus checkpoints that produced the paper numbers** (ASRU
2025). Use them for inference / evaluation without training. They are too large
for plain git, so they are hosted as release assets and git-ignored here.

| Model | Published file | Size | `--eval_model` path after fetch |
|-------|----------------|-----:|----------------------------------|
| CNN | `cnn_model.ckpt` | ~104 MB | `models/pretrained/cnn_model.ckpt` |
| MAE-ViT | `mae_vit_model_best_dev.ckpt` | ~1.4 GB | `models/pretrained/mae_vit_model_best_dev.ckpt` |

## Download

```bash
# From a GitHub Release (or Hugging Face) that hosts the two files:
MARMOSET_MODELS_URL=https://github.com/caibai36/marmoset-mae/releases/download/v1.0 \
    ./models/fetch_models.sh

# Or copy from a local folder that holds the two files:
MARMOSET_MODELS_DIR=/path/to/checkpoints ./models/fetch_models.sh
```

Then run inference / evaluation with the fetched checkpoint:

```bash
./run.sh         --stage 3 --eval_model models/pretrained/cnn_model.ckpt
./run_mae_vit.sh --stage 3 --eval_model models/pretrained/mae_vit_model_best_dev.ckpt
```

> Note: these models were trained on the full corpus, so stage-3 prediction
> expects that corpus's test inputs. For the public sample, train with
> `run.sh` / `run_mae_vit.sh` (stages 1–2) first.

## Publishing the checkpoints (maintainer notes)

Source files on the lab machine (rename to the published names when uploading):

| Published name | Source path (under the research dir `egs/riken/mit_cnn_s0_wu`) |
|----------------|----------------------------------------------------------------|
| `cnn_model.ckpt` | `exp/run/mit_data/mit_data0/mit_cnn_72-run0/bs25lr0.0003evalinterval1avgpredwin5/train/model.ckpt` |
| `mae_vit_model_best_dev.ckpt` | `exp/run_mae_vit/mit_data/mit_data0/mit_vit_72-run0/bs25lr0.0003evalinterval1avgpredwin5_dim768depth12heads12mlpdim3072patch16dropout0.1sharedim1024maemaepretrainedmodel_epoch_400_base_48days.pttrainlayers6/train/model_best_dev.ckpt` |

Also pack the sample audio (the `run*.sh` auto-download expects a tarball with a
top-level `Wave_files/` folder, CC BY 4.0):

```bash
tar -czf marmoset_sample.tar.gz -C /path/to/raw_sample Wave_files
```

Upload everything as the `v1.0` GitHub Release referenced by the scripts:

```bash
cp <cnn source>  cnn_model.ckpt
cp <mae source>  mae_vit_model_best_dev.ckpt
gh release create v1.0 cnn_model.ckpt mae_vit_model_best_dev.ckpt marmoset_sample.tar.gz \
    --title "Models + sample (ASRU 2025)" \
    --notes "CNN and MAE-ViT checkpoints (full corpus) and the runnable sample audio"
```

Checkpoints can instead live on a Hugging Face model repo; set `MARMOSET_MODELS_URL`
to `https://huggingface.co/<user>/<repo>/resolve/main`.
