#!/bin/bash
# =============================================================================
# MAE-ViT: two-stream Vision Transformer initialised from a masked
# autoencoder (facebook/vit-mae-base), for marmoset call segmentation,
# classification and caller identification. This is the best model in the paper.
#
# Pipeline (Kaldi-style stages; set --stage N to resume from stage N):
#   1. data preparation (shared with run.sh)
#   2. train
#   3. test / predict on the held-out pair
#   4. apply confidence cutoffs to the predictions
#   5. evaluate (classification accuracy, caller accuracy, boundary F1)
#
# Default dataset is the MIT marmoset sample (corpus on OSF https://osf.io/pswqd/,
# CC BY 4.0; the old marmosetbehavior.mit.edu site is offline).
# See README.md for the full-data setup used for the paper numbers.
# =============================================================================
set -uo pipefail

# ----- general configuration -----
stage=1
seed=2020
gpu=auto                # "auto" picks the freest GPU; or set an index like 0
run=run0
exp_dir=exp/run_mae_vit

dataset_name=mit_sample
data_name=mit_sample0   # train: Cricket,Enid / dev: Setta,Sailor / test: Athos,Porthos
model_name=mit_vit_72
eval_model=""           # checkpoint to evaluate at stage 3; empty = $result_dir/model_best_dev.ckpt

# ----- training hyper-parameters -----
batch_size=25
lr=0.0003
epsilon=0.001
num_epochs=100
eval_epoch_interval=1   # evaluate / checkpoint every this many epochs
avg_pred_win=5

# ----- evaluation -----
cutoffs="0 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"
report_cutoff=0.6
tolerance=0.1

# ----- ViT architecture -----
image_size=224
patch_size=16
num_classes=9
dim=768
depth=12
heads=12
mlp_dim=3072
pool=cls
channels=1
dim_head=64
vit_dropout=0.1
emb_dropout=0.1
dropout_rate=0.4        # dropout of the shared classification head
shared_dim=1024         # width of the shared fully connected layer

# ----- MAE initialisation -----
model_type=mae          # "mae" (MAE-pretrained ViT) or "vit" (from scratch)
mae_pretrained=base     # "base" = facebook/vit-mae-base; a path; or "" for none
mae_train_layers=6      # number of transformer layers to fine-tune (-1 = all)

# ----- data -----
# Default is the copy on our lab machine. The old marmosetbehavior.mit.edu site is
# offline; the corpus is on OSF (https://osf.io/pswqd/). Override with --mit_sample.
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files
first="Athos"
sec="Porthos"

. local/scripts/parse_options.sh || exit 1

if [ ${stage} -le 1 ]; then
    date; echo "Stage 1: data preparation..."
    ./local/mit_sample_data_prep.sh --mit_sample $mit_sample --stage 0
    date
fi

vit_params="dim${dim}depth${depth}heads${heads}mlpdim${mlp_dim}patch${patch_size}dropout${vit_dropout}sharedim${shared_dim}mae${model_type}pretrained${mae_pretrained:+$(basename ${mae_pretrained} .ckpt)}trainlayers${mae_train_layers}"
exp_root=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}evalinterval${eval_epoch_interval}avgpredwin${avg_pred_win}_${vit_params}
result_dir=$exp_root/train
eval_dir=$exp_root/eval

mkdir -p $result_dir
if [ ${stage} -le 2 ]; then
    date; echo "Stage 2: train MAE-ViT..."
    python local/mit_mae_vit_train.py \
       --train_input1 exp/data/$dataset_name/train_input1 \
       --train_input2 exp/data/$dataset_name/train_input2 \
       --train_target_single1 exp/data/$dataset_name/train_target_single1 \
       --train_target_single2 exp/data/$dataset_name/train_target_single2 \
       --dev_input1 exp/data/$dataset_name/dev_input1 \
       --dev_input2 exp/data/$dataset_name/dev_input2 \
       --dev_target_single1 exp/data/$dataset_name/dev_target_single1 \
       --dev_target_single2 exp/data/$dataset_name/dev_target_single2 \
       --batch_size $batch_size --lr $lr --epsilon $epsilon \
       --num_epochs $num_epochs --eval_epoch_interval $eval_epoch_interval \
       --avg_pred_win $avg_pred_win --result $result_dir \
       --image_size $image_size --patch_size $patch_size --num_classes $num_classes \
       --dim $dim --depth $depth --heads $heads --mlp_dim $mlp_dim \
       --pool $pool --channels $channels --dim_head $dim_head \
       --vit_dropout $vit_dropout --emb_dropout $emb_dropout \
       --dropout_rate $dropout_rate --shared_dim $shared_dim \
       --model_type $model_type --mae_pretrained "${mae_pretrained}" \
       --mae_train_layers $mae_train_layers --seed $seed --gpu $gpu --overwrite
    date
fi

mkdir -p $eval_dir
if [ ${stage} -le 3 ]; then
    date; echo "Stage 3: test / predict..."
    python local/mit_mae_vit_test.py \
       --test_input1 exp/data/$dataset_name/test_input1_$first \
       --test_input2 exp/data/$dataset_name/test_input2_$sec \
       --test_pred1 $eval_dir/test_pred1_$first \
       --test_pred2 $eval_dir/test_pred2_$sec \
       --batch_size $batch_size --avg_pred_win $avg_pred_win \
       --eval_model "${eval_model:-$result_dir/model_best_dev.ckpt}" \
       --image_size $image_size --patch_size $patch_size --num_classes $num_classes \
       --dim $dim --depth $depth --heads $heads --mlp_dim $mlp_dim \
       --pool $pool --channels $channels --dim_head $dim_head \
       --vit_dropout $vit_dropout --emb_dropout $emb_dropout \
       --dropout_rate $dropout_rate --shared_dim $shared_dim \
       --model_type $model_type --mae_pretrained "${mae_pretrained}" \
       --mae_train_layers $mae_train_layers --seed $seed --gpu $gpu
    date
fi

if [ ${stage} -le 4 ]; then
    date; echo "Stage 4: apply confidence cutoffs..."
    python local/mit_cnn_cutoff_predictor_single.py --cutoffs $cutoffs \
       --pred_files $eval_dir/test_pred1_$first.npy $eval_dir/test_pred2_$sec.npy \
       --out_dir $eval_dir
    date
fi

if [ ${stage} -le 5 ]; then
    date; echo "Stage 5: evaluate..."

    rm -f $eval_dir/results.txt
    for cutoff in $cutoffs; do
	echo Cutoff: $cutoff | tee -a $eval_dir/results.txt
	python local/mit_cnn_eval_acc.py --info_json data/$dataset_name/info.json \
	       --hypo_files $eval_dir/test_pred1_${first}_cutoff$cutoff.txt $eval_dir/test_pred2_${sec}_cutoff$cutoff.txt \
	       --ref_uttids $first $sec | tee -a $eval_dir/results.txt
	echo | tee -a $eval_dir/results.txt
    done

    ref1=$(python -c "import json;print(json.load(open('data/$dataset_name/info.json'))['$first']['aud'])")
    ref2=$(python -c "import json;print(json.load(open('data/$dataset_name/info.json'))['$sec']['aud'])")
    hypo1=$eval_dir/test_pred1_${first}_cutoff${report_cutoff}.txt
    hypo2=$eval_dir/test_pred2_${sec}_cutoff${report_cutoff}.txt
    echo -e "\n===== Detailed report (cutoff ${report_cutoff}) ====="
    python local/mit_cnn_eval_acc_spk.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2
    python local/mit_cnn_eval_acc_boundaries.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --tolerance $tolerance
    date
fi
