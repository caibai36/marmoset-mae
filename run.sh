#!/bin/bash
# =============================================================================
# CNN baseline: two-stream convolutional model for marmoset call
# segmentation, classification and caller identification.
#
# Pipeline (Kaldi-style stages; set --stage N to resume from stage N):
#   1. data preparation (spectrograms + targets from the raw sample)
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
run=run0
exp_dir=exp/run

# model: mit_cnn_72
#   72 = two-stream model with 9+9 output classes (model index 72 on
#   the MIT McGovern marmoset-behavior project, now offline). The 9 classes are "noise" plus
#   trill, twitter, phee, triphee, tsik, ek, chirp and chatter.
dataset_name=mit_sample
data_name=mit_sample0   # train: Cricket,Enid / dev: Setta,Sailor / test: Athos,Porthos
model_name=mit_cnn_72
eval_model=""           # optional: checkpoint to evaluate at stage 3; empty = the model just trained

# ----- training hyper-parameters (Oikarinen et al., 2019 setup) -----
batch_size=25
lr=0.0003               # learning rate, exponentially decayed every eval_interval
dropout_rate=0.4
epsilon=0.001           # Adam epsilon
num_iter=2601           # full-data runs used 74001
eval_interval=200       # evaluate (and decay lr by 0.97) every this many iterations
avg_pred_win=5          # average predictions over this many consecutive windows

# ----- evaluation -----
# A window is predicted "noise" when its top class probability is below the cutoff.
cutoffs="0 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95"
report_cutoff=0.6       # cutoff used for the detailed caller / boundary report
tolerance=0.1           # boundary-match tolerance in seconds

# ----- data -----
# Raw sample folder, passed through to local/mit_sample_data_prep.sh at stage 1.
# Default is the copy on our lab machine. The old marmosetbehavior.mit.edu site is
# offline; the corpus is on OSF (https://osf.io/pswqd/). Override with --mit_sample.
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files
first="Athos"           # first uttid of the test pair
sec="Porthos"           # second uttid of the test pair

. local/scripts/parse_options.sh || exit 1

exp_root=$exp_dir/$dataset_name/$data_name/${model_name}-${run}/bs${batch_size}lr${lr}evalinterval${eval_interval}avgpredwin${avg_pred_win}
result_dir=$exp_root/train
eval_dir=$exp_root/eval

if [ ${stage} -le 1 ]; then
    date; echo "Stage 1: data preparation..."
    ./local/mit_sample_data_prep.sh --mit_sample $mit_sample --stage 0
    date
fi

mkdir -p $result_dir
if [ ${stage} -le 2 ]; then
    date; echo "Stage 2: train CNN..."
    python local/mit_cnn_train.py \
	   --train_input1 exp/data/$dataset_name/train_input1 \
	   --train_input2 exp/data/$dataset_name/train_input2 \
	   --train_target_single1 exp/data/$dataset_name/train_target_single1 \
	   --train_target_single2 exp/data/$dataset_name/train_target_single2 \
	   --dev_input1 exp/data/$dataset_name/dev_input1 \
	   --dev_input2 exp/data/$dataset_name/dev_input2 \
	   --dev_target_single1 exp/data/$dataset_name/dev_target_single1 \
	   --dev_target_single2 exp/data/$dataset_name/dev_target_single2 \
	   --batch_size $batch_size \
	   --dropout_rate $dropout_rate \
	   --lr $lr \
	   --epsilon $epsilon \
	   --num_iter $num_iter \
	   --eval_interval $eval_interval \
	   --avg_pred_win $avg_pred_win \
	   --result $result_dir
    date
fi

mkdir -p $eval_dir
if [ ${stage} -le 3 ]; then
    date; echo "Stage 3: test / predict..."
    python local/mit_cnn_test.py \
	   --test_input1 exp/data/$dataset_name/test_input1_$first \
	   --test_input2 exp/data/$dataset_name/test_input2_$sec \
	   --test_pred1 $eval_dir/test_pred1_$first \
	   --test_pred2 $eval_dir/test_pred2_$sec \
	   --batch_size $batch_size \
	   --dropout_rate $dropout_rate \
	   --lr $lr \
	   --epsilon $epsilon \
	   --avg_pred_win $avg_pred_win \
	   --eval_model "${eval_model:-$result_dir/model.ckpt}"
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

    # (a) Classification accuracy swept over every cutoff.
    rm -f $eval_dir/results.txt
    for cutoff in $cutoffs; do
	echo Cutoff: $cutoff | tee -a $eval_dir/results.txt
	python local/mit_cnn_eval_acc.py --info_json data/$dataset_name/info.json \
	       --hypo_files $eval_dir/test_pred1_${first}_cutoff$cutoff.txt $eval_dir/test_pred2_${sec}_cutoff$cutoff.txt \
	       --ref_uttids $first $sec | tee -a $eval_dir/results.txt
	echo | tee -a $eval_dir/results.txt
    done

    # (b) Detailed report at report_cutoff: caller accuracy + boundary F1.
    ref1=$(python -c "import json;print(json.load(open('data/$dataset_name/info.json'))['$first']['aud'])")
    ref2=$(python -c "import json;print(json.load(open('data/$dataset_name/info.json'))['$sec']['aud'])")
    hypo1=$eval_dir/test_pred1_${first}_cutoff${report_cutoff}.txt
    hypo2=$eval_dir/test_pred2_${sec}_cutoff${report_cutoff}.txt
    echo -e "\n===== Detailed report (cutoff ${report_cutoff}) ====="
    python local/mit_cnn_eval_acc_spk.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2
    python local/mit_cnn_eval_acc_boundaries.py --hypo_files $hypo1 $hypo2 --ref_files $ref1 $ref2 --tolerance $tolerance
    date
fi
