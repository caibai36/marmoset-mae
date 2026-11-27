#!/bin/bash
# Data preparation for the public MIT marmoset sample.
#
# Turns raw paired recordings + Audacity label files into the spectrogram /
# target arrays consumed by the training and testing scripts.
#
# set -u : exit on undefined variable, -o pipefail : exit on error in a pipe
set -uo pipefail

# general configuration
stage=0

dataset_name="mit_sample"

# "train_dev" lists every uttid used for the training + development sets, as a
# sequence of animal pairs: "A B C D" -> (A,B) is the first pair, (C,D) the second.
# "dev_pair_ind" selects which pair(s) become the development set (0-indexed);
# 1 means the second pair. The remaining pairs are used for training.
train_dev="Cricket Enid Setta Sailor"
dev_pair_ind=1
test_wav_uttids="Athos Porthos" # the held-out test pair (prediction set)

# Path to the raw sample recordings. The folder holds subfolders of paired
# recordings, e.g.
#   $mit_sample/20150814_Cricket_Enid/20150814_Cricket_aligned.wav
#   $mit_sample/20150814_Cricket_Enid/Cricket.txt   (Audacity labels)
# Default is the copy on our lab machine. If it is missing (e.g. a fresh clone),
# the sample is downloaded automatically from sample_url below. Override the
# location with --mit_sample. (The old marmosetbehavior.mit.edu site is offline.)
mit_sample=/data/share/bin-wu/data/marmoset/vocalization/marmoset_mit_cnn/original/Wave_files
sample_url=https://github.com/caibai36/marmoset-mae/releases/download/v1.0/marmoset_sample.tar.gz

. local/scripts/parse_options.sh || exit 1

# Fetch the sample automatically when it is not present locally.
if [ ! -d "$mit_sample" ]; then
    echo "Sample not found at '$mit_sample'; downloading from $sample_url ..."
    mkdir -p downloads
    curl -L --fail "$sample_url" -o downloads/marmoset_sample.tar.gz
    tar -xzf downloads/marmoset_sample.tar.gz -C downloads
    mit_sample=downloads/Wave_files   # tarball contains a top-level Wave_files/
fi

if [ ${stage} -le 1 ]; then
    echo "Preparing wav.scp (audio) and aud.scp (Audacity labels)..."
    mkdir -p data/$dataset_name/
    rm -f data/$dataset_name/wav.scp data/$dataset_name/aud.scp

    for file in $mit_sample/*/*wav; do
	echo $(basename $file _aligned.wav | sed -e s'/20150814_//g' -e s'/_20161219//g') $file >> data/$dataset_name/wav.scp
    done

    for file in $mit_sample/*/*{Cricket,Enid,Sailor,Setta,Athos,Porthos}*txt; do
	echo $(basename $file .txt | sed 's/_20161219//g') $file >> data/$dataset_name/aud.scp
    done

    python local/scripts/data2info.py --data data/$dataset_name --scps aud:data/$dataset_name/aud.scp > data/$dataset_name/info.json
fi

if [ ${stage} -le 2 ]; then
    date
    echo "Preparing spectrograms and labels for the training and development sets..."
    python local/mit_cnn_data_prep.py --info_json data/$dataset_name/info.json \
	   --train_dev $train_dev \
	   --dev_pair_ind $dev_pair_ind \
	   --out_dir exp/data/$dataset_name
    date
fi

if [ ${stage} -le 3 ]; then
    date
    echo "Splitting each joint pair target into two per-animal targets..."
    python local/mit_cnn_target_norm_single2.py --train_target_multi exp/data/$dataset_name/train_target_multi \
	   --dev_target_multi exp/data/$dataset_name/dev_target_multi \
	   --train_target_single1 exp/data/$dataset_name/train_target_single1 \
	   --train_target_single2 exp/data/$dataset_name/train_target_single2 \
	   --dev_target_single1 exp/data/$dataset_name/dev_target_single1 \
	   --dev_target_single2 exp/data/$dataset_name/dev_target_single2
    date
fi

if [ ${stage} -le 4 ]; then
    date
    echo "Creating 2500ms spectral segments for the test set inputs..."
    python local/mit_cnn_wav_into_test_2500_raw.py --info_json data/$dataset_name/info.json \
	   --test_wav_uttids $test_wav_uttids \
	   --out_dir exp/data/$dataset_name
    date
fi
