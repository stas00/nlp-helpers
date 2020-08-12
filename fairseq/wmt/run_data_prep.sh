#!/bin/bash

# set -x

# this script takes a wmt dataset in the format of transformers summarization
# (i.e. with filenames of format `train.source`, `train.target`, `val.source`, etc.)
# and converts it to a format suitable to be fed to `fairseq-train` from
# https://github.com/pytorch/fairseq/.

# this is built based on instructions at
# https://github.com/pytorch/fairseq/blob/master/examples/mbart/README.md
# therefore if anything doesn't work, see the url above for details

# An example of a small wmt dataset can be found here
# https://github.com/huggingface/transformers/tree/master/examples/seq2seq/test_data/wmt_en_ro
# we recommend you use it first to get familiar with the process
#
# if you want to use a full wmt dataset you can download one with the help of this utility:
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/download_wmt.py

# clone transformers if you haven't already
#
# git clone https://github.com/huggingface/transformers/
#
# If you're using all defaults then you only need to change the path to where
# you cloned https://github.com/huggingface/transformers/
TRANSFORMERS=/code/huggingface/transformers-master

# if it's a different dataset and with a different language pair adjust the settings below:
WMT_DATASET=$TRANSFORMERS/examples/seq2seq/test_data/wmt_en_ro

# Change the source and destination languages (The short and the long form)
SRC_SHORT=en
TGT_SHORT=ro
SRC=en_XX
TGT=ro_RO

SPM=spm_encode # adjust the path if it was not installed system-wide

##############################################
# no need to change anything below this line #

# validate path
if [ ! -d "$WMT_DATASET" ]; then
    echo "$WMT_DATASET doesn't exist. Please check the WMT_DATASET setting"
    exit 1
fi

# validate we have a working spm_encode
command -v spm_encode >/dev/null 2>&1 || { echo >&2 "$me requires 'spm_encode' but it's not installed. Installation info: https://github.com/google/sentencepiece#installation"; exit 1; }

BASE=data
MODEL=$BASE/mbart.cc25/sentence.bpe.model
DATA=$BASE/wmt-$SRC_SHORT-$TGT_SHORT

TRAIN=train
VALID=val
TEST=test

BPE=$BASE/bpe

mkdir -p $DATA

# get model
echo "Downloading/extracting the model"
if [ ! -f "$BASE/mbart.CC25.tar.gz" ]; then
    cd $BASE
    wget https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.CC25.tar.gz
    tar -xzvf mbart.CC25.tar.gz
    cd -
fi

# spm conversion to bpe
echo "Doing spm conversion"
$SPM --model=$MODEL --input $WMT_DATASET/$TRAIN.source --output $DATA/$TRAIN.spm.$SRC
$SPM --model=$MODEL --input $WMT_DATASET/$TRAIN.target --output $DATA/$TRAIN.spm.$TGT
$SPM --model=$MODEL --input $WMT_DATASET/$VALID.source --output $DATA/$VALID.spm.$SRC
$SPM --model=$MODEL --input $WMT_DATASET/$VALID.target --output $DATA/$VALID.spm.$TGT
$SPM --model=$MODEL --input $WMT_DATASET/$TEST.source  --output $DATA/$TEST.spm.$SRC
$SPM --model=$MODEL --input $WMT_DATASET/$TEST.target  --output $DATA/$TEST.spm.$TGT

# preprocess data
echo "Doing data preprocessing"
DICT=$BASE/mbart.cc25/dict.txt
fairseq-preprocess \
  --source-lang $SRC \
  --target-lang $TGT \
  --trainpref $DATA/$TRAIN.spm \
  --validpref $DATA/$VALID.spm \
  --testpref $DATA/$TEST.spm \
  --destdir $BPE \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict $DICT \
  --tgtdict $DICT \
  --workers 70

echo "\nThe data under '$BPE' has been prepared for fairseq-train use"
