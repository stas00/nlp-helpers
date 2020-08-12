# finetune mbart with fairseq using wmt dataset from huggingface/transformers

`run_data_prep.sh` takes a [wmt](http://www.statmt.org/wmt19/) dataset in the format of transformers summarization (i.e. with filenames of format `train.source`, `train.target`, `val.source`, etc.) and converts it to a format suitable to be fed to `fairseq-train` from https://github.com/pytorch/fairseq/.

The code is based on notes from
https://github.com/pytorch/fairseq/blob/master/examples/mbart/README.md#finetune-on-en-ro

Instructions:

1. install spm: https://github.com/google/sentencepiece#installation

2. install fairseq: https://github.com/pytorch/fairseq#requirements-and-installation

3. adjust a few variables on top of `./run_data_prep.sh` to fit your environment and run it. The script will create a `data` dir in the current directory and build everything under it. There are more notes and ideas in the script.

4. when step 2 is complete, adjust the finetuning script `finetune_mbart_cc25.sh` and run it
