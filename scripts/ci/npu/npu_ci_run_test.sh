#!/bin/bash

SUITE_NAME="$1"
PARTITION_ID="$2"
PARTITION_SIZE="$3"

pip install sglang_router
hf download lmms-lab/MMMU --repo-type dataset
pip install sentence_transformers torchaudio==2.10.0
pip install protobuf==6.31.1 zss pre-commit wandb>=0.16.0 tenacity==8.3.0 loguru openpyxl latex2sympy2 zstandard transformers-stream-generator tqdm-multiprocess pycocoevalcap
pip install yt-dlp sentencepiece==0.1.99 nltk av ftfy sqlitedict==2.1.0 sacrebleu>=1.5.0 pytablewriter black==24.1.0 isort==5.13.2 peft>=0.2.0 accelerate>=0.29.1
pip install jsonlines httpx==0.25.0 evaluate>=0.4.0 datasets==2.16.1 numexpr xgrammar==0.2.0 numpy==1.26.4 dotenv
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd ./lmms-eval
nohup pip install . > lmmslog.txt 2>&1 &
sleep 120
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd ../
cd test
python3 run_suite.py --hw npu --suite "$SUITE_NAME" --nightly --continue-on-error --timeout-per-file 3600 --auto-partition-id "$PARTITION_ID" --auto-partition-size "$PARTITION_SIZE"