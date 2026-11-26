#!/bin/bash
# Eval packages and data the nightly NPU tests use on top of the sglang install.
set -euo pipefail

pip install sentence_transformers zss "wandb>=0.16.0" tenacity==8.3.0 loguru openpyxl latex2sympy2 zstandard transformers-stream-generator tqdm-multiprocess pycocoevalcap
pip install yt-dlp sentencepiece==0.1.99 nltk av ftfy sqlitedict==2.1.0 "sacrebleu>=1.5.0" pytablewriter "peft>=0.2.0" "accelerate>=0.29.1"
pip install jsonlines "evaluate>=0.4.0" numexpr dotenv

hf download MMMU/MMMU --repo-type dataset
