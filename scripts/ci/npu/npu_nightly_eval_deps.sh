#!/bin/bash
# Eval packages and data the nightly NPU tests use on top of the sglang install.
set -euo pipefail

# Use latex2sympy2-extended (the fork sgl-eval depends on) instead of the legacy
# latex2sympy2 1.9.1, which pins antlr4-python3-runtime==4.7.2 and breaks the
# antlr>=4.9.3 that math-verify / sgl-eval requires.
pip install sentence_transformers zss "wandb>=0.16.0" tenacity==8.3.0 loguru openpyxl latex2sympy2-extended zstandard transformers-stream-generator tqdm-multiprocess pycocoevalcap
pip install yt-dlp sentencepiece==0.1.99 nltk av ftfy sqlitedict==2.1.0 "sacrebleu>=1.5.0" pytablewriter "peft>=0.2.0" "accelerate>=0.29.1"
pip install jsonlines "evaluate>=0.4.0" numexpr dotenv

hf download MMMU/MMMU --repo-type dataset
