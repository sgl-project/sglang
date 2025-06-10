#!/bin/bash
set -euo pipefail

# Install the required dependencies in CI.
docker exec ci_sglang pip install --upgrade pip
docker exec ci_sglang pip uninstall sgl-kernel -y || true
docker exec -w /sglang-checkout/sgl-kernel ci_sglang bash -c "rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && python3 setup_rocm.py install"
docker exec ci_sglang pip install -e "python[dev_hip]"

docker exec -w / ci_sglang git clone https://github.com/merrymercy/human-eval.git
docker exec -w /human-eval ci_sglang pip install -e .

# For lmms_evals evaluating MMMU
docker exec -w / ci_sglang git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
docker exec -w /lmms-eval ci_sglang pip install -e .

docker exec -w / ci_sglang mkdir -p /dummy-grok
mkdir -p dummy-grok && wget https://sharkpublic.blob.core.windows.net/sharkpublic/sglang/dummy_grok.json -O dummy-grok/config.json
docker cp ./dummy-grok ci_sglang:/

docker exec ci_sglang pip install huggingface_hub[hf_xet]
