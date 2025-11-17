#!/bin/bash
set -euo pipefail
HOSTNAME_VALUE=$(hostname)
GPU_ARCH="mi30x"   # default

# Host names look like: linux-mi35x-gpu-1-xxxxx-runner-zzzzz
if [[ "${HOSTNAME_VALUE}" =~ ^linux-(mi[0-9]+[a-z]*)-gpu-[0-9]+ ]]; then
  GPU_ARCH="${BASH_REMATCH[1]}"
  echo "Detected GPU architecture from hostname: ${GPU_ARCH}"
else
  echo "Warning: could not parse GPU architecture from '${HOSTNAME_VALUE}', defaulting to ${GPU_ARCH}"
fi

# Install the required dependencies in CI.
docker exec ci_sglang pip install --upgrade pip
docker exec ci_sglang pip uninstall sgl-kernel -y || true
docker exec -w /sglang-checkout/sgl-kernel ci_sglang bash -c "rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && python3 setup_rocm.py install"

case "${GPU_ARCH}" in
  mi35x)
    echo "Runner uses ${GPU_ARCH}; will fetch mi35x image."
    docker exec ci_sglang rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
    docker exec ci_sglang pip install -e "python[dev_hip]" --no-deps # TODO: only for mi35x
    # For lmms_evals evaluating MMMU
    docker exec -w / ci_sglang git clone --branch v0.4.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    docker exec -w /lmms-eval ci_sglang pip install -e . --no-deps # TODO: only for mi35x
    ;;
  mi30x|mi300|mi325)
    echo "Runner uses ${GPU_ARCH}; will fetch mi30x image."
    docker exec ci_sglang rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
    docker exec ci_sglang pip install -e "python[dev_hip]"
    # For lmms_evals evaluating MMMU
    docker exec -w / ci_sglang git clone --branch v0.4.1 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    docker exec -w /lmms-eval ci_sglang pip install -e .
    ;;
  *)
    echo "Runner architecture '${GPU_ARCH}' unrecognised;" >&2
    ;;
esac

docker exec -w / ci_sglang git clone https://github.com/merrymercy/human-eval.git
docker exec -w /human-eval ci_sglang pip install -e .

docker exec -w / ci_sglang mkdir -p /dummy-grok
mkdir -p dummy-grok && wget https://sharkpublic.blob.core.windows.net/sharkpublic/sglang/dummy_grok.json -O dummy-grok/config.json
docker cp ./dummy-grok ci_sglang:/

docker exec ci_sglang pip install huggingface_hub[hf_xet]
docker exec ci_sglang pip install pytest
