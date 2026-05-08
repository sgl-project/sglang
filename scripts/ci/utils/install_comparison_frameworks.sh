#!/usr/bin/env bash
set -euxo pipefail

FRAMEWORK="${1:?usage: install_comparison_frameworks.sh <vllm-omni|lightx2v>}"
VENV_ROOT="${SGLANG_DIFFUSION_FRAMEWORK_VENV_ROOT:-/tmp/sglang-diffusion-framework-venvs}"
VENV_PATH="${VENV_ROOT}/${FRAMEWORK}"

mkdir -p "${VENV_ROOT}"
python3 -m venv --clear "${VENV_PATH}"
# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

python3 -m pip install --upgrade pip wheel setuptools

case "${FRAMEWORK}" in
  vllm-omni)
    python3 -m pip install --upgrade --force-reinstall "${VLLM_INSTALL_SPEC:-vllm==0.18.0}"
    constraints="${VENV_PATH}/vllm_omni_constraints.txt"
    python3 -m pip freeze \
      | grep -E '^(llvmlite|numba|numpy|setuptools|tokenizers|torch|torchaudio|torchvision|transformers|triton)==' \
      > "${constraints}"
    python3 -m pip install --upgrade --force-reinstall -c "${constraints}" "${VLLM_OMNI_INSTALL_SPEC:-vllm-omni==0.18.0}"
    ;;
  lightx2v)
    python3 -m pip install --upgrade --force-reinstall "${LIGHTX2V_INSTALL_SPEC:-git+https://github.com/ModelTC/LightX2V.git}"
    python3 -m pip install --upgrade --force-reinstall pyzmq
    ;;
  *)
    echo "Unknown comparison framework: ${FRAMEWORK}" >&2
    exit 1
    ;;
esac
