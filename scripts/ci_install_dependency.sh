# Install the dependency in CI.

# Use repo from environment variable, passed from GitHub Actions
# FLASHINFER_REPO="${FLASHINFER_REPO:-https://flashinfer.ai/whl/cu121/torch2.4}"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
bash "${SCRIPT_DIR}/killall_sglang.sh"

pip install --upgrade pip
# pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

# Force reinstall flashinfer
# pip install flashinfer -i ${FLASHINFER_REPO} --force-reinstall
pip install https://github.com/flashinfer-ai/flashinfer-nightly/releases/download/0.1.6%2B6819a0f/flashinfer-0.1.6+6819a0f.cu121torch2.4-cp310-cp310-linux_x86_64.whl  --force-reinstall

pip install transformers==4.45.2 sentence_transformers accelerate peft

# For compling eagle kernels
pip install cutex

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12
