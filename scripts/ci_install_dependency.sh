"""
Install the dependency in CI.
"""

./killall_sglang.sh

pip install --upgrade pip
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/

# Force reinstall flashinfer
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ --force-reinstall

pip install transformers==4.45.2 sentence_transformers accelerate peft

# For compling eagle kernels
pip install cutex

# For compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12
