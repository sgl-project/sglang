"""
Install the dependency in CI.
"""

./killall_sglang.sh

pip install --upgrade pip
pip install -e "python[all]" --find-links https://flashinfer.ai/whl/cu121/torch2.4/flashinfer/
pip install transformers==4.45.2 sentence_transformers accelerate peft
# for compling eagle kernels
pip install cutex
# for compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12
