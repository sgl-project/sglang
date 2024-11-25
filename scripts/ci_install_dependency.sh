"""
Install the dependency in CI.
"""

pip install --upgrade pip
pip install -e "python[all]"
pip install transformers==4.45.2 sentence_transformers accelerate peft
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/ --force-reinstall
# for compling eagle kernels
pip install cutex
# for compling xgrammar kernels
pip install cuda-python nvidia-cuda-nvrtc-cu12
