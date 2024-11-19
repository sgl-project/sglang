"""
Install the dependency in CI.
"""

pip install --upgrade pip
pip install -e "python[all]"
pip install transformers==4.45.2 sentence_transformers accelerate peft
pip install https://github.com/flashinfer-ai/flashinfer-nightly/releases/download/0.1.6%2Bdd3c836/flashinfer-0.1.6+dd3c836.cu121torch2.4-cp310-cp310-linux_x86_64.whl --force-reinstall
pip install ninja
