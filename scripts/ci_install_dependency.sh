"""
Install the dependency in CI.
"""

pip install --upgrade pip
pip install -e "python[all]"
pip install transformers==4.45.2 sentence_transformers accelerate peft
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer/python
pip install -e .
