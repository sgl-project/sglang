# MUSA Quickstart for SGLang-Diffusion

This page covers installation and usage of SGLang-Diffusion on Moore Threads GPU (MTGPU) with the MUSA software stack.

## Install from Source

```bash
# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

## Quick Test

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```
