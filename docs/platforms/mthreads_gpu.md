# Moore Threads GPUs

This document describes how run SGLang on Moore Threads GPUs. If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# Install sglang python package
cd ..
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```
