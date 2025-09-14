# Moore Threads GPUs

This document describes how run SGLang on Moore Threads GPUs. If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_musa.py install

# Install sglang python package
cd ..
pip install -e "python[all_musa]"
```