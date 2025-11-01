# Moore Threads GPUs

This document describes how run SGLang on Moore Threads GPUs. If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Install sglang python package
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```
