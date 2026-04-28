# Apple Silicon with Metal

This document describes how run SGLang on Apple Silicon using [Metal](https://developer.apple.com/metal/). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Use the default branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install sglang python package
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```
