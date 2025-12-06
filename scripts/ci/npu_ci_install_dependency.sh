#!/bin/bash
set -euo pipefail

PIP_INSTALL="python3 -m pip install --no-cache-dir"

python3 -m pip uninstall sglang
### Install SGLang
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
${PIP_INSTALL} -v -e "python[srt_npu]"
