#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
# Import the real checkout through PYTHONPATH; never build SGLang/Rust/kernels.
python3 -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0
python3 - "$ROOT/python/pyproject_other.toml" <<'PY'
import importlib.metadata
import subprocess
import sys
import tomllib
with open(sys.argv[1], 'rb') as f:
    project = tomllib.load(f)['project']
extras = project['optional-dependencies']
requirements = project['dependencies'] + extras['runtime_base']
requirements += [r for r in extras['runtime_common'] if not r.startswith('sglang[')]
requirements += ['pytest==8.3.5', 'PyYAML==6.0.3', 'jsonschema', 'tabulate', 'expecttest', 'dill', 'accelerate']
# Real pool imports define Triton kernels even on CPU; install a wheel only.
requirements += ['triton==3.6.0; sys_platform == "linux"']
# Exact local versions prevent transitive dependencies replacing CPU Torch wheels.
requirements += [f'{name}=={importlib.metadata.version(name)}' for name in ('torch', 'torchvision', 'torchaudio')]
# Binary-only also prevents accidental extension source builds in the CPU lane.
subprocess.run([sys.executable, '-m', 'pip', 'install', '--only-binary=:all:', *requirements], check=True)
PY
