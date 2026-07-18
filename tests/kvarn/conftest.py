# SPDX-License-Identifier: Apache-2.0
"""Test bootstrap for KVarN unit tests.

These tests only need torch + triton.  The kvarn modules under test have
no sglang framework imports, so we create synthetic namespace packages
for the sglang parent chain (which normally has heavy __init__ side
effects) and let Python's normal import mechanism handle the kvarn
sub-packages themselves.
"""

import sys
import types
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent / "python"

# Add python/ to sys.path so Python can find leaf sub-packages.
sys.path.insert(0, str(_PKG_ROOT))

# Shim the *parent* packages that have heavy __init__.py side effects.
# The leaf kvarn/kvarn_ops packages have lightweight __init__.py and will be
# imported normally by Python's import machinery.
_chain = [
    "sglang",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.quantization",
    "sglang.srt.layers.attention",
]

for _pkg in _chain:
    if _pkg in sys.modules:
        continue
    _mod = types.ModuleType(_pkg)
    _dir = _PKG_ROOT / _pkg.replace(".", "/")
    _mod.__path__ = [str(_dir)]
    _mod.__package__ = _pkg
    sys.modules[_pkg] = _mod
    if "." in _pkg:
        _parent, _child = _pkg.rsplit(".", 1)
        setattr(sys.modules[_parent], _child, _mod)
