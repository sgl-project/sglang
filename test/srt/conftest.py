"""pytest configuration for test/srt.

When tests are run from a sparse git checkout (e.g. during local development
where only ``python/sglang/srt/mem_cache/`` was fetched), ``sglang.lang`` and
other frontend modules may be missing or the installed sglang version may
differ from the fork being tested.

This conftest ensures the fork's ``python/`` directory takes precedence over
any installed sglang package, and stubs out missing frontend modules so that
tests focusing on the server-side runtime (``sglang.srt.*``) can run without
a full install.

In CI — where SGLang is installed from the complete source tree being tested —
the fork's python/ directory is already the installed package, so these stubs
are never needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

# ── Ensure the fork's python/ directory takes precedence ─────────────────────
_FORK_PYTHON = str(Path(__file__).parent.parent.parent / "python")

# Remove any previously-loaded sglang modules so the fork's versions are used.
for _key in list(sys.modules):
    if _key == "sglang" or _key.startswith("sglang."):
        del sys.modules[_key]

# Insert the fork at the very front of sys.path.
if _FORK_PYTHON in sys.path:
    sys.path.remove(_FORK_PYTHON)
sys.path.insert(0, _FORK_PYTHON)

# ── Stub out frontend modules missing from the sparse checkout ───────────────
_STUB_MODULES = [
    "sglang.lang",
    "sglang.lang.api",
    "sglang.lang.backend",
    "sglang.lang.backend.runtime_endpoint",
    "sglang.lang.backend.anthropic",
    "sglang.lang.backend.litellm",
    "sglang.lang.backend.openai",
    "sglang.lang.backend.vertexai",
    "sglang.lang.choices",
]
for _mod_name in _STUB_MODULES:
    sys.modules[_mod_name] = MagicMock()
