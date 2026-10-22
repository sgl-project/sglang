"""Shared helpers for PPU-owned registered tests."""

import hashlib
import importlib.util
import sys
from pathlib import Path


def load_legacy_ppu_module(relative):
    """Load a checkout test without confusing the two directories named ppu."""
    root = Path(__file__).resolve().parents[2] / "ppu"
    source = (root / relative).resolve()
    source.relative_to(root)
    name = "_ppu_pr_legacy_" + hashlib.sha256(str(source).encode()).hexdigest()
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load PPU test module: {source}")
    module = importlib.util.module_from_spec(spec)
    # Dataclasses and pytest introspection require a registered module identity.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
