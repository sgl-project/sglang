"""Import SGLang-vendored or upstream AITER Kimi-K3 FlyDSL operators."""

import importlib


def load_module(local_module: str, aiter_module: str, *, source: str):
    """Import ``local_module`` for the "sglang" source, ``aiter_module`` for "aiter"."""
    if source not in ("sglang", "aiter"):
        raise ValueError(f"Unknown Kimi-K3 FlyDSL source {source!r}")
    return importlib.import_module(local_module if source == "sglang" else aiter_module)
