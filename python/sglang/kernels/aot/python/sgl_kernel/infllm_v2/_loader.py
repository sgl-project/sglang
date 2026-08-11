"""Robust loader for the standalone ``infllm_ops`` pybind extension.

The InfLLM-V2 FlashAttention backend is built as its own module ``infllm_ops``
(installed into the ``sgl_kernel`` package directory). Under editable installs
the compiled ``.so`` may live in ``site-packages/sgl_kernel`` while the imported
``sgl_kernel`` package resolves to the source tree, so a plain ``from sgl_kernel
import infllm_ops`` is not always sufficient. This loader searches the known
candidate locations and loads the extension by file path.
"""

import glob
import importlib.util
import site
import sys
from pathlib import Path
from typing import List, Optional

_infllm_ops = None


def _candidate_dirs() -> List[Path]:
    dirs: List[Path] = []

    # 1) The directory of the sgl_kernel package as currently imported.
    try:
        import sgl_kernel

        dirs.append(Path(sgl_kernel.__file__).parent)
    except Exception:
        pass

    # 2) This module's parent package directory (source tree).
    dirs.append(Path(__file__).resolve().parent.parent)

    # 3) Every ``sgl_kernel`` directory found on the install paths.
    search_roots: List[str] = []
    try:
        search_roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        search_roots.append(site.getusersitepackages())
    except Exception:
        pass
    search_roots.extend(p for p in sys.path if p)
    for root in search_roots:
        dirs.append(Path(root) / "sgl_kernel")

    # De-duplicate while preserving order.
    seen = set()
    unique: List[Path] = []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _find_so() -> Optional[Path]:
    for d in _candidate_dirs():
        if not d.is_dir():
            continue
        matches = sorted(glob.glob(str(d / "infllm_ops*.so")))
        if matches:
            return Path(matches[0])
    return None


def load_infllm_ops():
    """Import and return the ``infllm_ops`` extension module (cached)."""
    global _infllm_ops
    if _infllm_ops is not None:
        return _infllm_ops

    # Fast path: a normal import may already work.
    try:
        from sgl_kernel import infllm_ops as _mod  # type: ignore

        _infllm_ops = _mod
        return _infllm_ops
    except Exception:
        pass

    so_path = _find_so()
    if so_path is None:
        raise ImportError(
            "[sgl_kernel] Could not locate the 'infllm_ops' extension (infllm_ops*.so). "
            "Ensure sgl-kernel was built with the InfLLM-V2 FlashAttention backend."
        )

    spec = importlib.util.spec_from_file_location("infllm_ops", str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"[sgl_kernel] Could not create module spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _infllm_ops = module
    return _infllm_ops
