"""Build-and-load shim for the ``mem_cache`` Rust extension.

Importing this module loads the compiled extension, building it with cargo on
first use (a Rust toolchain is required); libtorch comes from the installed
torch package.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

# libtorch must be resident before the extension's cdylib loads.
import torch

_CRATE_DIR = Path(__file__).resolve().parents[5] / "rust" / "mem-cache"
_BUILT_SO = _CRATE_DIR / "target" / "release" / "libmem_cache.so"
_PACKAGED_SO = Path(__file__).parent / "mem_cache.so"


def _build() -> None:
    torch_dir = Path(torch.__file__).parent
    env = dict(os.environ)
    env["LIBTORCH"] = str(torch_dir)
    env["PYO3_PYTHON"] = sys.executable
    # torch-sys discovers python include dirs via `python3` on PATH.
    env["PATH"] = f"{Path(sys.executable).parent}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = (
        f"{torch_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    )
    # The cdylib finds libtorch at runtime through this rpath.
    env["RUSTFLAGS"] = (
        f"{env.get('RUSTFLAGS', '')} -C link-arg=-Wl,-rpath,{torch_dir / 'lib'}"
    ).strip()
    subprocess.run(
        ["cargo", "build", "--release"], cwd=_CRATE_DIR, env=env, check=True
    )
    shutil.copy2(_BUILT_SO, _PACKAGED_SO)


def _load():
    if not _PACKAGED_SO.exists():
        _build()
    spec = importlib.util.spec_from_file_location("mem_cache", _PACKAGED_SO)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mem_cache"] = module
    spec.loader.exec_module(module)
    return module


_ext = _load()

DecLockRefParamsBinding = _ext.DecLockRefParamsBinding
InsertParamsBinding = _ext.InsertParamsBinding
MatchParamsBinding = _ext.MatchParamsBinding
RustBigramUnifiedTreeCoreBinding = _ext.RustBigramUnifiedTreeCoreBinding
RustUnifiedTreeCoreBinding = _ext.RustUnifiedTreeCoreBinding
TreeCoreInitParamsBinding = _ext.TreeCoreInitParamsBinding
