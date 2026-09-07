"""Load the bundled Rust TreeCore extension or a fingerprinted local build."""

from pathlib import Path

# Loading torch first makes its libtorch dependencies resident before dlopen.
import torch

from sglang.srt.rust_extensions import load_rust_extension
from sglang.srt.rust_extensions.torch_build import torch_build_configuration

_PYTHON_MODULE = "sglang.srt.mem_cache.rust_tree_core.mem_cache"
_INSPECTION_MODULE = "sglang.srt.mem_cache.rust_tree_core.mem_cache_inspection"
_CRATE_DIR = Path(__file__).resolve().parents[5] / "rust" / "sglang-radix-tree"
_TORCH_COMPAT_HEADER = _CRATE_DIR / "torch_2_13_compat.h"


def load_tree_core_extension(*, inspection: bool = False):
    """Load the production binding or the test-only inspection variant."""
    build = torch_build_configuration(
        compat_header=_TORCH_COMPAT_HEADER,
        python_module=_PYTHON_MODULE,
        torch_module=torch,
    )
    return load_rust_extension(
        _PYTHON_MODULE,
        additional_features=("inspection",) if inspection else (),
        extension_module=_INSPECTION_MODULE if inspection else None,
        build_environment=build.environment,
        build_fingerprint=build.fingerprint,
    )


bindings = load_tree_core_extension()

DecLockRefParamsBinding = bindings.DecLockRefParamsBinding
InsertParamsBinding = bindings.InsertParamsBinding
MatchParamsBinding = bindings.MatchParamsBinding
RustBigramUnifiedTreeCoreBinding = bindings.RustBigramUnifiedTreeCoreBinding
RustUnifiedTreeCoreBinding = bindings.RustUnifiedTreeCoreBinding
TreeCoreInitParamsBinding = bindings.TreeCoreInitParamsBinding
