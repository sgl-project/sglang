# Adapter from https://github.com/vllm-project/vllm-metal/blob/a06cd65a35b5c61c9a7f9d5f5ae00b30d9603379/vllm_metal/metal/__init__.py
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType

from sglang.srt.hardware_backend.mlx.attention.build import build, PARTITION_SIZE

logger = logging.getLogger(__name__)

_ops_module = None
_THIS_DIR = Path(__file__).resolve().parent
_METAL_DIR = _THIS_DIR / "metal_kernels"

def _read_metal(filename: str) -> str:
    return (_METAL_DIR / filename).read_text(encoding="utf-8")

def _get_utils_src() -> str:
    utils_src = _read_metal("utils.metal")
    # Comment out float8.metal include if present as it might be missing
    utils_src = utils_src.replace('#include "float8.metal"', '// #include "float8.metal"')
    # Add definition of VLLM_METAL_PARTITION_SIZE that used to be passed by compiler flag
    utils_src = f"#define VLLM_METAL_PARTITION_SIZE {PARTITION_SIZE}\n" + utils_src
    return utils_src

def _build_reshape_cache_source() -> str:
    return _get_utils_src() + "\n" + _read_metal("reshape_and_cache.metal").replace('#include "utils.metal"', '')

def _build_paged_attention_source() -> str:
    return _get_utils_src() + "\n" + _read_metal("pagedattention.metal").replace('#include "utils.metal"', '')

def _build_v2_paged_attention_source() -> str:
    return _get_utils_src() + "\n" + _read_metal("pagedattention.metal").replace('#include "utils.metal"', '')

def get_ops() -> ModuleType:
    """JIT-build and import the native paged_ops extension.

    The Metal shader sources are read, pre-processed (includes inlined),
    and passed to the C++ extension which JIT-compiles them via
    ``mlx::core::metal::Device::get_library()``.

    Returns:
        The ``_paged_ops`` module with ``reshape_and_cache()`` and
        ``paged_attention_v1()``.
    """
    global _ops_module
    if _ops_module is not None:
        return _ops_module

    # 1. JIT-build the C++ extension if needed
    so_path = build()

    # 2. Import the built extension
    spec = importlib.util.spec_from_file_location("_paged_ops", str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_paged_ops"] = mod
    spec.loader.exec_module(mod)

    # 3. Initialise Metal libraries (JIT-compile shaders)
    reshape_src = _build_reshape_cache_source()
    paged_attn_src = _build_paged_attention_source()
    mod.init_libraries(reshape_src, paged_attn_src)

    # 4. Initialise v2 library (online softmax kernel)
    v2_src = _build_v2_paged_attention_source()
    mod.init_v2_library(v2_src)

    _ops_module = mod
    logger.info("Native paged-attention Metal kernels loaded")
    return mod
