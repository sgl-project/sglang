"""Batch-1..4 NVFP4 GEMV for SM70 Marlin weights with N a multiple of 256.

Replaces the grouped Marlin GEMM on GLM-5.3 dense MLP and shared-expert
decode. Prefill (M > 4) stays on Marlin.
"""

from __future__ import annotations

import logging
import os

import torch

from sglang.kernels.sm70_paths import sm70_csrc
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_SRC_PATH = sm70_csrc("sm70_glm_nvfp4_gemv.cu")
_EXT = None
_LOAD_ATTEMPTED = False
_PACKED: dict[tuple, torch.Tensor] = {}


def sm70_glm_nvfp4_gemv_available() -> bool:
    if not envs.SGLANG_SM70_GLM_NVFP4_GEMV.get():
        return False
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() == (7, 0)
    except (AssertionError, RuntimeError):
        return False


def _load():
    global _EXT, _LOAD_ATTEMPTED
    if _EXT is not None:
        return _EXT
    if _LOAD_ATTEMPTED:
        return None
    _LOAD_ATTEMPTED = True
    if not _SRC_PATH.is_file():
        logger.warning("GLM NVFP4 GEMV source missing: %s", _SRC_PATH)
        return None
    from torch.utils.cpp_extension import load_inline

    build_directory = os.path.expanduser(envs.SGLANG_SM70_GLM_NVFP4_BUILD_DIR.get())
    os.makedirs(build_directory, exist_ok=True)
    try:
        _EXT = load_inline(
            name="sglang_sm70_glm_nvfp4_gemv",
            cpp_sources="",
            cuda_sources=_SRC_PATH.read_text(),
            functions=None,
            is_python_module=True,
            verbose=False,
            build_directory=build_directory,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    except Exception:
        logger.exception("GLM NVFP4 GEMV extension failed to build")
        return None
    logger.info("SM70 (V100): GLM NVFP4 GEMV kernel loaded.")
    return _EXT


def choose_split(groups: int, qwords: int) -> int:
    """Enough blocks to cover the SMs, and a divisor of the K-group count."""
    target_threads = 320 * 128
    split = min(groups, max(1, target_threads // max(qwords, 1)))
    while split > 1 and groups % split != 0:
        split -= 1
    return split


def prepack_glm_nvfp4_weight(weight: torch.Tensor) -> torch.Tensor | None:
    """Pack a Marlin NVFP4 matrix once, before CUDA graph capture allocates."""
    ext = _load()
    if ext is None or not weight.is_cuda:
        return None
    key = (int(weight.data_ptr()), int(weight.numel()), weight.device.index)
    packed = _PACKED.get(key)
    if packed is None:
        packed = ext.repack(weight if weight.is_contiguous() else weight.contiguous())
        _PACKED[key] = packed
    return packed


def sm70_glm_nvfp4_gemv(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor | None:
    ext = _load()
    if ext is None:
        return None
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    n = int(scales.shape[1])
    output = torch.empty(
        (hidden_states.shape[0], n),
        dtype=torch.float16,
        device=hidden_states.device,
    )
    global_scale = global_scale.reshape(-1)[:1].contiguous()
    if global_scale.dtype != torch.float32:
        global_scale = global_scale.float()
    packed = prepack_glm_nvfp4_weight(weight)
    if packed is None:
        return None
    ext.gemv_hmma_splitk(hidden_states, packed, scales, global_scale, output, 0)
    return output
