"""Small-batch Qwen3.8 NVFP4 MoE decode specialized for SM70."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from sglang.kernels.sm70_paths import sm70_csrc
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# The .cu lives in the shared JIT csrc tree, not beside this wrapper: the
# wrapper moved to kernels/ops/moe/ while the sources moved to
# kernels/jit/csrc/ (RFC #29630).
_SRC_PATH = sm70_csrc("sm70_nvfp4_moe_decode.cu")
_EXT = None
_LOAD_ATTEMPTED = False


def sm70_nvfp4_moe_decode_enabled() -> bool:
    return os.environ.get("SGLANG_V100_NVFP4_MOE_DECODE", "1") == "1"


def sm70_nvfp4_moe_decode_max_rows() -> int:
    """Row cap the SM70 NVFP4 decode/top-k kernels may dispatch from Python.

    The kernels themselves are row-independent (grid = rows * topk), so this
    only keeps dispatch inside the row counts that were measured for each
    profile: 4 by default, 8 for the native MTP chain, then 16/32 for wide
    batched verification.
    """
    if envs.SGLANG_SM70_BATCHED_MOE32.get():
        return 32
    if envs.SGLANG_SM70_BATCHED_MOE.get():
        return 16
    if envs.SGLANG_SM70_MTP_WIDE_MOE.get():
        return 8
    return 4


def sm70_nvfp4_moe_decode_available() -> bool:
    if not sm70_nvfp4_moe_decode_enabled() or not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() == (7, 0)
    except (AssertionError, RuntimeError):
        return False


def _load_sm70_nvfp4_moe_decode_ops():
    global _EXT, _LOAD_ATTEMPTED
    if _EXT is not None:
        return _EXT
    if _LOAD_ATTEMPTED:
        return None
    _LOAD_ATTEMPTED = True
    if not _SRC_PATH.is_file():
        logger.warning("SM70 NVFP4 MoE decode source not found: %s", _SRC_PATH)
        return None

    from torch.utils.cpp_extension import load_inline

    build_directory = os.environ.get(
        "SGLANG_V100_NVFP4_MOE_BUILD_DIR", "/tmp/sglang_sm70_nvfp4_moe"
    )
    os.makedirs(build_directory, exist_ok=True)
    try:
        _EXT = load_inline(
            name="sglang_sm70_nvfp4_moe_v100",
            cpp_sources="",
            cuda_sources=_SRC_PATH.read_text(),
            functions=None,
            is_python_module=True,
            verbose=False,
            build_directory=build_directory,
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
    except Exception:  # pragma: no cover - build failures are environment-specific
        logger.exception("SM70 NVFP4 MoE decode extension failed to build")
        return None
    logger.info_once("SM70 (V100): small-batch NVFP4 MoE decode kernel loaded.")
    return _EXT


def sm70_nvfp4_moe_decode(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_scales: torch.Tensor,
    w13_global: torch.Tensor,
    w2_global: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Run the exact TP4 M<=4/H=2560/I=160/top-k=10 decode shape.

    Rows above four are dispatched only when the environment opts in (see
    sm70_nvfp4_moe_decode_max_rows), up to the 32-row kernel cap.
    """
    ext = _load_sm70_nvfp4_moe_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 NVFP4 MoE decode extension is unavailable")
    batch_size = hidden_states.shape[0]
    num_routes = batch_size * 10
    gate_up_partials = torch.empty(
        (40, num_routes, 320), dtype=torch.float32, device=hidden_states.device
    )
    activated = torch.empty(
        (num_routes, 160), dtype=torch.float16, device=hidden_states.device
    )
    down_partials = torch.empty(
        (num_routes, 5, 2560), dtype=torch.float32, device=hidden_states.device
    )
    output = torch.empty_like(hidden_states)
    ext.decode(
        hidden_states,
        w13,
        w2,
        w13_scales,
        w2_scales,
        w13_global,
        w2_global,
        topk_ids,
        topk_weights,
        gate_up_partials,
        activated,
        down_partials,
        output,
    )
    return output


def sm70_topk10_softmax(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select and renormalize ten of 512 logits for up to 32 rows."""
    ext = _load_sm70_nvfp4_moe_decode_ops()
    if ext is None:
        raise RuntimeError("SM70 NVFP4 decode extension is unavailable")
    weights = torch.empty(
        (logits.shape[0], 10), dtype=torch.float32, device=logits.device
    )
    ids = torch.empty(
        (logits.shape[0], 10), dtype=torch.int32, device=logits.device
    )
    ext.topk10_softmax(logits, weights, ids)
    return weights, ids
