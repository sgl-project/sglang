# Copyright (c) 2026 LightSeek Foundation / SGLang team.
# Adapted from tokenspeed-kernel fused_topk_topp (MIT).
"""JIT fused top-k + top-p probability renormalization."""

from __future__ import annotations

from pathlib import Path

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

_CSRC_SUBDIR = Path(__file__).resolve().parent / "csrc" / "sampling" / "fused_topk_topp"

# Persistent per-device side streams for overlapping top-p radix with top-k.
# Must be created outside CUDA graph capture (cudaStreamCreate is illegal
# inside capture).
_side_streams: dict[torch.device, torch.cuda.Stream] = {}


@cache_once
def _jit_module():
    return load_jit(
        "fused_topk_topp",
        cuda_files=[
            "sampling/fused_topk_topp/fused_topk_topp.cu",
            "sampling/fused_topk_topp/fused_topk_topp_binding.cu",
        ],
        extra_include_paths=[str(_CSRC_SUBDIR)],
        extra_cuda_cflags=["--expt-extended-lambda"],
        header_only=False,
    )


def prepare_for_device(device: torch.device | str | int) -> None:
    """Pre-create the side stream used by ``fused_topk_topp_renorm``."""
    device = torch.device(device)
    if device.type != "cuda":
        return
    if device not in _side_streams:
        _side_streams[device] = torch.cuda.Stream(device=device)


def _side_stream_handle(device: torch.device) -> int:
    stream = _side_streams.get(device)
    if stream is not None:
        return int(stream.cuda_stream)
    if torch.cuda.is_current_stream_capturing():
        return 0
    prepare_for_device(device)
    return int(_side_streams[device].cuda_stream)


def fused_topk_topp_workspace_size(batch_size: int, vocab_size: int) -> int:
    return int(
        _jit_module().fused_topk_topp_workspace_size(int(batch_size), int(vocab_size))
    )


def fused_topk_topp_renorm(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    workspace: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused top-k + top-p probability renormalization.

    Matches ``top_k_renorm_prob`` followed by deterministic ``top_p_renorm_prob``.
    Finite ``top_k`` values must be ``<= 128``; use ``1 << 30`` for top-p-only.
    """
    probs = probs.float().contiguous()
    top_ks = top_ks.to(dtype=torch.int32, device=probs.device).contiguous()
    top_ps = top_ps.to(dtype=torch.float32, device=probs.device).contiguous()
    if out is None:
        out = torch.empty_like(probs)
    else:
        out = out.float().contiguous()
    if workspace is None:
        ws_bytes = fused_topk_topp_workspace_size(probs.size(0), probs.size(1))
        workspace = torch.empty(ws_bytes, dtype=torch.uint8, device=probs.device)
    side_handle = _side_stream_handle(probs.device)
    _jit_module().fused_topk_topp_renorm(
        probs, top_ks, top_ps, out, workspace, side_handle
    )
    return out


def is_fused_topk_topp_available() -> bool:
    """True when the JIT module can be loaded on this platform."""
    if not torch.cuda.is_available():
        return False
    try:
        _jit_module()
        return True
    except Exception:
        return False
