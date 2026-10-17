"""Fused SM100 DSv4.1 sparse decode attention (CUDA, cluster combine), small batch.

Emits the attention output in FlashMLA MegaAttention's fp8 format, the activation DeepGEMM
``fp8_einsum`` (wo_a) consumes: ``o_fp8 [B, 2, 4096]`` e4m3 in ``interleave_output_columns``
order and ``o_s [B, 2, 32]`` int32 packed UE8M0 per-32 scales, stored MN-major
``[2, 32, ceil4(B)]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HEAD_DIM = 512
NUM_HEADS = 16
GROUP_HEADS = 8
SOFTMAX_SCALE = HEAD_DIM**-0.5


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "dsv4_decode_attention_sm100",
        cuda_files=["deepseek_v4/low_latency/decode_attn.cuh"],
        cuda_wrappers=[("run", "dsv4_decode_attention_sm100")],
        extra_cuda_cflags=["-O3"],
    )


def alloc_mega_fp8_output(num_tokens: int, device: torch.device):
    """``(o_fp8, o_s)`` laid out like FlashMLA's ``allocate_scale_factor``."""
    groups = NUM_HEADS // GROUP_HEADS
    o_fp8 = torch.empty(
        (num_tokens, groups, GROUP_HEADS * HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    t_al = (num_tokens + 3) // 4 * 4
    o_s = torch.empty(
        (groups, GROUP_HEADS * HEAD_DIM // 128, t_al),
        dtype=torch.int32,
        device=device,
    )
    return o_fp8, o_s.permute(2, 0, 1)[:num_tokens]


def decode_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sink: torch.Tensor,
    extra_kv: Optional[torch.Tensor] = None,
    extra_indices: Optional[torch.Tensor] = None,
    extra_lengths: Optional[torch.Tensor] = None,
    *,
    inv_rope: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
    out: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(o_fp8, o_s)`` (see the module docstring).

    kv is the V4.1 fp8 SWA cache (528 B/token); extra_kv, if any, is V4.1 fp8 or fp4
    (288 B/token). inv_rope: ``(freqs [rows, >=64] f32 (cos, sin) pairs, positions [B]
    int32/int64)``, or positions None for a per-row ``[B, >=64]`` buffer; fuses the inverse
    RoPE of each head's last 64 dims. PDL dependents are released at kernel start, so a
    weight-streaming successor (wo_a) overlaps the attention. out: optional preallocated
    ``(o_fp8, o_s)`` of the shapes above, any strides (e.g. a ``[:, :2]`` slice of padded groups).
    """
    b = q.shape[0]
    if q.ndim == 4:
        assert q.shape[1] == 1
        q = q[:, 0]
    if indices.ndim == 3:
        indices = indices[:, 0]
    if extra_indices is not None and extra_indices.ndim == 3:
        extra_indices = extra_indices[:, 0]
    # [pages, page_size, bytes per token], whatever the cache's head dims
    kv = kv.view(torch.uint8).flatten(2)
    if extra_kv is not None:
        extra_kv = extra_kv.view(torch.uint8).flatten(2)
    o_fp8, o_s = alloc_mega_fp8_output(b, q.device) if out is None else out
    freqs, positions = (None, None) if inv_rope is None else inv_rope
    _jit_module().run(
        q,
        kv,
        indices,
        lengths,
        sink,
        o_fp8,
        o_s,
        SOFTMAX_SCALE,
        extra_kv,
        extra_indices,
        extra_lengths,
        freqs,
        positions,
    )
    return o_fp8, o_s
