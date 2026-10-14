# SPDX-License-Identifier: Apache-2.0
"""Packed Step5 QKV -> normalized/rotated Q/K and unchanged V."""

import math
from typing import Optional

import torch

_QKV_WIDTHS = (1920, 3456)
_QKV_GATE_WIDTHS = (1928, 3472)


def step5_qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_cache: torch.Tensor,
    *,
    eps: float,
    kv_write: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source CuTe math for local Q8 or Q16/KV1/D192, with FP32 zero-centered weights.

    QKV is contiguous BF16 [T,(Qheads+2)*192]. The model-owned table is contiguous BF16
    [capacity,64 or 192], merged [cos,sin]. This entry creates views, never a
    copied or recomputed table. Positions are int64 [T] within table capacity;
    the serving owner checks that value bound before device execution.

    Norm materializes BF16 before FP32 RoPE and its final BF16 store. Returns
    independent contiguous BF16 Q[T,Qheads,192], K/V[T,1,192]. Inputs are read-only;
    output allocation belongs to the current forward/graph memory lifetime.

    ``kv_write`` is ``(k_cache, v_cache, loc)`` when the SWA paged store is fused:
    K/V cache are contiguous BF16 ``[slots, 192]`` and ``loc`` is int64 ``[T]``.
    Values written at ``cache[loc]`` match the K/V outputs bit-exactly; ``loc < 0``
    skips the row. Shape or dtype mismatch raises; there is no silent fallback.
    """
    if qkv.ndim != 2 or qkv.shape[1] not in _QKV_WIDTHS:
        raise ValueError("Step5 QKV must have shape [tokens,1920 or 3456]")
    q, k, v, _ = _qknorm_rope(
        qkv,
        q_weight,
        k_weight,
        positions,
        rotary_cache,
        heads=qkv.shape[1] // 192 - 2,
        gated=False,
        eps=eps,
        kv_write=kv_write,
    )
    return q, k, v


def step5_qknorm_rope_gated(
    qkv_gate: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_cache: torch.Tensor,
    *,
    eps: float,
    kv_write: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Same kernel over a fused [QKV | head gate] row of width (Qheads+2)*192 + Qheads.

    The gate columns follow V in each row and come back as a contiguous BF16
    [T, Qheads] copy; everything else matches ``step5_qknorm_rope``.
    """
    if qkv_gate.ndim != 2 or qkv_gate.shape[1] not in _QKV_GATE_WIDTHS:
        raise ValueError("Step5 QKV with gate must have shape [tokens,1928 or 3472]")
    return _qknorm_rope(
        qkv_gate,
        q_weight,
        k_weight,
        positions,
        rotary_cache,
        heads=(qkv_gate.shape[1] - 384) // 193,
        gated=True,
        eps=eps,
        kv_write=kv_write,
    )


def _qknorm_rope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    rotary_cache: torch.Tensor,
    *,
    heads: int,
    gated: bool,
    eps: float,
    kv_write: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if qkv.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("Step5 Q/K norm requires NVIDIA CUDA")
    if rotary_cache.ndim != 2 or rotary_cache.shape[1] not in (64, 192):
        raise ValueError("Step5 main rotary cache must have width 64 or 192")
    if rotary_cache.shape[0] == 0 or not math.isfinite(eps) or eps <= 0:
        raise ValueError(
            "Step5 Q/K norm needs nonempty rotary capacity and positive eps"
        )
    tokens = qkv.shape[0]
    row_width = (heads + 2) * 192 + (heads if gated else 0)
    for name, tensor, shape, dtype, alignment in (
        ("QKV", qkv, (tokens, row_width), torch.bfloat16, 16),
        ("Q weight", q_weight, (192,), torch.float32, 16),
        ("K weight", k_weight, (192,), torch.float32, 16),
        ("positions", positions, (tokens,), torch.int64, 8),
        ("rotary cache", rotary_cache, tuple(rotary_cache.shape), torch.bfloat16, 16),
    ):
        if (
            tensor.device != qkv.device
            or tensor.dtype != dtype
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.data_ptr() % alignment
        ):
            raise ValueError(f"Step5 Q/K norm invalid {name}")
    k_cache = v_cache = loc = None
    store_cache = kv_write is not None
    if store_cache:
        if len(kv_write) != 3:
            raise ValueError("Step5 Q/K norm kv_write must be (k_cache, v_cache, loc)")
        k_cache, v_cache, loc = kv_write
        if k_cache.ndim != 2 or k_cache.shape[1] != 192:
            raise ValueError("Step5 Q/K norm K cache must have shape [slots, 192]")
        slots = k_cache.shape[0]
        for name, tensor, shape, dtype, alignment in (
            ("K cache", k_cache, (slots, 192), torch.bfloat16, 16),
            ("V cache", v_cache, (slots, 192), torch.bfloat16, 16),
            ("cache loc", loc, (tokens,), torch.int64, 8),
        ):
            if (
                tensor.device != qkv.device
                or tensor.dtype != dtype
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
                or tensor.data_ptr() % alignment
            ):
                raise ValueError(f"Step5 Q/K norm invalid {name}")
    q = qkv.new_empty((tokens, heads * 192))
    k = qkv.new_empty((tokens, 192))
    v = qkv.new_empty((tokens, 192))
    gate = qkv.new_empty((tokens, heads)) if gated else None
    if tokens:
        from sglang.kernels.ops.attention.step5.cute.cute_utils import device_cache_key
        from sglang.kernels.ops.attention.step5.cute.qknorm_dispatch import (
            compile_qknorm,
        )

        width = rotary_cache.shape[1]
        cos, sin = rotary_cache.split(width // 2, dim=-1)
        compile_qknorm(
            width,
            device_cache_key(qkv.device),
            logical_q_heads=heads,
            store_cache=store_cache,
            gate=gated,
        )(
            qkv,
            q_weight,
            k_weight,
            cos,
            sin,
            positions,
            q,
            k,
            v,
            gate,
            k_cache,
            v_cache,
            loc,
            192,
            heads,
            1,
            eps,
            1.0,
        )
    return (
        q.view(tokens, heads, 192),
        k.view(tokens, 1, 192),
        v.view(tokens, 1, 192),
        gate,
    )
