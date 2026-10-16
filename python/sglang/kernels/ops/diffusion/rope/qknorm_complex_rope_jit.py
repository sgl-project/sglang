# SPDX-License-Identifier: Apache-2.0
"""Bit-exact CUDA RMSNorm + complex RoPE for Qwen-Image-2.1 attention prep.

One launch replaces the Triton ``qknorm_complex_rope`` (Q) and
``qknorm_complex_rope_kv`` (K, V, prefix) pair: it normalizes and rotates the Q
rows in place, writes the normalized/rotated K rows into rows ``[P:P+S]`` of a
``[B, P+S, H, 128]`` buffer (in place when the projection already wrote them
there), copies the cached prefix K/V into rows ``[0:P]`` and, only when asked,
the raw V rows into ``[P:P+S]``.

Numerical contract (bf16 in and out): fp32 squares summed in aten's vectorized
128-wide mean order (lane ``t`` owns elements ``4t..4t+3`` left to right, then
a 32-lane shuffle tree), ``rsqrtf(var + eps)``, bf16 rounding after the
normalize and after the weight multiply (``RMSNorm(cast_x_before_out_mul=True)``),
then explicit-FMA rotation in the orientation ``_fuse_real_sin`` probes for the
running GPU. Verified ``torch.equal`` against the eager chain on the production
shape ``[1, 4096, 32, 128]`` with an 18-token prefix and on ``[1, 18, 32, 128]``.
CUDA only: the ROCm reduction tree differs. Callers verify once through a
``BitExactFusionGate`` and keep the eager chain as the fallback.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.diffusion.rope.complex_rope_triton import _fuse_real_sin
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_HEAD_DIM = 128


@cache_once
def _jit_module(fuse_real_sin: bool, copy_v: bool) -> Module:
    args = make_cpp_args(fuse_real_sin, copy_v)
    return load_jit(
        "qknorm_complex_rope",
        *args,
        cuda_files=["diffusion/qknorm_complex_rope.cuh"],
        cuda_wrappers=[
            ("norm_rope", f"qknorm_complex_rope::Kernel<{args}>::run"),
            ("norm_rope_pack", f"qknorm_complex_rope::Kernel<{args}>::run_pack"),
        ],
    )


@torch.compiler.assume_constant_result
@cache_once
def can_use_qknorm_complex_rope_cuda(dtype: torch.dtype, head_dim: int) -> bool:
    """The CUDA reduction is specialized to BF16, 128-wide heads."""
    return dtype is torch.bfloat16 and head_dim == _HEAD_DIM


def _rope_real(rope: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(rope)


def _fake_norm_rope(x, weight, rope, eps):
    return torch.empty(x.shape, dtype=x.dtype, device=x.device)


@register_custom_op(
    op_name="qknorm_complex_rope_cuda",
    mutates_args=[],
    fake_impl=_fake_norm_rope,
)
def qknorm_complex_rope_cuda(
    x: torch.Tensor, weight: torch.Tensor, rope: torch.Tensor, eps: float
) -> torch.Tensor:
    """``rope(rmsnorm(x))`` for one ``[B, S, H, 128]`` tensor with packed head rows, out of place."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    module = _jit_module(_fuse_real_sin(x.device), False)
    with torch.cuda.device(x.device):
        module.norm_rope(x, out, weight, _rope_real(rope), float(eps))
    return out


@register_custom_op(
    op_name="qknorm_complex_rope_pack_",
    mutates_args=["q", "k_out", "v_out"],
)
def qknorm_complex_rope_pack_(
    q: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    rope: torch.Tensor,
    k_prefix: torch.Tensor,
    v_prefix: torch.Tensor,
    k_src: Optional[torch.Tensor],
    v_src: Optional[torch.Tensor],
    eps: float,
) -> None:
    """Normalize/rotate Q in place and K into ``k_out[:, P:]``; fill the prefix rows of ``k_out``/``v_out``.

    ``k_src=None`` normalizes ``k_out[:, P:]`` in place (the projection wrote
    the raw K there). ``v_src=None`` leaves ``v_out[:, P:]`` untouched (the
    projection wrote V there); otherwise ``v_src`` is copied into it.
    """
    prefix = k_prefix.shape[1]
    token_rows = k_out[:, prefix:]
    copy_v = v_src is not None
    module = _jit_module(_fuse_real_sin(q.device), copy_v)
    with torch.cuda.device(q.device):
        module.norm_rope_pack(
            q,
            q_weight,
            token_rows if k_src is None else k_src,
            k_out,
            k_weight,
            v_out,
            v_src if copy_v else token_rows,
            _rope_real(rope),
            k_prefix,
            v_prefix,
            float(eps),
        )


__all__ = [
    "can_use_qknorm_complex_rope_cuda",
    "qknorm_complex_rope_cuda",
    "qknorm_complex_rope_pack_",
]
