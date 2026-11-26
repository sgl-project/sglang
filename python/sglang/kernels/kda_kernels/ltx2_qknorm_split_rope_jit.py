from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.kda_kernels import _cuda_source
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_ltx2_qknorm_split_rope_module(round_intermediates: bool = False) -> Module:
    sm90 = current_arch().is_sm(9, 0)
    namespace = "ltx2_qknorm_split_rope_kda" if sm90 else "ltx2_qknorm_split_rope"
    return load_jit(
        "diffusion_ltx2_qknorm_split_rope",
        cuda_files=[
            _cuda_source(
                "diffusion/ltx2_qknorm_split_rope_sm90.cuh"
                if sm90
                else "diffusion/ltx2_qknorm_split_rope.cuh"
            )
        ],
        extra_cuda_cflags=["-DKDA_SPLIT_THRESHOLD=1000000000"] if sm90 else [],
        cuda_wrappers=[
            (
                "ltx2_qknorm_split_rope_pair",
                (
                    f"{namespace}::LTX2QKNormSplitRopeKernel::run"
                    f"<{str(round_intermediates).lower()}>"
                ),
            )
        ],
    )


def _fake_impl(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
    round_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(q, dtype=torch.bfloat16), torch.empty_like(
        k, dtype=torch.bfloat16
    )


def _ltx2_qknorm_split_rope_into(
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
    round_intermediates: bool,
) -> None:
    _jit_ltx2_qknorm_split_rope_module(round_intermediates).ltx2_qknorm_split_rope_pair(
        q_out,
        k_out,
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        float(eps),
        int(num_heads),
        int(head_dim),
    )


@register_custom_op(
    op_name="diffusion_ltx2_qknorm_split_rope",
    mutates_args=[],
    fake_impl=_fake_impl,
)
def _ltx2_qknorm_split_rope_custom_op(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    num_heads: int,
    head_dim: int,
    round_intermediates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_out = torch.empty_like(q, dtype=torch.bfloat16)
    k_out = torch.empty_like(k, dtype=torch.bfloat16)
    _ltx2_qknorm_split_rope_into(
        q_out,
        k_out,
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        float(eps),
        int(num_heads),
        int(head_dim),
        round_intermediates,
    )
    return q_out, k_out


def _supported_side(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
) -> bool:
    return (
        x.is_cuda
        and cos.is_cuda
        and sin.is_cuda
        and weight.is_cuda
        and x.device == cos.device == sin.device == weight.device
        and x.dtype == torch.bfloat16
        and cos.dtype == torch.bfloat16
        and sin.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.ndim == 3
        and cos.ndim == 4
        and sin.ndim == 4
        and x.is_contiguous()
        and cos.shape == sin.shape
        and cos.shape[0] == x.shape[0]
        and cos.shape[1] == num_heads
        and cos.shape[2] == x.shape[1]
        and cos.shape[3] * 2 == head_dim
        and x.shape[2] == num_heads * head_dim
        and x.shape[2] == weight.shape[0]
        and weight.ndim == 1
        and weight.is_contiguous()
        and head_dim % 2 == 0
        and x.shape[2] % 4 == 0
        and cos.stride(-1) == 1
        and sin.stride(-1) == 1
    )


def _is_sm100_or_newer(x: torch.Tensor) -> bool:
    if not x.is_cuda:
        return False
    try:
        return torch.cuda.get_device_capability(x.device)[0] >= 10
    except RuntimeError:
        return False


def _is_sm90(x: torch.Tensor) -> bool:
    return x.is_cuda and current_arch().is_sm(9, 0)


def _validate_output_aliases(
    out: tuple[torch.Tensor, torch.Tensor], inputs: tuple[torch.Tensor, ...]
) -> None:
    for output in out:
        if not output.is_contiguous():
            raise RuntimeError("LTX2 QKNorm split-RoPE outputs must be contiguous")
    input_storages = {
        value.untyped_storage().data_ptr() for value in inputs if value.numel() != 0
    }
    output_storages = set()
    for output in out:
        if output.numel() == 0:
            continue
        storage = output.untyped_storage().data_ptr()
        if storage in input_storages or storage in output_storages:
            raise RuntimeError(
                "LTX2 QKNorm split-RoPE outputs must not alias inputs or each other"
            )
        output_storages.add(storage)


def can_use_ltx2_qknorm_split_rope_cuda(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    allow_sm90: bool = False,
    backend: str = "kda",
) -> bool:
    if backend == "vibecuda":
        from sglang.kernels.ops.diffusion.rope.ltx2_qknorm_split_rope_vibecuda_jit import (
            can_use_ltx2_qknorm_split_rope_vibecuda,
        )

        return can_use_ltx2_qknorm_split_rope_vibecuda(
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            k_cos,
            k_sin,
            k_weight,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    if backend != "kda":
        raise ValueError(f"unknown LTX2 QKNorm split-RoPE backend: {backend!r}")
    return (
        (_is_sm100_or_newer(q) or (allow_sm90 and _is_sm90(q)))
        and _supported_side(
            q,
            q_cos,
            q_sin,
            q_weight,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        and _supported_side(
            k,
            k_cos,
            k_sin,
            k_weight,
            num_heads=num_heads,
            head_dim=head_dim,
        )
    )


def ltx2_qknorm_split_rope_cuda(
    q: torch.Tensor,
    q_cos: torch.Tensor,
    q_sin: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor,
    k_cos: torch.Tensor,
    k_sin: torch.Tensor,
    k_weight: torch.Tensor,
    *,
    eps: float,
    num_heads: int,
    head_dim: int,
    allow_sm90: bool = False,
    backend: str = "kda",
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "vibecuda":
        from sglang.kernels.ops.diffusion.rope.ltx2_qknorm_split_rope_vibecuda_jit import (
            ltx2_qknorm_split_rope_vibecuda,
        )

        return ltx2_qknorm_split_rope_vibecuda(
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            k_cos,
            k_sin,
            k_weight,
            eps=eps,
            num_heads=num_heads,
            head_dim=head_dim,
            out=out,
        )
    if out is not None:
        _validate_output_aliases(
            out, (q, q_cos, q_sin, q_weight, k, k_cos, k_sin, k_weight)
        )
    if not can_use_ltx2_qknorm_split_rope_cuda(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        num_heads=num_heads,
        head_dim=head_dim,
        allow_sm90=allow_sm90,
        backend=backend,
    ):
        raise RuntimeError("unsupported input for LTX2 QKNorm split-RoPE CUDA")
    round_intermediates = allow_sm90 and _is_sm90(q)
    if out is not None:
        if torch.compiler.is_compiling():
            raise RuntimeError("out= is supported for eager and CUDA Graph calls only")
        _ltx2_qknorm_split_rope_into(
            *out,
            q,
            q_cos,
            q_sin,
            q_weight,
            k,
            k_cos,
            k_sin,
            k_weight,
            float(eps),
            int(num_heads),
            int(head_dim),
            round_intermediates,
        )
        return out
    return _ltx2_qknorm_split_rope_custom_op(
        q,
        q_cos,
        q_sin,
        q_weight,
        k,
        k_cos,
        k_sin,
        k_weight,
        float(eps),
        int(num_heads),
        int(head_dim),
        round_intermediates=allow_sm90 and _is_sm90(q),
    )
