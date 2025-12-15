from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.utils import direct_register_custom_op, is_hip

_is_hip = is_hip()


fused_softcap_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 128}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 256}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 512}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 32768}, num_warps=32),
    ],
    key=["n_ele"],
)


@triton.jit
def fused_softcap_kernel(
    output_ptr,
    input_ptr,
    n_ele,
    softcap_const: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_ele
    x = tl.load(input_ptr + offsets, mask=mask)
    fx = x.to(tl.float32)
    fxs = fx / softcap_const
    exped = tl.exp(2 * fxs)
    top = exped - 1
    bottom = exped + 1
    output = top / bottom * softcap_const
    tl.store(output_ptr + offsets, output, mask=mask)


fused_softcap_kernel_autotuned = fused_softcap_autotune(fused_softcap_kernel)


def fused_softcap(x, softcap_const, autotune=False):
    output = torch.empty_like(x, dtype=torch.float32)
    n_elements = output.numel()
    if autotune:
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        fused_softcap_kernel_autotuned[grid](output, x, n_elements, softcap_const)
    else:
        fused_softcap_kernel[(triton.cdiv(n_elements, 128),)](
            output, x, n_elements, softcap_const, BLOCK_SIZE=128, num_warps=8
        )
    return output


# cast to float + softcap
class Softcap:
    def __init__(self, softcap_const: float):
        self.softcap_const = softcap_const

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            return self.forward_cuda(x)
        else:
            return self.forward_native(x)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x.float() / self.softcap_const) * self.softcap_const

    def forward_cuda(self, x: torch.Tensor, autotune=False) -> torch.Tensor:
        return fused_softcap(x, self.softcap_const, autotune=autotune)


rmsnorm_autotune = triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=4, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=8, num_stages=8),
        triton.Config(kwargs={"BLOCK_SIZE": 1024}, num_warps=16, num_stages=8),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 2048}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 8192}, num_warps=32, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=1),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=8, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=16, num_stages=4),
        triton.Config(kwargs={"BLOCK_SIZE": 16384}, num_warps=32, num_stages=4),
    ],
    key=["hidden_dim"],
)


@triton.jit
def fused_dual_residual_rmsnorm_kernel(
    output_ptr,
    mid_ptr,
    activ_ptr,
    residual_ptr,
    weight1_ptr,
    weight2_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    r = tl.load(residual_ptr + input_start + offsets, mask=mask, other=0.0)
    w1_ = tl.load(weight1_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a2r = r + (a / rms * w1).to(r.dtype)
    tl.store(
        mid_ptr + input_start + offsets,
        a2r,
        mask=mask,
    )

    a2r = a2r.to(tl.float32)
    rms2 = tl.sqrt(tl.sum(a2r * a2r, axis=0) / hidden_dim + eps)

    w2_ = tl.load(weight2_ptr + offsets, mask=mask, other=0.0)
    w2 = w2_.to(tl.float32)

    tl.store(
        output_ptr + input_start + offsets,
        a2r / rms2 * w2,  # implicitly casts to output dtype here
        mask=mask,
    )


fused_dual_residual_rmsnorm_kernel_autotune = rmsnorm_autotune(
    fused_dual_residual_rmsnorm_kernel
)


def fused_dual_residual_rmsnorm(x, residual, weight1, weight2, eps, autotune=False):
    assert len(x.shape) == 2
    assert (
        x.shape == residual.shape and x.dtype == residual.dtype
    ), f"{x.shape=} {residual.shape=} {x.dtype=} {residual.dtype=}"
    output, mid = torch.empty_like(x), torch.empty_like(x)
    bs, hidden_dim = x.shape
    if autotune:
        fused_dual_residual_rmsnorm_kernel_autotune[(bs,)](
            output, mid, x, residual, weight1, weight2, eps=eps, hidden_dim=hidden_dim
        )
    else:
        max_warps = 16 if _is_hip else 32
        config = {
            "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
            "num_warps": max(
                min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
            ),
        }

        fused_dual_residual_rmsnorm_kernel[(bs,)](
            output,
            mid,
            x,
            residual,
            weight1,
            weight2,
            eps=eps,
            hidden_dim=hidden_dim,
            **config,
        )

    return output, mid


@triton.jit
def fused_rmsnorm_kernel(
    output_ptr,
    activ_ptr,
    weight_ptr,
    eps: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    input_start = pid * hidden_dim

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim

    a_ = tl.load(activ_ptr + input_start + offsets, mask=mask, other=0.0)
    a = a_.to(tl.float32)
    rms = tl.sqrt(tl.sum(a * a, axis=0) / hidden_dim + eps)

    w1_ = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    w1 = w1_.to(tl.float32)

    a_rms = a / rms * w1

    tl.store(
        output_ptr + input_start + offsets,
        a_rms,  # implicitly casts to output dtype here
        mask=mask,
    )


def fused_rmsnorm(x, weight, eps, autotune=False, inplace=False):
    assert len(x.shape) == 2
    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    bs, hidden_dim = x.shape
    max_warps = 16 if _is_hip else 32
    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 256)), max_warps), 4
        ),
    }

    fused_rmsnorm_kernel[(bs,)](
        output, x, weight, eps=eps, hidden_dim=hidden_dim, **config
    )
    return output


class FusedDualResidualRMSNorm:
    """
    Fused implementation of
    y = RMSNorm2(RMSNorm1(x) + residual))
    """

    def __init__(self, rmsnorm1, rmsnorm2) -> None:  # the one after rmsnorm1
        self.rmsnorm1 = rmsnorm1
        self.rmsnorm2 = rmsnorm2
        self.variance_epsilon = self.rmsnorm1.variance_epsilon
        assert self.rmsnorm1.variance_epsilon == self.rmsnorm2.variance_epsilon
        assert self.rmsnorm1.weight.shape == self.rmsnorm2.weight.shape

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.is_cuda:
            return self.forward_cuda(x, residual)
        else:
            return self.forward_flashinfer(x, residual)

    def forward_cuda(
        self, x: torch.Tensor, residual: torch.Tensor, autotune=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return fused_dual_residual_rmsnorm(
            x,
            residual,
            self.rmsnorm1.weight,
            self.rmsnorm2.weight,
            self.variance_epsilon,
            autotune=autotune,
        )

    def forward_flashinfer(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed1 = self.rmsnorm1(x)
        residual = normed1 + residual
        return self.rmsnorm2(residual), residual

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        normed1 = self.rmsnorm1.forward_native(x)
        residual = normed1 + residual
        return self.rmsnorm2.forward_native(residual), residual


@triton.jit
def experts_combine_kernel(
    out_hidden_states,
    moe_hidden_states,
    mlp_hidden_states,
    combine_k: tl.constexpr,
    hidden_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_index_mlp = pid * hidden_dim
    start_index_rmoe = pid * hidden_dim * combine_k
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_dim
    combine_k_offsets = tl.arange(0, combine_k)

    moe_x = tl.load(
        moe_hidden_states
        + start_index_rmoe
        + combine_k_offsets[:, None] * hidden_dim
        + offsets[None, :],
        mask=mask[None, :],
        other=0.0,
    )
    moe_x = tl.sum(moe_x, axis=0)
    mlp_x = tl.load(mlp_hidden_states + start_index_mlp + offsets, mask=mask, other=0.0)
    combined_x = (moe_x + mlp_x) / 1.4142135623730951

    tl.store(out_hidden_states + start_index_mlp + offsets, combined_x, mask=mask)


def experts_combine_triton(
    moe_hidden_states: torch.Tensor,
    mlp_hidden_states: torch.Tensor,
    output_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert moe_hidden_states.is_contiguous()
    assert mlp_hidden_states.is_contiguous()

    if len(moe_hidden_states.shape) == 2:
        combine_k = 1  # pre-combined
    else:
        combine_k = moe_hidden_states.shape[1]

    if output_buffer is None:
        out_hidden_states = torch.empty_like(mlp_hidden_states)
    else:
        flat_output_buffer = output_buffer.view(mlp_hidden_states.dtype).reshape(-1)
        assert flat_output_buffer.numel() >= mlp_hidden_states.numel()
        out_hidden_states = flat_output_buffer[: mlp_hidden_states.numel()].reshape(
            mlp_hidden_states.shape
        )

    bs, hidden_dim = mlp_hidden_states.shape

    config = {
        "BLOCK_SIZE": triton.next_power_of_2(hidden_dim),
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 1024)), 8), 4
        ),
    }

    experts_combine_kernel[(bs,)](
        out_hidden_states,
        moe_hidden_states,
        mlp_hidden_states,
        combine_k,
        hidden_dim,
        **config,
    )

    return out_hidden_states


def experts_combine_triton_fake(
    moe_hidden_states: torch.Tensor,
    mlp_hidden_states: torch.Tensor,
    output_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(mlp_hidden_states)


direct_register_custom_op(
    op_name="experts_combine_triton",
    op_func=experts_combine_triton,
    mutates_args=[],
    fake_impl=experts_combine_triton_fake,
)


# gelu on first half of vector
@triton.jit
def gelu_and_mul_kernel(
    out_hidden_states_ptr,  # (bs, hidden_dim)
    out_scales_ptr,  # (bs,)
    hidden_states_ptr,  # (bs, hidden_dim * 2)
    quant_max: tl.constexpr,
    static_scale: tl.constexpr,
    hidden_dim: tl.constexpr,  # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # gelu
    # cast down before mul to better match training?
    gelu_x1 = 0.5 * (1.0 + tl.erf(x1 * 0.7071067811865475)) * x1
    out = x3 * gelu_x1.to(hidden_states_ptr.dtype.element_ty)

    if quant_max is not None:
        raise NotImplementedError()

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


def gelu_and_mul_triton(
    hidden_states,
    scales=None,
    quantize=None,  # dtype to quantize to
    out=None,
):
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2

    if out is None:
        out_hidden_states = torch.empty(
            (bs, hidden_dim),
            dtype=quantize or hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        assert out.shape == (bs, hidden_dim)
        assert out.dtype == (quantize or hidden_states.dtype)
        out_hidden_states = out
    out_scales = None
    static_scale = False
    if quantize is not None:
        if scales is None:
            out_scales = torch.empty(
                (bs,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            out_scales = scales
            static_scale = True

    max_warps = 16 if _is_hip else 32
    config = {
        # 8 ele per thread (not tuned)
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),
    }

    gelu_and_mul_kernel[(bs,)](
        out_hidden_states,
        out_scales,
        hidden_states,
        quant_max=torch.finfo(quantize).max if quantize is not None else None,
        static_scale=static_scale,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        **config,
    )

    if quantize is not None:
        return out_hidden_states, out_scales
    else:
        return out_hidden_states, None


# silu on first half of vector
@triton.jit
def silu_and_mul_kernel(
    out_hidden_states_ptr,  # (bs, hidden_dim)
    out_scales_ptr,  # (bs,)
    hidden_states_ptr,  # (bs, hidden_dim * 2)
    quant_max: tl.constexpr,
    static_scale: tl.constexpr,
    hidden_dim: tl.constexpr,  # the output hidden_dim
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    input_start = pid * hidden_dim * 2
    output_start = pid * hidden_dim

    input1_offs = tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < hidden_dim  # shared for input1, input3, output
    input3_offs = hidden_dim + tl.arange(0, BLOCK_SIZE)
    output_offs = tl.arange(0, BLOCK_SIZE)

    x1 = tl.load(
        hidden_states_ptr + input_start + input1_offs, mask=mask, other=0.0
    ).to(tl.float32)
    x3 = tl.load(
        hidden_states_ptr + input_start + input3_offs, mask=mask, other=0.0
    ).to(tl.float32)

    # silu
    # cast down before mul to better match training?
    silu_x1 = x1 * tl.sigmoid(x1)
    out = x3 * silu_x1.to(hidden_states_ptr.dtype.element_ty)

    if quant_max is not None:
        raise NotImplementedError()

    tl.store(out_hidden_states_ptr + output_start + output_offs, out, mask=mask)


def silu_and_mul_triton(
    hidden_states,
    scales=None,
    quantize=None,  # dtype to quantize to
    out=None,
):
    bs, in_hidden_dim = hidden_states.shape
    hidden_dim = in_hidden_dim // 2

    if out is None:
        out_hidden_states = torch.empty(
            (bs, hidden_dim),
            dtype=quantize or hidden_states.dtype,
            device=hidden_states.device,
        )
    else:
        assert out.shape == (bs, hidden_dim)
        assert out.dtype == (quantize or hidden_states.dtype)
        out_hidden_states = out
    out_scales = None
    static_scale = False
    if quantize is not None:
        if scales is None:
            out_scales = torch.empty(
                (bs,), dtype=torch.float32, device=hidden_states.device
            )
        else:
            out_scales = scales
            static_scale = True

    max_warps = 16 if _is_hip else 32
    config = {
        # 8 ele per thread (not tuned)
        "num_warps": max(
            min(triton.next_power_of_2(triton.cdiv(hidden_dim, 8 * 32)), max_warps), 4
        ),
    }

    silu_and_mul_kernel[(bs,)](
        out_hidden_states,
        out_scales,
        hidden_states,
        quant_max=torch.finfo(quantize).max if quantize is not None else None,
        static_scale=static_scale,
        hidden_dim=hidden_dim,
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
        **config,
    )

    if quantize is not None:
        return out_hidden_states, out_scales
    else:
        return out_hidden_states, None
