from typing import Optional

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[triton.Config({"BT": bt}) for bt in (32, 64, 128)],
    key=["H", "D"],
)
@triton.jit
def _kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    lower_bound: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """0728-compatible FP32 KDA gate activation for Ascend verify."""
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    neg_exp_a = -tl.exp(tl.load(A + i_h).to(tl.float32))
    stride_row = H * D
    gate_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, 1),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    out_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, 1),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )
    gate = tl.load(gate_ptr, boundary_check=(0, 1)).to(tl.float32)
    if HAS_BIAS:
        offsets = tl.arange(0, BD)
        bias = tl.load(
            g_bias + i_h * D + offsets,
            mask=offsets < D,
            other=0.0,
        ).to(tl.float32)
        gate += bias[None, :]

    if USE_LOWER_BOUND:
        activated = lower_bound * tl.sigmoid(-neg_exp_a * gate)
    else:
        scaled = gate * beta
        softplus = tl.where(
            scaled > threshold,
            gate,
            tl.log(1.0 + tl.exp(scaled)) / beta,
        )
        activated = neg_exp_a * softplus
    tl.store(out_ptr, activated.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate_npu(
    gate: torch.Tensor,
    A_log: torch.Tensor,
    head_dim: int,
    *,
    gate_bias: Optional[torch.Tensor] = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    lower_bound: Optional[float] = None,
) -> torch.Tensor:
    """Apply the 0728 model-side KDA gate contract without touching GPU code."""
    original_shape = gate.shape[:-1]
    gate_2d = gate.reshape(-1, gate.shape[-1])
    tokens, hidden = gate_2d.shape
    heads = A_log.numel()
    if heads * head_dim != hidden:
        raise ValueError("KDA gate shape does not match A_log and head_dim")
    if gate_bias is not None and gate_bias.numel() != hidden:
        raise ValueError("KDA gate bias shape does not match the gate")

    output = torch.empty_like(gate_2d, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(tokens, meta["BT"]), heads)
    _kda_gate_fwd_kernel[grid](
        gate_2d,
        A_log,
        output,
        gate_bias,
        beta,
        threshold,
        lower_bound,
        tokens,
        heads,
        head_dim,
        BD=triton.next_power_of_2(head_dim),
        HAS_BIAS=gate_bias is not None,
        USE_LOWER_BOUND=lower_bound is not None,
    )
    return output.view(*original_shape, heads, head_dim)
