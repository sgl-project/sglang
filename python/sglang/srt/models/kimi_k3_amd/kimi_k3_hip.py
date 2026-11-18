"""gfx950 glue of kimi_k3: the branches the model takes under ``_is_gfx95``. Its dense BF16
GEMMs that bypass UnquantizedLinearMethod go through AITER's tuned table (or hipBLASLt
where it is faster) and AITER's tuned FlyDSL kernel into caller storage; the gfx950 MoE
layer lives in ``moe``."""

from typing import Optional

import torch

from sglang.srt.layers.attn_residual import AttnResidual, aggregate_stream
from sglang.srt.layers.layer_boundary.residual.attn_bank import _bank_rows
from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import gather_row_shards
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.unquant_hip import (
    a16w16_into,
    a16w16_tgemm,
    a16w16_tuned,
)


def k3_bf16_gemm_hip(
    x: torch.Tensor, weight: torch.Tensor, out: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    """_k3_bf16_gemm's ROCm routing: the shapes AITER's bf16 table tunes (and
    hipBLASLt does not run faster) through AITER, and an out= GEMM through AITER's
    tuned FlyDSL kernel; None leaves the GEMM to torch (hipBLASLt)."""
    if out is None:
        return a16w16_tgemm(x, weight)
    return out if a16w16_into(x, weight, out) else None


def kda_f_b_gemm_hip(f_a: torch.Tensor, f_b_weight: torch.Tensor) -> torch.Tensor:
    """KDA's f_b GEMM on f_a, a row slice of the fused in-proj output: AITER's tuned
    kernel reads it in place, fastest at M=8..128 on MI350X."""
    forget_gate = a16w16_tuned(f_a, f_b_weight)
    if forget_gate is None:
        from sglang.kernels.ops.gemm import kimi_k3_tiny_gemm

        forget_gate = kimi_k3_tiny_gemm(f_a, f_b_weight)
    return forget_gate


def mono_output_gate_hip(pre: dict, alt: Optional[torch.cuda.Stream]) -> torch.Tensor:
    """g_proj(x) from a monokernel plugin's projection front (``_mono_pre``),
    once the gate stream the layer left pending has finished."""
    if alt is not None:
        torch.cuda.current_stream().wait_stream(alt)
    return pre["gate"]


def aggregate_stream_hip(
    bank: AttnResidual,
    hidden_states: torch.Tensor,
    nvb: int,
    score_proj: ReplicatedLinear,
    score_norm: RMSNorm,
) -> torch.Tensor:
    """aggregate_stream of ``bank``'s stream on every row. Off a stream the row-sharded
    MoE tail left on this rank's rows, it aggregates those bank rows and gathers the
    mixture, keeping the bank sharded."""
    rows = _bank_rows(bank, hidden_states)
    mixed = aggregate_stream(
        hidden_states,
        bank.block_residual if rows is None else bank.block_residual[rows],
        nvb,
        score_proj,
        score_norm,
    )
    if rows is None:
        return mixed
    # the draft reads every row
    return gather_row_shards(mixed, bank.block_residual.shape[0])
