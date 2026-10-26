"""TP4 all-reduce + residual add + Gemma RMSNorm + per-token fp8 quant in one kernel on
AITER's custom all-reduce (gfx950), at the decode boundaries of the smallest batches: the
FFN input (the attention output's sum) and the next attention input (the FFN output's
sum). Larger batches keep the all-reduce and the norm kernel."""

from __future__ import annotations

from functools import partial
from typing import Callable, Optional

import torch
from aiter.dist.device_communicators.custom_all_reduce import (
    CustomAllreduce as AiterCustomAllreduce,
)

from sglang.kernels.ops.communication.all_reduce_norm_quant_hip import (
    HIDDEN_SIZE,
    WORLD_SIZE,
    all_reduce_gemma_norm_fp8,
)
from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
from sglang.srt.layers.layer_boundary import (
    FfnInputFusion,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
    get_attn_tp_context,
)
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_exec, get_forward, get_parallel

# Every rank reads all ranks' rows, so past these custom AR + the norm kernel is as fast
# (TP4 on MI350X).
MAX_ROWS = 16


class AllReduceNormQuantFusion:
    """One layer's fused boundaries, given to both of its stages. Each stage's
    ``*_quant_format_for(rows)`` names the fp8 format its GEMM reads
    (fp8_input_quant_format); the kernel takes the per-token one."""

    def __init__(
        self,
        *,
        attn_quant_format_for: Callable[[int], str],
        ffn_quant_format_for: Callable[[int], str],
    ) -> None:
        self.attn_quant_format_for = attn_quant_format_for
        self.ffn_quant_format_for = ffn_quant_format_for

    def attn_input_fusions(self, plan) -> tuple:
        return (partial(self._attn_input, plan),)

    def ffn_input_fusions(self, plan) -> tuple:
        if TokenAxis.ATTN_TP in plan.incoming_residual_rows.sharded:
            return ()
        return (
            FfnInputFusion(
                completes=SumGroup.ATTN_TP, run=partial(self._ffn_input, plan)
            ),
        )

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def _attn_input(self, plan, owed, residual, forward_batch, post_residual_addition):
        if (
            not isinstance(owed, UnreducedOutput)
            or owed.group is None
            or owed.reduce_to_dp_local is not None
            or post_residual_addition is not None
        ):
            return None
        x = owed.partial
        return self._run(
            plan,
            x,
            residual,
            forward_batch,
            owed.group.ca_comm,
            self.attn_quant_format_for(int(x.shape[0])),
            keep_bf16=False,
        )

    def _ffn_input(self, plan, hidden_states, residual, forward_batch):
        # NormFp8Readout: the routed experts read the bf16 norm, the shared expert the fp8
        return self._run(
            plan,
            hidden_states,
            residual,
            forward_batch,
            get_parallel().attn_tp_group.ca_comm,
            self.ffn_quant_format_for(int(hidden_states.shape[0])),
            keep_bf16=True,
        )

    def _run(
        self, plan, x, residual, forward_batch, comm, quant_format, *, keep_bf16
    ) -> Optional[tuple]:
        if quant_format != "fp8_per_token" or not _eligible(
            plan, x, residual, forward_batch, comm
        ):
            return None
        return all_reduce_gemma_norm_fp8(
            x,
            residual,
            plan.norm.weight.data,
            plan.norm.variance_epsilon,
            comm,
            keep_bf16=keep_bf16,
        )


def _eligible(plan, x, residual, forward_batch: ForwardBatch, comm) -> bool:
    mode = forward_batch.forward_mode
    return bool(
        (mode.is_decode() or mode.is_target_verify())
        and isinstance(plan.norm, GemmaRMSNorm)
        and residual is not None
        and x.dim() == 2
        and 0 < x.shape[0] <= MAX_ROWS
        and x.shape[1] == HIDDEN_SIZE
        and x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and residual.shape == x.shape
        and x.is_contiguous()
        and residual.is_contiguous()
        and not forward_batch.can_run_tbo
        and not is_in_breakable_cuda_graph()
        # The norm's reduction order changes with the launch geometry.
        and not is_batch_invariant_mode_enabled()
        and not get_exec().deterministic.enable_deterministic_inference
        and not get_exec().comm.enable_quant_communications
        and not get_forward().sp_active
        and not get_attn_tp_context().input_scattered
        and TokenAxis.ATTN_TP not in plan.fused_input_rows(forward_batch).sharded
        # The communicator the unfused all-reduce takes for these rows; the kernel runs on
        # AITER's buffers, and SGLANG_USE_AITER_AR=0 / 1-stage AR dispatch sglang's own class.
        and isinstance(comm, AiterCustomAllreduce)
        and not comm.disabled
        and comm.enable_register_for_capturing
        and comm.world_size == WORLD_SIZE
        and comm.should_custom_ar(x)
    )
