"""TP all-reduce + residual add + RMSNorm on the CustomAllReduceV2 push plane.

One kernel per boundary, with the MoE finalize and the shared-expert add folded
in when the runner hands back a MoeDeferredFinalize: at the FFN input (the
attention output's sum), at the next attention input (the FFN output's sum or
finalize handoff), and through install_terminal_finalize at the final norm. The plane
is the one ``register_fused_all_reduce_comm`` registered before graph capture.
"""

from __future__ import annotations

from functools import partial
from typing import Optional

import torch

from sglang.kernels.ops.communication.all_reduce_fusion import (
    all_reduce_residual_norm,
    moe_finalize_all_reduce_residual_norm,
)
from sglang.srt.layers.layer_boundary import (
    FfnInputFusion,
    SumGroup,
    TokenAxis,
    UnreducedOutput,
    get_attn_tp_context,
)
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import MoeDeferredFinalize
from sglang.srt.layers.layer_boundary.fusions.norm import fused_norm_gamma
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
    can_fuse_all_reduce,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_exec, get_parallel


def _bf16_norm_gamma(norm: torch.nn.Module) -> Optional[torch.Tensor]:
    gamma = fused_norm_gamma(norm)
    return gamma if gamma is not None and gamma.dtype == torch.bfloat16 else None


class CustomAllReduceFusion:
    """One layer's CustomAllReduceV2 fusion kernels, given to both of its stages. The model
    gates the static configuration by constructing it; per batch it takes
    decode and target-verify rows that fit the registered plane.
    ``defers_finalize``: the layer's MoE can hand off its finalize.
    ``final_norm_eps``: set by install_terminal_finalize when the model's final
    norm takes a handoff, so the last layer may defer too."""

    def __init__(self, *, hidden_size: int, defers_finalize: bool) -> None:
        self.hidden_size = int(hidden_size)
        self.defers_finalize = defers_finalize
        self.final_norm_eps: Optional[float] = None

    def attn_input_fusions(self, plan) -> tuple:
        return (
            partial(self._finalize_add_norm, plan),
            partial(self._all_reduce_add_norm, plan),
        )

    def ffn_input_fusions(self, plan) -> tuple:
        parallel = get_parallel()
        if (
            TokenAxis.ATTN_TP in plan.incoming_residual_rows.sharded
            # The plane sums over TP, the attention-TP group only without attention DP.
            or parallel.attn_tp_size != parallel.tp_size
            or _bf16_norm_gamma(plan.norm) is None
        ):
            return ()
        return (
            FfnInputFusion(
                completes=SumGroup.ATTN_TP,
                run=partial(self._ffn_input_add_norm, plan),
            ),
        )

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        return (
            self.defers_finalize
            and (not plan.terminal or self.final_norm_eps is not None)
            and self._eligible(plan, forward_batch, forward_batch.input_ids.shape[0])
        )

    def finalize(
        self, handoff: MoeDeferredFinalize, residual: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The final norm's finalize + all-reduce + residual add + norm."""
        return self._run_finalize(handoff, residual, gamma, self.final_norm_eps)

    def _finalize_add_norm(
        self, plan, owed, residual, forward_batch, post_residual_addition
    ):
        if not isinstance(owed, MoeDeferredFinalize):
            return None
        gamma = _bf16_norm_gamma(plan.norm)
        if (
            gamma is None
            or residual is None
            or any(
                tensor is not None and tensor.dtype != torch.bfloat16
                for tensor in (
                    owed.routed_output,
                    owed.gated_shared_output,
                    residual,
                )
            )
            or not self._eligible(plan, forward_batch, owed.m)
        ):
            return None
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        return self._run_finalize(owed, residual, gamma, plan.norm.variance_epsilon)

    def _all_reduce_add_norm(
        self, plan, owed, residual, forward_batch, post_residual_addition
    ):
        if not isinstance(owed, UnreducedOutput) or owed.group is None:
            return None
        gamma = _bf16_norm_gamma(plan.norm)
        if (
            gamma is None
            or residual is None
            or residual.dtype != torch.bfloat16
            or owed.partial.dtype != torch.bfloat16
            or owed.group.ranks != get_parallel().tp_group.ranks
            or not self._eligible(plan, forward_batch, int(owed.partial.shape[0]))
        ):
            return None
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        return all_reduce_residual_norm(
            owed.partial,
            residual,
            gamma,
            plan.norm.variance_epsilon,
            world_size=get_parallel().tp_size,
            residual_out=torch.empty_like(residual),
        )

    def _ffn_input_add_norm(self, plan, hidden_states, residual, forward_batch):
        gamma = _bf16_norm_gamma(plan.norm)
        if (
            gamma is None
            or residual is None
            or residual.dtype != torch.bfloat16
            or hidden_states.dtype != torch.bfloat16
            or not self._eligible(plan, forward_batch, int(hidden_states.shape[0]))
        ):
            return None
        return all_reduce_residual_norm(
            hidden_states,
            residual,
            gamma,
            plan.norm.variance_epsilon,
            world_size=get_parallel().tp_size,
            residual_out=torch.empty_like(residual),
        )

    def _run_finalize(self, handoff, residual, gamma, eps):
        return moe_finalize_all_reduce_residual_norm(
            handoff.routed_output,
            handoff.permuted_indices.reshape(-1),
            handoff.expert_weights,
            handoff.expert_weights.shape[1],
            handoff.gated_shared_output,
            residual,
            gamma,
            eps,
            world_size=get_parallel().tp_size,
            hidden_dim=self.hidden_size,
            residual_out=torch.empty_like(residual),
        )

    def _eligible(self, plan, forward_batch: ForwardBatch, m: int) -> bool:
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        mode = forward_batch.forward_mode
        return bool(
            (mode.is_decode() or mode.is_target_verify())
            # Gate the producer too: an unsupported norm must not request a
            # deferred finalize that the terminal consumer cannot fuse.
            and _bf16_norm_gamma(plan.norm) is not None
            and not forward_batch.can_run_tbo
            and not is_in_breakable_cuda_graph()
            # The norm's reduction order changes with the launch geometry.
            and not is_batch_invariant_mode_enabled()
            and not get_attn_tp_context().input_scattered
            and not get_exec().comm.enable_quant_communications
            and TokenAxis.ATTN_TP not in plan.fused_input_rows(forward_batch).sharded
            and can_fuse_all_reduce(m, self.hidden_size)
        )


def install_terminal_finalize(
    layers, final_norm: torch.nn.Module
) -> Optional[CustomAllReduceFusion]:
    """Let the last of ``layers`` hand its finalize to ``final_norm``; returns
    the adapter to pass as ``final_norm(finalize_norm=...)``, or None.
    final_norm() reads gemma_weight, so only a GemmaRMSNorm takes it."""
    fusion = getattr(layers[-1].__dict__.get("ffn_boundary"), "fusions", None)
    if not isinstance(fusion, CustomAllReduceFusion) or not isinstance(
        final_norm, GemmaRMSNorm
    ):
        return None
    if _bf16_norm_gamma(final_norm) is None:
        return None
    fusion.final_norm_eps = final_norm.variance_epsilon
    return fusion
