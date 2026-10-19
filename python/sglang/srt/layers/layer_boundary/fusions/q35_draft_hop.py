"""One-row MTP draft steps' all-reduce + residual + norm boundaries as q35 hops (SGLANG_Q35MF_MKD_HOP).

The draft keeps the bits of the kernels it replaces. At the FFN input the attention output's sum,
residual add and post-attention norm replace FlashInfer's legacy MNNVL fused kernel, which the draft
runs under SGLANG_Q35MF_DRAFT_MNNVL_AR. At the final norm the MoE output's one-shot all-reduce and
FlashInfer's CuTe DSL FusedAddRMSNorm become one hop: the terminal FFN hands its unreduced output on
as a DraftMoePartial, which the final norm passes to DraftHopFusion.finalize.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.flashinfer_comm_fusion import (
    resolve_flashinfer_allreduce_fusion_backend,
)
from sglang.srt.layers.layer_boundary import DeferredFinalize, FfnInputFusion, SumGroup
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# Draft steps carry one row; the verify and extend widths keep the stock kernels.
_MAX_M = 1


def moe_push_target():
    """The hop a deferring single-token draft MoE pushes its output into, or None to let the final
    norm's hop publish it (SGLANG_Q35MF_MKD_HOP_PUSH)."""
    if not envs.SGLANG_Q35MF_MKD_HOP_PUSH.get():
        return None
    from sglang.srt.layers.q35_hop import active

    return active()


class DraftMoePartial(DeferredFinalize, frozen=True):
    """A draft MoE output whose TP sum the final norm's hop completes; ``complete`` is the stock sum
    for any other reader."""

    partial: torch.Tensor
    reduce: Callable[[torch.Tensor], torch.Tensor]

    def complete(self) -> torch.Tensor:
        return self.reduce(self.partial)


class DraftHopFusion:
    """The draft layer's stage fusions, and the final norm's handoff adapter. The model sets
    ``final_norm`` and ``moe_ready`` once both exist."""

    def __init__(self) -> None:
        # Built inside the draft's speculative context, so this is the backend the draft's attention
        # output runs (SGLANG_Q35MF_DRAFT_MNNVL_AR turns cutedsl into auto, which is mnnvl here).
        backend = resolve_flashinfer_allreduce_fusion_backend()
        self._stock_matches = backend == "mnnvl"
        logger.info(
            "MKD draft hop fusion built: draft fusion backend %s, hops %s.",
            backend,
            "armed" if self._stock_matches else "off",
        )
        self.final_norm: Optional[GemmaRMSNorm] = None
        # Callable(num_tokens): whether the single-token draft MoE runs the batch and can hand its
        # unreduced output to finalize.
        self.moe_ready: Optional[Callable[[int], bool]] = None
        # Entry release is limited to dependents audited to read only static data before their PDL wait:
        # the single-launch draft MoE after the attention-output hop, the draft head after the final one.
        self.moe_single_launch: Optional[Callable[[int], bool]] = None
        self.head_follows: Optional[Callable[[], bool]] = None
        self._early = envs.SGLANG_Q35MF_MKD_HOP_EARLY.get()
        self._early_head = self._early and envs.SGLANG_Q35MF_MKD_HOP_EARLY_HEAD.get()
        self._delay_ns = envs.SGLANG_Q35MF_MKD_HOP_DELAY_NS.get()
        self._logged = False

    def attn_input_fusions(self, plan) -> tuple:
        return ()

    def ffn_input_fusions(self, plan) -> tuple:
        if not self._stock_matches or not isinstance(plan.norm, GemmaRMSNorm):
            return ()
        return (
            FfnInputFusion(
                completes=SumGroup.ATTN_TP, run=partial(self._ffn_input, plan)
            ),
        )

    def attn_push_target(self, m: int):
        """The hop the attention output's producer may push into: the one this layer's FFN input
        consumes for m rows, or None."""
        return self._hop(m)

    def can_defer_all_reduce(self, plan, forward_batch: ForwardBatch) -> bool:
        return False

    def can_defer_finalize(self, plan, forward_batch: ForwardBatch) -> bool:
        m = int(forward_batch.input_ids.shape[0])
        return (
            plan.terminal
            and self.final_norm is not None
            and self.moe_ready is not None
            and self.moe_ready(m)
            and self._hop(m) is not None
        )

    def finalize(self, handoff, residual: torch.Tensor, gamma: torch.Tensor):
        """The final norm of the terminal FFN's handoff; ``gamma`` (the folded Gemma multiplier) is
        unused, since FusedAddRMSNorm adds 1 to the raw weight in fp32."""
        norm = self.final_norm
        hop = (
            self._hop(int(handoff.partial.shape[0]))
            if isinstance(handoff, DraftMoePartial)
            else None
        )
        if hop is None:
            return norm(handoff.complete(), residual)
        return hop.draft_all_reduce_rms_norm(
            local_contribution=handoff.partial,
            residual=residual,
            gamma=norm.weight,
            eps=norm.variance_epsilon,
            weight_bias=1.0,
            add_norm=True,
            early_release=self._early_head
            and self.head_follows is not None
            and self.head_follows(),
            delay_ns=self._delay_ns,
        )

    def _ffn_input(self, plan, hidden_states, residual, forward_batch):
        hop = self._hop(int(hidden_states.shape[0]))
        if hop is None or residual is None:
            return None
        m = int(hidden_states.shape[0])
        return hop.draft_all_reduce_rms_norm(
            local_contribution=hidden_states,
            residual=residual,
            gamma=plan.norm.gemma_weight,
            eps=plan.norm.variance_epsilon,
            weight_bias=0.0,
            add_norm=False,
            early_release=self._early
            and self.moe_single_launch is not None
            and self.moe_single_launch(m),
            delay_ns=self._delay_ns,
        )

    def _hop(self, m: int):
        if not self._stock_matches or not 1 <= m <= _MAX_M:
            return None
        from sglang.srt.layers.q35_hop import active

        hop = active()
        if hop is None or hop.world != get_parallel().tp_size or not hop.supports(m):
            return None
        if not self._logged:
            self._logged = True
            logger.info(
                "MKD draft hop active (tp=%d): attention output with MNNVL rounding, "
                "MoE output with one-shot + CuTe add-norm rounding; entry release MoE %s, head %s.",
                hop.world,
                self._early,
                self._early_head,
            )
        return hop
