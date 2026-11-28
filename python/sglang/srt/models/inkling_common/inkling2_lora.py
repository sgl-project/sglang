"""Shared-outer LoRA on Inkling2's routed experts.

The adapter layout is Tinker's: one gate/up lora_A and one down lora_B shared by all
experts, a gate/up lora_B and a down lora_A per expert. The memory pool stores both B
factors pre-multiplied by the adapter scaling (alpha / r), so no delta here scales again.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.kernels.ops.lora.moe import inkling2_routed
from sglang.kernels.ops.moe.inkling2_moe import unweighted_post_reorder
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.layers import (
    BaseLayerWithLoRA,
    register_shared_outer_expert_lora,
)
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.models.inkling_common.inkling2_moe import Inkling2Experts, RoutedOutput
from sglang.srt.runtime_context import LoRABatchLayout, get_parallel


def add_down_delta(out, delta: torch.Tensor):
    """The experts' output plus the [T, hidden] down delta, in the output's form.

    Routed rows the consumer would combine are combined here first, since the delta
    is per token; the result is a RoutedOutput of already-combined rows (top_k 0).
    """
    if not isinstance(out, RoutedOutput):
        return out + delta.to(out.dtype)
    rows = out.rows
    if out.top_k:
        rows = unweighted_post_reorder(rows, out.src2dst, out.top_k)
    return RoutedOutput(rows + delta.to(rows.dtype), None, 0)


class Inkling2ExpertsWithLoRA(BaseLayerWithLoRA):
    """LoRA steps around Inkling2Experts' own kernels: the gate/up delta enters the
    W13 output before the capped SwiGLU, and the down delta is added per token after
    the rows are combined (summed across the EP group in rank space first)."""

    def __init__(self, base_layer: Inkling2Experts, lora_backend: BaseLoRABackend):
        super().__init__(base_layer, lora_backend)
        lora_backend.is_moe_lora = True
        self.experts_shared_outer_loras = True
        self._token_mask: torch.Tensor | None = None
        self._down_rows = None
        base_layer.prepare_lora_prefill_exchange()

    def set_lora_info(
        self,
        gate_up_lora_a_weights: torch.Tensor,
        gate_up_lora_b_weights: torch.Tensor,
        down_lora_a_weights: torch.Tensor,
        down_lora_b_weights: torch.Tensor,
    ):
        # [loras, 1, 2r, hidden], [loras, experts, 2I, r], [loras, experts, r, I], [loras, 1, hidden, r]
        if gate_up_lora_a_weights.shape[0] != 1:
            raise ValueError(
                "Inkling2 expert LoRA serves one adapter slot per batch; "
                "set --max-loras-per-batch 1"
            )
        rank = gate_up_lora_b_weights.shape[-1]
        # tl.dot in the routed LoRA kernels needs every dimension >= 16.
        if rank < 16 or rank & (rank - 1):
            raise ValueError(
                f"Inkling2 expert LoRA needs --max-lora-rank a power of two >= 16, got {rank}"
            )
        self.set_lora = True
        self.gate_up_lora_a_weights = gate_up_lora_a_weights
        self.gate_up_lora_b_weights = gate_up_lora_b_weights
        self.down_lora_a_weights = down_lora_a_weights
        self.down_lora_b_weights = down_lora_b_weights

    def slice_lora_a_weights(self, A: torch.Tensor):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor):
        return B

    def slice_moe_lora_a_weights(self, A, tp_rank: int, target_module: str):
        self._require_expert_parallel()
        return A

    def slice_moe_lora_b_weights(self, B, tp_rank: int, target_module: str):
        self._require_expert_parallel()
        return B

    def _require_expert_parallel(self):
        if self.base_layer.tp_size != 1:
            raise ValueError(
                "Inkling2 expert LoRA needs expert, not tensor, parallelism"
            )

    def forward(self, x, weights, ids, combine=True):
        batch_info = self.lora_backend.get_batch_info(LoRABatchLayout.TP_GLOBAL)
        # Capture records the LoRA path: a graph replays it whatever the batch holds.
        if not self.set_lora or (
            not get_is_capture_mode() and not batch_info.has_active_lora
        ):
            return self.base_layer(x, weights, ids, combine)
        if x.shape[0] == 0:
            return x
        mapping = batch_info.moe_lora_info.token_lora_mapping[: x.shape[0]]
        self._token_mask = (mapping >= 0).to(x.dtype)[:, None]
        out = self.base_layer(x, weights, ids, combine, lora=self)
        return add_down_delta(out, self._down_delta())

    def gate_up_delta(
        self, *, x, acc, dst2src, src2dst, top_k, gammas, offsets, nvfp4_scales=None
    ):
        """Add the gate/up delta into this rank's W13 rows in place (before the
        SwiGLU reads them), and keep the [rows, r] A2 projection of their activations."""
        experts = self.base_layer
        h = F.linear(x, self.gate_up_lora_a_weights[0, 0]) * self._token_mask
        inkling2_routed.routed_gate_up_delta(
            h=h.contiguous(),
            b=self._local_experts(self.gate_up_lora_b_weights[0]),
            acc=acc,
            dst2src=dst2src,
            offsets=offsets,
            top_k=top_k,
            global_experts=experts.num_global_experts,
            nvfp4_scales=nvfp4_scales,
        )
        inner = inkling2_routed.routed_down_inner(
            acc=acc,
            gammas=gammas,
            a=self._local_experts(self.down_lora_a_weights[0]),
            offsets=offsets,
            gate_cap=experts.gate_cap,
            up_cap=experts.up_cap,
            global_experts=experts.num_global_experts,
            nvfp4_scales=nvfp4_scales,
        )
        self._down_rows = (inner, src2dst, offsets, top_k)

    def _down_delta(self) -> torch.Tensor:
        """[T, hidden] down delta: each token's rows summed to rank space on this
        rank, summed across the EP group, then projected once by the shared B."""
        inner, src2dst, offsets, top_k = self._down_rows
        self._down_rows = None
        ranks = inkling2_routed.routed_token_rank(
            inner=inner,
            src2dst=src2dst,
            offsets=offsets,
            mask=self._token_mask.flatten(),
            top_k=top_k,
        )
        if self.base_layer.ep_size > 1:
            ranks = get_parallel().moe_ep_group.all_reduce(ranks)
        return F.linear(ranks.to(torch.bfloat16), self.down_lora_b_weights[0, 0])

    def _local_experts(self, weight: torch.Tensor) -> torch.Tensor:
        """The pool keys per-expert factors globally or by local id; the kernels index local experts."""
        experts = self.base_layer
        num_local = experts.w13_weight.shape[0]
        if weight.shape[0] == num_local:
            return weight
        assert weight.shape[0] == experts.num_global_experts, weight.shape
        start = experts.ep_rank * num_local
        return weight[start : start + num_local]


register_shared_outer_expert_lora(Inkling2Experts, Inkling2ExpertsWithLoRA)
