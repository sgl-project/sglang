# SPDX-License-Identifier: Apache-2.0
"""The gfx950 MoE layer of minimax_m3: folds the shared expert into the aiter MoE output.
minimax_m3_hip.moe_cls() reaches it lazily, since it subclasses minimax_m3's MiniMaxM3MoE."""

from contextlib import nullcontext
from typing import Union

import torch

from sglang.kernels.ops.moe.aiter_moe_addend import (
    apply_aiter_moe_addend_patch,
    moe_output_addend,
)
from sglang.srt.layers.layer_boundary.fusions.moe_finalize import MoeDeferredFinalize
from sglang.srt.models.minimax_m3 import MiniMaxM3MoE, _input_rows
from sglang.srt.runtime_context import get_forward


class MiniMaxM3MoEGfx950(MiniMaxM3MoE):
    """The MoE layer on gfx950; a gfx950 FFN norm hands it ``(fp8, scale, bf16 norm)``."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # aiter's reducing stage2 must fold the addend that forward_normal offers
        apply_aiter_moe_addend_patch()

    def forward_normal(
        self, hidden_states: torch.Tensor
    ) -> Union[torch.Tensor, MoeDeferredFinalize]:
        shared_input = hidden_states
        if isinstance(hidden_states, tuple):
            # (fp8, scale) for the shared expert, the bf16 norm for router and experts
            shared_input, hidden_states = hidden_states[:2], hidden_states[2]
        defer_finalize = get_forward().defer_moe_finalize
        if hidden_states.shape[0] > 0:
            # the alt-stream overlap of the parent is CUDA-only (alt_stream is None on ROCm)
            shared_output = self._forward_shared_experts(shared_input)
            # dynamo cannot trace the context-variable handoff
            fold_shared = (
                shared_output is not None
                and not defer_finalize
                and not torch.compiler.is_compiling()
            )
            with (
                moe_output_addend(shared_output) if fold_shared else nullcontext()
            ) as addend:
                final_hidden_states = self._forward_router_experts(
                    hidden_states, defer_finalize=defer_finalize
                )
            if addend is not None and addend.consumed:
                shared_output = None
        else:
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(hidden_states, topk_output)

        if defer_finalize:
            from sglang.srt.layers.moe.utils import sum_post_experts_output

            return MoeDeferredFinalize.from_flashinfer(
                final_hidden_states,
                gated_shared_output=shared_output,
                m=hidden_states.shape[0],
                reduce=sum_post_experts_output,
            )
        if shared_output is not None:
            final_hidden_states = final_hidden_states + shared_output
        return final_hidden_states

    def _forward_shared_experts(self, hidden_states):
        if _input_rows(hidden_states) > 0 and self.num_fused_shared_experts == 0:
            return self.shared_experts(hidden_states)
        return None
