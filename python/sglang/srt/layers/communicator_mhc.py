# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.kernels.ops.layernorm.mhc import hc_contract, hc_expand
from sglang.srt.layers.boundary_layout import DecoderLayerSides, TokenAxis
from sglang.srt.layers.communicator import (
    BoundarySteps,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)


@dataclass
class MHCState:
    """A layer's residual as hyper-connection streams, with the residual
    operations the boundary steps run (communicator.ResidualOps): the residual
    is hc_mult streams, a stage's output is written in with hc_post and the
    next stage's input read with hc_pre and the norm. A read produces the
    h_res / h_post the next write-back consumes; they move with the tokens.
    Parameters belong to the owning layer; this state only holds scratch
    shared across communication stages."""

    hc_mult: int
    hc_attn_pre: Callable
    hc_ffn_pre: Callable
    hc_post: Callable
    hc_ffn_post_pre: Optional[Callable] = None
    # The last layer's write-back also contracts the streams into the hidden
    # states the layer stack hands on.
    is_last_layer: bool = False
    h_res: Optional[torch.Tensor] = None
    h_post: Optional[torch.Tensor] = None

    # hc_post is not a plain add, so it runs only once the sum it writes in
    # is complete.
    adds_plainly = False
    # The FFN output's write-back uses the coefficients this layer's FFN input
    # read produced, so the layer runs it itself.
    updates_residual_after_ffn = True

    @staticmethod
    def _resolve_out_norm(out_norm):
        if out_norm is None:
            return None, None
        return out_norm.weight.data, out_norm.variance_epsilon

    def attn_split(self, hidden_states, out_norm: Optional[torch.nn.Module] = None):
        residual = hidden_states
        out_norm_weight, out_norm_eps = self._resolve_out_norm(out_norm)
        hidden_states, self.h_res, self.h_post, norm_fused = self.hc_attn_pre(
            hidden_states, out_norm_weight, out_norm_eps
        )
        if out_norm is not None and not norm_fused and hidden_states.shape[0] != 0:
            hidden_states = out_norm(hidden_states)
        return hidden_states, residual

    def attn_to_mlp(
        self, hidden_states, residual, out_norm: Optional[torch.nn.Module] = None
    ):
        out_norm_weight, out_norm_eps = self._resolve_out_norm(out_norm)
        if self.hc_ffn_post_pre is not None and hidden_states.shape[0] != 0:
            # Returns None when it declines -- no fused kernel for this platform
            # or shape, or a shape the fusion is slower at -- and the chain runs.
            fused = self.hc_ffn_post_pre(
                hidden_states=hidden_states,
                residual=residual,
                h_res=self.h_res,
                h_post=self.h_post,
                out_norm_weight=out_norm_weight,
                out_norm_eps=out_norm_eps,
            )
            if fused is not None:
                hidden_states, residual, self.h_res, self.h_post, norm_fused = fused
                if out_norm is not None and not norm_fused:
                    hidden_states = out_norm(hidden_states)
                return hidden_states, residual

        hidden_states = self.hc_post(hidden_states, residual, self.h_res, self.h_post)
        residual = hidden_states
        hidden_states, self.h_res, self.h_post, norm_fused = self.hc_ffn_pre(
            hidden_states, out_norm_weight, out_norm_eps
        )
        if out_norm is not None and not norm_fused and hidden_states.shape[0] != 0:
            hidden_states = out_norm(hidden_states)
        return hidden_states, residual

    def mlp_combine(self, hidden_states, residual):
        return self.hc_post(hidden_states, residual, self.h_res, self.h_post)

    def reset_aux(self):
        self.h_res = None
        self.h_post = None

    def enter(self, hidden_states):
        return hc_expand(hidden_states, self.hc_mult)

    def read_attention_input(self, residual, norm, quant_format):
        if quant_format:
            raise NotImplementedError(f"an MHC attention input in {quant_format=}")
        return self.attn_split(residual, out_norm=norm)

    def update_and_read_attention_input(
        self, hidden_states, residual, norm, quant_format, post_residual_addition
    ):
        # An MHC layer's input arrives with the previous FFN output written in.
        raise NotImplementedError("an MHC layer takes its input written back")

    def update_and_read_ffn_input(self, hidden_states, residual, norm):
        return self.attn_to_mlp(hidden_states, residual, out_norm=norm)

    def update_residual(self, hidden_states, residual):
        hidden_states = self.mlp_combine(hidden_states, residual)
        self.reset_aux()
        if self.is_last_layer:
            hidden_states = hc_contract(hidden_states, self.hc_mult)
        return hidden_states

    def residual_to_attn_tp_shard(self, residual, context):
        rank, size = context.attn_tp_rank, context.attn_tp_size
        self.h_res = self.h_res.tensor_split(size)[rank]
        self.h_post = self.h_post.tensor_split(size)[rank]
        return residual.tensor_split(size)[rank]

    def residual_from_attn_tp_shards(self, residual):
        raise NotImplementedError(
            "Unsupported: h_res/h_post allgather not implemented."
        )


class MHCLayerCommunicator(LayerCommunicator):
    """A layer whose residual is hyper-connection streams: the shared boundary
    steps, run with MHCState's residual operations."""

    def __init__(
        self,
        layer_scatter_modes: LayerScatterModes,
        input_layernorm: torch.nn.Module,
        post_attention_layernorm: torch.nn.Module,
        allow_reduce_scatter: bool = False,
        qkv_latent_func: Optional[Callable] = None,
        *,
        hc_mult: int,
        hc_attn_pre: Callable,
        hc_ffn_pre: Callable,
        hc_post: Callable,
        hc_ffn_post_pre: Optional[Callable] = None,
    ):
        self.mhc = MHCState(
            hc_mult=hc_mult,
            hc_attn_pre=hc_attn_pre,
            hc_ffn_pre=hc_ffn_pre,
            hc_post=hc_post,
            hc_ffn_post_pre=hc_ffn_post_pre,
            is_last_layer=layer_scatter_modes.is_last_layer,
        )
        # The postprocess writes the FFN output into the streams, so the FFN's
        # sum never waits for the next layer.
        super().__init__(
            layer_scatter_modes,
            input_layernorm,
            post_attention_layernorm,
            allow_reduce_scatter,
            qkv_latent_func,
            allow_deferred_ffn_reduction=False,
            residual_ops=self.mhc,
        )

    def _steps_from_declarations(
        self, sides: DecoderLayerSides, **kwargs
    ) -> BoundarySteps:
        # MHC has not been run with an FFN input gathered over attention CP.
        if (
            TokenAxis.ATTN_CP
            in sides.attention_output.layout.sharded - sides.ffn.layout.sharded
        ):
            raise NotImplementedError(
                f"MHCLayerCommunicator with a gather over attention CP: {sides=}"
            )
        return super()._steps_from_declarations(sides, **kwargs)

    def _post_init_communicate(self):
        if self.layer_scatter_modes.mlp_mode == ScatterMode.MOE_FULL:
            raise NotImplementedError(
                "MHCLayerCommunicator does not support MOE_FULL "
                "(moe_dp_size < attention_context_parallel_size). Increase "
                "moe_dp_size to match attention_context_parallel_size."
            )
        return super()._post_init_communicate()
