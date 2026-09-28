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
"""Hyper-connection residual streams."""

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from sglang.kernels.ops.layernorm.mhc import hc_contract, hc_expand
from sglang.srt.layers.communicator.residual import LayerResidual


@dataclass
class MHCState:
    """A layer's residual as hyper-connection streams: the residual is hc_mult
    streams, a stage's output is written in with hc_post and the next stage's
    input read with hc_pre and the norm. A read produces the h_res / h_post the
    next write-back consumes; they move with the tokens. ``layer_residual()``
    gives the reads and updates the layer's two stages declare. Parameters
    belong to the owning layer; this state only holds scratch shared across
    communication stages."""

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

    def residual_to_attn_tp_shard(self, residual, context):
        rank, size = context.attn_tp_rank, context.attn_tp_size
        self.h_res = self.h_res.tensor_split(size)[rank]
        self.h_post = self.h_post.tensor_split(size)[rank]
        return residual.tensor_split(size)[rank]

    def residual_from_attn_tp_shards(self, residual):
        raise NotImplementedError(
            "Unsupported: h_res/h_post allgather not implemented."
        )

    def layer_residual(self) -> LayerResidual:
        return LayerResidual(
            attention_read=_AttentionRead(self),
            attention_update=_AttentionUpdate(self),
            ffn_read=_FfnRead(self),
            ffn_update=_FfnUpdate(self),
        )


class _AttentionRead:
    """hc_pre and the input norm, from streams that already hold the previous
    layer's output: an MHC layer takes its input written back."""

    norms_plainly = False

    def __init__(self, state: MHCState):
        self.state = state

    def enter(self, hidden_states):
        return hc_expand(hidden_states, self.state.hc_mult)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        if quant_format:
            raise NotImplementedError(f"an MHC attention input in {quant_format=}")
        return self.state.attn_split(residual, out_norm=norm)

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        raise NotImplementedError("an MHC layer takes its input written back")


class _AttentionUpdate:
    """hc_post with the coefficients the attention's read produced. It is not a
    plain add, so it runs only once the sum it writes in is complete."""

    adds_plainly = False
    at_producer = False

    def __init__(self, state: MHCState):
        self.state = state

    def update(self, hidden_states, residual):
        return self.state.mlp_combine(hidden_states, residual)

    def residual_to_attn_tp_shard(self, residual, context):
        return self.state.residual_to_attn_tp_shard(residual, context)

    def residual_from_attn_tp_shards(self, residual):
        return self.state.residual_from_attn_tp_shards(residual)


class _FfnRead:
    """The attention output's hc_post and the FFN input's hc_pre and norm, fused
    in hc_ffn_post_pre when it takes the batch."""

    norms_plainly = False

    def __init__(self, state: MHCState):
        self.state = state

    def enter(self, hidden_states):
        return hc_expand(hidden_states, self.state.hc_mult)

    def read(self, residual, norm, quant_format="", post_residual_addition=None):
        raise NotImplementedError("an MHC FFN input read without its attention")

    def update_and_read(self, update, hidden_states, residual, norm, **kwargs):
        if not isinstance(update, _AttentionUpdate) or update.state is not self.state:
            raise NotImplementedError(f"an MHC FFN input after {update=}")
        return self.state.attn_to_mlp(hidden_states, residual, out_norm=norm)


class _FfnUpdate:
    """hc_post with the coefficients the FFN input's read produced, which this
    layer runs itself; the last layer also contracts the streams into the
    hidden states the layer stack hands on."""

    adds_plainly = False
    at_producer = True

    def __init__(self, state: MHCState):
        self.state = state

    def update(self, hidden_states, residual):
        hidden_states = self.state.mlp_combine(hidden_states, residual)
        self.state.reset_aux()
        if self.state.is_last_layer:
            hidden_states = hc_contract(hidden_states, self.state.hc_mult)
        return hidden_states

    def residual_to_attn_tp_shard(self, residual, context):
        return self.state.residual_to_attn_tp_shard(residual, context)

    def residual_from_attn_tp_shards(self, residual):
        return self.state.residual_from_attn_tp_shards(residual)
