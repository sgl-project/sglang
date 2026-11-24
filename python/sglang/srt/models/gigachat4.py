"""GigaChat 4 model.

Built on GigaChat 3.5 layers, norms and attention gates. DSA is enabled by the
model config and implemented by the inherited DeepSeek attention;
LogitsProcessor applies the configured final-logit soft-capping.

The decoder adapts DSA attention outputs and NextN flags for MTP. The model
also supports auxiliary hidden-state capture for DFlash.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from sglang.srt.configs.gigachat4 import GigaChat4Config
from sglang.srt.configs.model_config import dsa_draft_shards_under_dcp
from sglang.srt.layers.aux_hidden_states import (
    AuxHiddenStateAccumulator,
    AuxHiddenStatePacker,
)
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.gigachat35 import (
    GigaChat35DecoderLayer,
    GigaChat35ForCausalLM,
    GigaChat35Model,
)
from sglang.srt.utils import BumpAllocator


class GigaChat4DecoderLayer(GigaChat35DecoderLayer):
    def __init__(
        self,
        config: GigaChat4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
            alt_stream=alt_stream,
            is_nextn=is_nextn,
        )
        if is_nextn and not self.use_linear_attn and self.self_attn.use_dsa:
            attn = self.self_attn
            attn.is_nextn = True
            attn.skip_topk = True
            attn.next_skip_topk = True
            if not dsa_draft_shards_under_dcp(config, attn.dcp_size):
                attn.dcp_size = 1

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        captured_last_layer_outputs: Optional[AuxHiddenStateAccumulator] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = self.attn_boundary.prepare(
            hidden_states,
            forward_batch,
            capture_gathered=captured_last_layer_outputs,
        )

        if self.use_linear_attn:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                input_on_attn_tp_slices=(self.attn_boundary.input_on_attn_tp_slices),
            )
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        get_attn_tp_context().clear_attn_inputs()

        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)

        # Unlike deepseek_v2, no moe_output_buffer_ctx here: non-inplace MoE
        # runners then allocate their output per forward instead of recycling
        # the layer-input buffer. The default (inplace) runners are unaffected.
        with self.ffn_boundary.exit(forward_batch) as ffn_exit:
            hidden_states = self.mlp(hidden_states, forward_batch)

        hidden_states = ffn_exit.finish(hidden_states)
        return hidden_states


class GigaChat4Model(GigaChat35Model):
    decoder_layer_cls = GigaChat4DecoderLayer

    def __init__(
        self,
        config: GigaChat4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.layers_to_capture: list[int] = []

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        total_num_layers = self.end_layer - self.start_layer
        zero_allocator = BumpAllocator(
            buffer_size=total_num_layers * 2,
            dtype=torch.float32,
            device=hidden_states.device,
        )

        aux_hidden_states = AuxHiddenStatePacker.for_batch(
            forward_batch, len(self.layers_to_capture)
        )
        for i in range(self.start_layer, self.end_layer):
            hidden_states = self.layers[i](
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                captured_last_layer_outputs=(
                    aux_hidden_states if i in self.layers_to_capture else None
                ),
            )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        hidden_states = residual_batch.final_norm(
            hidden_states,
            forward_batch,
            self.norm,
            capture=aux_hidden_states.capture
            if self.end_layer in self.layers_to_capture
            else None,
            skip_empty=True,
        )
        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states.finalize()


class GigaChat4ForCausalLM(GigaChat35ForCausalLM):
    model_cls = GigaChat4Model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        inputs_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids, positions, forward_batch, inputs_embeds, pp_proxy_tensors
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if not self.pp_group.is_last_rank:
            return hidden_states
        return self.logits_processor(
            input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    @property
    def capture_aux_hidden_states(self) -> bool:
        return bool(self.model.layers_to_capture)

    def get_aux_hidden_states_width(self) -> int:
        return len(self.model.layers_to_capture) * self.config.hidden_size

    def set_dflash_layers_to_capture(self, layer_ids: list[int]) -> None:
        if self.pp_group.is_last_rank:
            # Layer k's output is captured at the input of layer k + 1.
            self.model.layers_to_capture = [i + 1 for i in layer_ids]


EntryClass = [GigaChat4ForCausalLM]
