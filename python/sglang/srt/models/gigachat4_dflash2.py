"""GigaChat 4 DFlash2 draft: DFlash2 backbone with GigaChat norms and attention gate."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch

from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.models.dflash import (
    DFlash2DraftModel,
    DFlashDecoderLayer,
    DFlashMLAAttention,
)
from sglang.srt.models.gigachat35 import (
    GigaChat35AttentionMLA,
    _remap_gigachat_weight_names,
    build_norm,
)
from sglang.srt.utils import add_prefix


class GigaChat4DFlash2Attention(GigaChat35AttentionMLA, DFlashMLAAttention):
    def __init__(
        self, config, layer_id: int, quant_config=None, prefix: str = ""
    ) -> None:
        DFlashMLAAttention.__init__(
            self,
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=prefix,
        )

        hidden_size = config.hidden_size
        use_scaling = config.use_mla_scaling_factor
        if self.q_lora_rank is not None:
            alpha_q = (hidden_size / self.q_lora_rank) ** 0.5 if use_scaling else 1.0
            self.q_a_layernorm = build_norm(config, self.q_lora_rank, scale=alpha_q)
        alpha_kv = (hidden_size / self.kv_lora_rank) ** 0.5 if use_scaling else 1.0
        self.kv_a_layernorm = build_norm(config, self.kv_lora_rank, scale=alpha_kv)

        self.gated_attention = config.gated_attention
        self._gate_input = None
        if self.gated_attention:
            self.attn_gate = ColumnParallelLinear(
                hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("attn_gate", prefix),
                parallel_group="attn_tp",
            )
            self.o_proj.register_forward_pre_hook(self._o_proj_gate_hook)


class GigaChat4DFlash2DecoderLayer(DFlashDecoderLayer):
    attention_cls = GigaChat4DFlash2Attention

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self.input_layernorm = build_norm(config, config.hidden_size)
        self.post_attention_layernorm = build_norm(config, config.hidden_size)
        if config.layernorm_type == "pre_post":
            self.post_self_attn_layernorm = build_norm(config, config.hidden_size)
            self.post_feedforward_layernorm = build_norm(config, config.hidden_size)


class GigaChat4DFlash2DraftModel(DFlash2DraftModel):
    decoder_layer_cls = GigaChat4DFlash2DecoderLayer

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.norm = build_norm(config, config.hidden_size)
        self.hidden_norm = build_norm(config, config.hidden_size)
        # GigaChat trains every DFlash2 draft so row i drafts token i + 1.
        self.shift_label = True

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        super().load_weights(_remap_gigachat_weight_names(weights))


EntryClass = [GigaChat4DFlash2DraftModel]
