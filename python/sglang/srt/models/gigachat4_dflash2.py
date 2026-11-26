"""GigaChat 4 DFlash2 draft: DFlash2 backbone with GigaChat norms and attention gate."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.ops.activation.silu_and_mul_clamp import silu_and_mul_clamp
from sglang.srt.layers.linear import ColumnParallelLinear
from sglang.srt.layers.utils.common import add_prefix
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
from sglang.srt.utils import is_cuda

_is_cuda = is_cuda()


class GigaChat4DFlash2SiluAndMul(nn.Module):
    def __init__(self, limit: float) -> None:
        super().__init__()
        self.limit = limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            _is_cuda
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and x.numel() > 0
        ):
            out = x.new_empty((*x.shape[:-1], x.shape[-1] // 2))
            silu_and_mul_clamp(x, out, self.limit)
            return out
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate.clamp(max=self.limit)) * up.clamp(-self.limit, self.limit)


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
        swiglu_limit = float(getattr(config, "swiglu_limit", None) or 0.0)
        if config.hidden_act == "silu" and swiglu_limit > 0:
            self.mlp.act_fn = GigaChat4DFlash2SiluAndMul(swiglu_limit)
        self.input_layernorm = build_norm(config, config.hidden_size)
        self.post_attention_layernorm = build_norm(config, config.hidden_size)
        if config.layernorm_type == "pre_post":
            self.post_self_attn_layernorm = build_norm(config, config.hidden_size)
            self.post_feedforward_layernorm = build_norm(config, config.hidden_size)


class GigaChat4DFlash2DraftModel(DFlash2DraftModel):
    decoder_layer_cls = GigaChat4DFlash2DecoderLayer

    def __init__(self, config, quant_config=None, prefix: str = "") -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        self.shift_label = bool(config.dflash_config.get("shift_label", True))
        self.num_stages = int(config.num_hidden_layers)
        self.norm = build_norm(config, config.hidden_size)
        self.hidden_norm = build_norm(config, config.hidden_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        super().load_weights(_remap_gigachat_weight_names(weights))


EntryClass = [GigaChat4DFlash2DraftModel]
