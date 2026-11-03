# SPDX-License-Identifier: Apache-2.0
from sglang.srt.layers.utils import PPMissingLayer

from .base import WeightLayoutAdapter
from .layers import LayerLayoutBuilder


class Qwen3WeightAdapter(WeightLayoutAdapter):
    def build_layout(self, model, context):
        b = LayerLayoutBuilder(model)
        if not isinstance(model.model.embed_tokens, PPMissingLayer):
            b.vocab_parallel(model.model.embed_tokens, "model.embed_tokens.weight")
        for index in range(model.start_layer, model.end_layer):
            layer = model.model.layers[index]
            prefix = f"model.layers.{index}"
            attn = layer.self_attn
            b.qkv_parallel(
                attn.qkv_proj,
                tuple(f"{prefix}.self_attn.{p}_proj" for p in ("q", "k", "v")),
            )
            b.row_parallel(attn.o_proj, prefix + ".self_attn.o_proj")
            for name in ("q_norm", "k_norm"):
                b.replicated(
                    getattr(attn, name).weight, f"{prefix}.self_attn.{name}.weight"
                )
            b.rotary(attn.rotary_emb, prefix + ".self_attn.rotary_emb")
            b.mlp(layer.mlp, prefix + ".mlp")
            for name in ("input_layernorm", "post_attention_layernorm"):
                b.replicated(getattr(layer, name).weight, f"{prefix}.{name}.weight")
        if not isinstance(model.model.norm, PPMissingLayer):
            b.replicated(model.model.norm.weight, "model.norm.weight")
        if not isinstance(model.lm_head, PPMissingLayer):
            name = (
                "model.embed_tokens.weight"
                if context.config.tie_word_embeddings
                else "lm_head.weight"
            )
            b.vocab_parallel(model.lm_head, name)
        return b.build()
