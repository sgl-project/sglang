# SPDX-License-Identifier: Apache-2.0
from sglang.srt.layers.utils import PPMissingLayer

from ..weight_runtime_manifest import WeightManifestError
from .base import WeightLayoutAdapter
from .layers import LayerLayoutBuilder


class DeepseekV2WeightAdapter(WeightLayoutAdapter):
    def validate(self, model, context):
        super().validate(model, context)
        from sglang.srt.runtime_context import get_exec

        if (
            get_exec().moe.enable_eplb
            or get_exec().moe.ep_num_redundant_experts
            or model.num_fused_shared_experts
        ):
            raise WeightManifestError(
                "DeepSeek reshard does not support EPLB or fused shared experts"
            )

    def build_layout(self, model, context):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE

        b = LayerLayoutBuilder(model)
        if not isinstance(model.model.embed_tokens, PPMissingLayer):
            b.vocab_parallel(model.model.embed_tokens, "model.embed_tokens.weight")
        for index in range(model.start_layer, model.end_layer):
            layer = model.model.layers[index]
            prefix = f"model.layers.{index}"
            attn = layer.self_attn
            if attn.q_lora_rank is not None or attn.use_dsa:
                raise WeightManifestError("unsupported DeepSeek MLA variant")
            b.column_parallel(attn.q_proj, prefix + ".self_attn.q_proj")
            b.replicated_linear(
                attn.kv_a_proj_with_mqa, prefix + ".self_attn.kv_a_proj_with_mqa"
            )
            b.replicated(
                attn.kv_a_layernorm.weight, prefix + ".self_attn.kv_a_layernorm.weight"
            )
            b.column_parallel(attn.kv_b_proj, prefix + ".self_attn.kv_b_proj")
            b.row_parallel(attn.o_proj, prefix + ".self_attn.o_proj")
            if attn.rotary_emb is not None:
                b.rotary(attn.rotary_emb, prefix + ".self_attn.rotary_emb")
            b.derived(
                *(
                    f"{prefix}.self_attn.{name}"
                    for name in ("w_kc", "w_vc", "w_scale", "use_deep_gemm_bmm")
                )
            )
            if type(layer.mlp) is DeepseekV2MLP:
                b.mlp(layer.mlp, prefix + ".mlp")
            elif type(layer.mlp) is DeepseekV2MoE:
                b.replicated(layer.mlp.gate.weight, prefix + ".mlp.gate.weight")
                if layer.mlp.gate.e_score_correction_bias is not None:
                    b.replicated(
                        layer.mlp.gate.e_score_correction_bias,
                        prefix + ".mlp.gate.e_score_correction_bias",
                    )
                b.fused_moe(layer.mlp.experts, prefix + ".mlp.experts")
                if layer.mlp.shared_experts is not None:
                    b.mlp(layer.mlp.shared_experts, prefix + ".mlp.shared_experts")
            else:
                raise WeightManifestError(f"unsupported DeepSeek MLP: {prefix}")
            for name in ("input_layernorm", "post_attention_layernorm"):
                b.replicated(getattr(layer, name).weight, f"{prefix}.{name}.weight")
        if not isinstance(model.model.norm, PPMissingLayer):
            b.replicated(model.model.norm.weight, "model.norm.weight")
        if not isinstance(model.lm_head, PPMissingLayer):
            b.vocab_parallel(model.lm_head, "lm_head.weight")
        return b.build()

    def finalize_target(self, model, context):
        from ..weight_heterogeneous_transfer import rebuild_transferred_weight_state

        rebuild_transferred_weight_state(model)
