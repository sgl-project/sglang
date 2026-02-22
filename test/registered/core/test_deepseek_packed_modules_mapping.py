from sglang.srt.layers.quantization.compressed_tensors.utils import should_ignore_layer
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM
from sglang.srt.models.glm4_moe import Glm4MoeForCausalLM
from sglang.srt.models.glm4_moe_lite import Glm4MoeLiteForCausalLM


def test_deepseek_v2_gate_up_proj_ignore_via_fused_mapping():
    assert should_ignore_layer(
        "model.layers.0.mlp.gate_up_proj",
        ignore=[
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
        ],
        fused_mapping=DeepseekV2ForCausalLM.packed_modules_mapping,
    )


def test_glm4_moe_lite_fused_qkv_ignore_via_fused_mapping():
    assert should_ignore_layer(
        "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa",
        ignore=[
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
        ],
        fused_mapping=Glm4MoeLiteForCausalLM.packed_modules_mapping,
    )


def test_glm4_moe_gate_up_proj_ignore_via_fused_mapping():
    assert should_ignore_layer(
        "model.layers.0.mlp.gate_up_proj",
        ignore=[
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
        ],
        fused_mapping=Glm4MoeForCausalLM.packed_modules_mapping,
    )
