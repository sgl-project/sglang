import unittest

from sglang.srt.layers.quantization.quark.quark import _exclude_module_names
from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

# amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (dense, all 15 mtp.* entries): the bf16 MTP
# layer is listed by parameter name, with a `.weight` suffix.
_DENSE_WEIGHT_SUFFIX_EXCLUDES = [
    "mtp.fc.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
]


class TestQuarkExcludeParamNames(CustomTestCase):
    def test_weight_suffixed_excludes_keep_dense_mtp_layers_bf16(self):
        """Excludes that name parameters must still exclude their modules.

        Matched as written, `mtp.layers.0.self_attn.q_proj.weight` never equals a
        module name, so the draft's linears were built MXFP4 and loading the bf16
        checkpoint shards failed a shape assertion.
        """
        excludes = _exclude_module_names(_DENSE_WEIGHT_SUFFIX_EXCLUDES)
        for module in (
            "mtp.fc",
            "mtp.layers.0.self_attn.qkv_proj",
            "mtp.layers.0.self_attn.o_proj",
            "mtp.layers.0.mlp.gate_up_proj",
            "mtp.layers.0.mlp.down_proj",
        ):
            with self.subTest(module=module):
                self.assertTrue(
                    should_ignore_layer(
                        module,
                        ignore=excludes,
                        fused_mapping=Qwen3_5ForCausalLMMTP.packed_modules_mapping,
                    )
                )
        self.assertFalse(
            should_ignore_layer(
                "model.layers.0.mlp.gate_up_proj",
                ignore=excludes,
                fused_mapping=Qwen3_5ForCausalLMMTP.packed_modules_mapping,
            )
        )

    def test_exclude_module_names_keeps_regex_and_module_entries(self):
        self.assertEqual(
            _exclude_module_names(["mtp.fc.weight", "mtp.fc", "re:.*proj.weight"]),
            ["mtp.fc", "mtp.fc", "re:.*proj.weight"],
        )


if __name__ == "__main__":
    unittest.main()
