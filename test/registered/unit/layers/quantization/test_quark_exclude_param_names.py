import unittest

from sglang.srt.layers.quantization.quark.quark import _exclude_module_names
from sglang.srt.layers.quantization.quark.utils import should_ignore_layer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQuarkExcludeParamNames(CustomTestCase):
    def test_weight_suffixed_exclude_matches_its_module(self):
        # amd/Qwen3.8-27B-Quark-AWQ-MXFP4 lists its bf16 MTP layer by parameter
        # name ("mtp.fc.weight"), which never equals a module name.
        excludes = _exclude_module_names(
            [
                "mtp.fc.weight",
                "mtp.layers.0.self_attn.q_proj.weight",
                "mtp.layers.0.self_attn.k_proj.weight",
                "mtp.layers.0.self_attn.v_proj.weight",
                "re:.*lm_head.weight",
            ]
        )
        fused = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
        self.assertTrue(should_ignore_layer("mtp.fc", ignore=excludes))
        self.assertTrue(
            should_ignore_layer(
                "mtp.layers.0.self_attn.qkv_proj", ignore=excludes, fused_mapping=fused
            )
        )
        self.assertFalse(
            should_ignore_layer(
                "model.layers.0.self_attn.qkv_proj",
                ignore=excludes,
                fused_mapping=fused,
            )
        )
        self.assertIn("re:.*lm_head.weight", excludes)


if __name__ == "__main__":
    unittest.main()
