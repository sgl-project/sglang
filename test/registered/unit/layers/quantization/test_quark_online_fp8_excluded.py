"""The online-FP8 skip list must match whole module names."""

import unittest

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _bare_config() -> QuarkConfig:
    # no quantized groups, nothing excluded by name
    return QuarkConfig(quant_config={"packed_modules_mapping": {}, "exclude": []})


class TestQuarkOnlineFp8Excluded(CustomTestCase):
    def test_skip_list_matches_whole_module_names(self):
        """A substring match would keep `gate_proj` in bf16 along with the router `gate`."""
        cfg = _bare_config()
        with envs.SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED.override(True):
            self.assertTrue(
                cfg._serves_excluded_as_online_fp8("model.layers.0.mlp.gate_proj")
            )
            self.assertFalse(
                cfg._serves_excluded_as_online_fp8(
                    "model.layers.3.block_sparse_moe.gate"
                )
            )
            self.assertFalse(cfg._serves_excluded_as_online_fp8("lm_head"))


if __name__ == "__main__":
    unittest.main()
