import unittest
from unittest.mock import patch

from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe_triton_config as cfg
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeterministicMoeConfig(CustomTestCase):
    def test_gfx950_bf16_switches_above_small_m(self):
        with patch.object(cfg, "_is_gfx95", True):
            small = cfg._deterministic_config(
                cfg._DETERMINISTIC_GFX950_SMALL_M, None, False, None
            )
            large = cfg._deterministic_config(
                cfg._DETERMINISTIC_GFX950_SMALL_M + 1, None, False, None
            )
        self.assertEqual(small, cfg._DETERMINISTIC_CONFIG)
        self.assertEqual(large, cfg._DETERMINISTIC_CONFIG_GFX950_LARGE_M)

    def test_quantized_and_other_gpus_keep_one_config(self):
        """Only unquantized MoE on gfx950 was shown to give the same bits with both configs."""
        cases = [
            (True, "fp8_w8a8", False, None),
            (True, "fp8_w8a8", False, [128, 128]),
            (True, "int8_w8a16", False, None),
            (True, None, True, None),
            (True, None, False, [128, 128]),
            (False, None, False, None),
        ]
        for is_gfx95, dtype, is_marlin, block_shape in cases:
            with self.subTest(
                is_gfx95=is_gfx95,
                dtype=dtype,
                is_marlin=is_marlin,
                block_shape=block_shape,
            ):
                with patch.object(cfg, "_is_gfx95", is_gfx95):
                    for m in (1, 64, 65, 16384):
                        self.assertEqual(
                            cfg._deterministic_config(m, dtype, is_marlin, block_shape),
                            cfg._DETERMINISTIC_CONFIG,
                        )

    def test_returned_config_is_a_copy(self):
        """Callers pop keys from the config; the module constants must not change."""
        with patch.object(cfg, "_is_gfx95", True):
            c = cfg._deterministic_config(4096, None, False, None)
        c.pop("BLOCK_SIZE_M")
        self.assertIn("BLOCK_SIZE_M", cfg._DETERMINISTIC_CONFIG_GFX950_LARGE_M)


if __name__ == "__main__":
    unittest.main()
