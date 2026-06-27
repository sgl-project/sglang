import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.low_precision_linear import (
    TE_NVFP4_LINEAR_TARGETS_ENV,
    TeNvfp4LinearRunner,
    maybe_get_te_nvfp4_linear_runner,
    te_nvfp4_linear_target_enabled,
)


class TestTeNvfp4LinearTargetPolicy(unittest.TestCase):
    def test_targets_default_disabled(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(te_nvfp4_linear_target_enabled("ltx2.video_ffn"))
            self.assertIsNone(maybe_get_te_nvfp4_linear_runner("ltx2.video_ffn"))

    def test_specific_target_enabled(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_TARGETS_ENV: "qwen_image.ffn, ltx2.video_ffn"},
            clear=True,
        ):
            self.assertTrue(te_nvfp4_linear_target_enabled("ltx2.video_ffn"))
            self.assertFalse(te_nvfp4_linear_target_enabled("wan.video_ffn"))

            runner = maybe_get_te_nvfp4_linear_runner("ltx2.video_ffn")
            self.assertIsInstance(runner, TeNvfp4LinearRunner)
            self.assertEqual(runner.target, "ltx2.video_ffn")

    def test_all_target_enabled(self):
        with patch.dict(
            "os.environ",
            {TE_NVFP4_LINEAR_TARGETS_ENV: "all"},
            clear=True,
        ):
            self.assertTrue(te_nvfp4_linear_target_enabled("ltx2.video_ffn"))
            self.assertTrue(te_nvfp4_linear_target_enabled("wan.video_ffn"))

    def test_cpu_input_short_circuits_before_te_or_distributed_setup(self):
        runner = TeNvfp4LinearRunner(target="unit.test")
        layer = nn.Linear(4, 4)
        x = torch.ones(2, 4, dtype=torch.float32)

        self.assertIsNone(runner.try_apply("linear", layer, x, training=False))


if __name__ == "__main__":
    unittest.main()
