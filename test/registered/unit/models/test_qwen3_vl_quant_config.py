"""Regression tests for Qwen3-VL vision quantization configuration."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestQwen3VLVisionQuantConfig(CustomTestCase):
    def test_passes_quant_config_to_vision_model(self):
        quant_config = MagicMock()
        config = SimpleNamespace(
            encoder_only=True,
            rope_scaling={},
            vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
        )
        server_args = SimpleNamespace(mm_enable_dp_encoder=False)

        with (
            patch(
                "sglang.srt.models.qwen3_vl.get_pp_group",
                return_value=MagicMock(),
            ),
            patch(
                "sglang.srt.models.qwen3_vl.get_server_args",
                return_value=server_args,
            ),
            patch(
                "sglang.srt.models.qwen3_vl.Qwen3VLMoeVisionModel"
            ) as vision_model_cls,
            patch("sglang.srt.models.qwen3_vl.LogitsProcessor"),
            patch("sglang.srt.models.qwen3_vl.Pooler"),
        ):
            Qwen3VLForConditionalGeneration(config, quant_config=quant_config)

        self.assertIs(vision_model_cls.call_args.kwargs["quant_config"], quant_config)


if __name__ == "__main__":
    import unittest

    unittest.main()
