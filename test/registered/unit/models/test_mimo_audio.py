"""CPU-only regression tests for MiMo audio attention parallelism wiring."""

import unittest
from unittest.mock import patch

import torch.nn as nn

from sglang.srt.models import mimo_audio
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestAudioEncoderAttention(CustomTestCase):
    def test_output_reduce_follows_dp_attention_mode(self):
        for enabled in (False, True):
            with (
                self.subTest(enabled=enabled),
                patch.object(
                    mimo_audio, "is_dp_attention_enabled", return_value=enabled
                ),
                patch.object(
                    mimo_audio, "VisionAttention", return_value=nn.Identity()
                ) as vision_attention,
            ):
                layer = mimo_audio.AudioEncoderAttention(
                    embed_dim=16,
                    num_heads=2,
                )

            self.assertIsInstance(layer.attn, nn.Identity)
            self.assertEqual(
                vision_attention.call_args.kwargs["use_dp_attention_reduce"],
                enabled,
            )


if __name__ == "__main__":
    unittest.main()
