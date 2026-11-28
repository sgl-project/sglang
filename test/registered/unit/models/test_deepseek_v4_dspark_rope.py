"""DeepSeek V4 RoPE covers internal DSpark lookahead at the context limit."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_v4 import MqaAttentionBase
from sglang.srt.runtime_context import get_spec, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepSeekV4DSparkRoPE(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", page_size=256), role="test"
        )

    def test_rope_rows_cover_overlapped_verify_windows(self):
        """An in-flight verify and its successor need RoPE rows past the limit."""
        context_len = 32
        verify_width = 6
        config = SimpleNamespace(
            model_type="deepseek_v41",
            hidden_size=8,
            head_dim=8,
            qk_rope_head_dim=4,
            num_attention_heads=8,
            num_key_value_heads=1,
            o_groups=4,
            q_lora_rank=8,
            o_lora_rank=8,
            rms_norm_eps=1e-6,
            q_head_norm=True,
            rope_theta=10000,
            compress_rope_theta=40000,
            max_position_embeddings=context_len,
            rope_scaling={
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 16,
            },
        )
        for compress_ratio in (0, 2):
            with self.subTest(compress_ratio=compress_ratio):
                attention_kwargs = dict(
                    config=config,
                    layer_id=0,
                    quant_config=None,
                    prefix="",
                    compress_ratio=compress_ratio,
                    fuse_wqa_wkv=False,
                    wo_a_fp8=False,
                )
                base = MqaAttentionBase(**attention_kwargs).freqs_cis
                with get_spec().override(
                    speculative_algorithm="DSPARK",
                    speculative_num_draft_tokens=verify_width,
                ):
                    dspark = MqaAttentionBase(**attention_kwargs).freqs_cis

                self.assertEqual(base.shape[0], context_len)
                self.assertEqual(config.max_position_embeddings, context_len)
                torch.testing.assert_close(dspark[:context_len], base, rtol=0, atol=0)

                # Select the same absolute RoPE rows the indexer consumes.
                positions = torch.arange(
                    context_len - 1, context_len - 1 + 2 * verify_width
                )
                freqs = dspark[positions]
                self.assertTrue(torch.isfinite(freqs).all())


if __name__ == "__main__":
    unittest.main()
