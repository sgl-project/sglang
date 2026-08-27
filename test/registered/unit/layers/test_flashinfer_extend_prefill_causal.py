"""Unit tests for FlashInfer extend prefill causal flag — #36663."""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.flashinfer_backend import _extend_prefill_causal
from sglang.srt.layers.radix_attention import AttentionType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _layer(attn_type, *, is_cross_attention=False):
    return SimpleNamespace(
        attn_type=attn_type,
        is_cross_attention=is_cross_attention,
    )


class TestFlashInferExtendPrefillCausal(unittest.TestCase):
    def test_decoder_is_causal(self):
        self.assertTrue(_extend_prefill_causal(_layer(AttentionType.DECODER)))

    def test_decoder_bidirectional_is_not_causal(self):
        self.assertFalse(
            _extend_prefill_causal(_layer(AttentionType.DECODER_BIDIRECTIONAL))
        )

    def test_encoder_only_is_not_causal(self):
        self.assertFalse(_extend_prefill_causal(_layer(AttentionType.ENCODER_ONLY)))

    def test_cross_attention_is_not_causal(self):
        self.assertFalse(
            _extend_prefill_causal(
                _layer(AttentionType.DECODER, is_cross_attention=True)
            )
        )


if __name__ == "__main__":
    unittest.main()
