# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LiteAttention backend registration and skip policy."""

import unittest
from unittest import mock

from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum


class TestLiteAttentionBackendEnum(unittest.TestCase):
    def test_enum_registered(self):
        self.assertEqual(
            str(AttentionBackendEnum.LITE_ATTENTION), "lite_attention"
        )
        self.assertTrue(AttentionBackendEnum.LITE_ATTENTION.is_sparse)

    def test_cross_attention_disables_skipping(self):
        lite_attn = unittest.mock.MagicMock()
        seq_la = unittest.mock.MagicMock()
        with (
            mock.patch.dict(
                "sys.modules",
                {
                    "lite_attention": mock.MagicMock(
                        LiteAttention=lite_attn,
                        SeqParallelLiteAttention=seq_la,
                    )
                },
            ),
            mock.patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.lite_attn.get_ulysses_parallel_world_size",
                return_value=1,
            ),
            mock.patch(
                "sglang.multimodal_gen.runtime.layers.attention.backends.lite_attn.get_global_server_args",
            ) as mock_args,
        ):
            mock_args.return_value.attention_backend_config = {}
            from sglang.multimodal_gen.runtime.layers.attention.backends.lite_attn import (
                LiteAttentionImpl,
            )

            LiteAttentionImpl(
                num_heads=8,
                head_size=64,
                causal=False,
                softmax_scale=0.125,
                is_cross_attention=True,
            )
            lite_attn.assert_called_once()
            self.assertFalse(lite_attn.call_args.kwargs["enable_skipping"])

    def test_causal_raises(self):
        with mock.patch.dict(
            "sys.modules",
            {
                "lite_attention": mock.MagicMock(
                    LiteAttention=mock.MagicMock(),
                    SeqParallelLiteAttention=mock.MagicMock(),
                )
            },
        ):
            from sglang.multimodal_gen.runtime.layers.attention.backends.lite_attn import (
                LiteAttentionImpl,
            )

            with self.assertRaises(NotImplementedError):
                LiteAttentionImpl(
                    num_heads=8,
                    head_size=64,
                    causal=True,
                    softmax_scale=0.125,
                )


if __name__ == "__main__":
    unittest.main()
