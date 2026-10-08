"""Heads sharded by TP or Ulysses must reproduce the unsharded torch SDPA.

PyTorch's flash forward switches to split-KV when few heads cover a short
sequence, so a head shard can take a different kernel than the full layer.
``SDPAImpl`` pads query rows to keep the full layer's kernel; concatenating
the per-shard outputs must then equal the unsharded output bit for bit.
The padding serves "exact" quality requests only."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    CudnnSDPAImpl,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.test.test_utils import CustomTestCase

_GLOBAL_HEADS = 24


def _impl(cls=SDPAImpl, heads=_GLOBAL_HEADS, head_dim=128, **kwargs):
    return cls(
        num_heads=heads,
        head_size=head_dim,
        causal=kwargs.pop("causal", False),
        softmax_scale=head_dim**-0.5,
        **kwargs,
    )


def _qkv(seq_len, head_dim, dtype=torch.bfloat16, heads=_GLOBAL_HEADS):
    generator = torch.Generator(device="cuda").manual_seed(seq_len + head_dim)
    return tuple(
        torch.randn(
            1, seq_len, heads, head_dim, device="cuda", dtype=dtype, generator=generator
        )
        for _ in range(3)
    )


def _request(quality="exact"):
    return set_forward_context(
        current_timestep=0,
        attn_metadata=None,
        forward_batch=SimpleNamespace(quality=quality),
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSDPAShardedHeads(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._request = _request()
        self._request.__enter__()

    def tearDown(self):
        self._request.__exit__(None, None, None)
        super().tearDown()

    @torch.no_grad()
    def test_head_shards_match_unsharded_attention(self):
        for head_dim in (64, 128):
            for seq_len in (1025, 1536, 4608):
                query, key, value = _qkv(seq_len, head_dim)
                expected = F.scaled_dot_product_attention(
                    *(x.transpose(1, 2) for x in (query, key, value)),
                    scale=head_dim**-0.5,
                ).transpose(1, 2)
                for local_heads in (24, 12, 6, 3):
                    with self.subTest(
                        head_dim=head_dim, seq_len=seq_len, local_heads=local_heads
                    ):
                        impl = _impl(
                            heads=local_heads,
                            head_dim=head_dim,
                            global_num_heads=_GLOBAL_HEADS,
                        )
                        actual = torch.cat(
                            [
                                impl.forward(q, k, v, None)
                                for q, k, v in zip(
                                    query.split(local_heads, dim=2),
                                    key.split(local_heads, dim=2),
                                    value.split(local_heads, dim=2),
                                )
                            ],
                            dim=2,
                        )
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_padding_only_when_the_shard_alone_would_split(self):
        query = torch.empty(1, 6, 1025, 128, device="cuda", dtype=torch.bfloat16)
        self.assertGreater(
            _impl(heads=6, global_num_heads=24)._unsplit_query_len(query), 1025
        )
        # Unknown, unsharded, FP32, dropout and cuDNN keep the input length.
        cases = {
            "unknown": _impl(heads=6),
            "unsharded": _impl(heads=6, global_num_heads=6),
            "dropout": _impl(heads=6, global_num_heads=24, dropout_p=0.1),
            "cudnn": _impl(CudnnSDPAImpl, heads=6, global_num_heads=24),
            "cudnn_priority": _impl(heads=6, global_num_heads=24, allow_cudnn_sdp=True),
        }
        for name, impl in cases.items():
            with self.subTest(name):
                self.assertEqual(impl._unsplit_query_len(query), 1025)
        with self.subTest("fp32"):
            impl = _impl(heads=6, global_num_heads=24)
            self.assertEqual(impl._unsplit_query_len(query.float()), 1025)
        with self.subTest("lossless request"), _request("lossless"):
            impl = _impl(heads=6, global_num_heads=24)
            self.assertEqual(impl._unsplit_query_len(query), 1025)
        with self.subTest("full layer also splits"):
            short = query[:, :, :64]
            self.assertEqual(
                _impl(heads=6, global_num_heads=24)._unsplit_query_len(short), 64
            )

    @torch.no_grad()
    def test_causal_and_masked_calls_are_not_padded(self):
        impl = _impl(heads=6, global_num_heads=24, causal=True)
        query, key, value = _qkv(1025, 128, heads=6)
        shapes = []
        original = F.scaled_dot_product_attention

        def record(q, *args, **kwargs):
            shapes.append(q.shape[-2])
            return original(q, *args, **kwargs)

        with patch.object(F, "scaled_dot_product_attention", side_effect=record):
            impl.forward(query, key, value, None)
            impl.forward(query[:, :512], key, value, None)
        self.assertEqual(shapes, [1025, 512])


if __name__ == "__main__":
    unittest.main()
