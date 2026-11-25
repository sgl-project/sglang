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

from sglang.multimodal_gen.runtime.layers.attention.backends import sdpa
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    CudnnSDPAImpl,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.managers import forward_context
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


def _qkv(seq_len, head_dim, dtype=torch.bfloat16, heads=_GLOBAL_HEADS, batch=1):
    generator = torch.Generator(device="cuda").manual_seed(seq_len + head_dim)
    return tuple(
        torch.randn(
            batch,
            seq_len,
            heads,
            head_dim,
            device="cuda",
            dtype=dtype,
            generator=generator,
        )
        for _ in range(3)
    )


def _request(quality="exact"):
    return set_forward_context(current_timestep=0, attn_metadata=None, quality=quality)


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
        probe = torch.empty(1, 3, 1025, 128, device="cuda", dtype=torch.bfloat16)
        if _impl(heads=3, global_num_heads=24)._unsplit_query_len(probe, 1025) == 1025:
            self.skipTest("too few SMs for these shapes to split")
        for dtype in (torch.float16, torch.bfloat16):
            for head_dim in (64, 128, 256):
                # 1025 and 1154 sit at a split boundary on 132 and 148 SMs.
                for seq_len in (1025, 1154, 1536, 4608):
                    query, key, value = _qkv(seq_len, head_dim, dtype=dtype)
                    expected = F.scaled_dot_product_attention(
                        *(x.transpose(1, 2) for x in (query, key, value)),
                        scale=head_dim**-0.5,
                    ).transpose(1, 2)
                    for local_heads in (24, 12, 6, 3):
                        with self.subTest(
                            dtype=dtype,
                            head_dim=head_dim,
                            seq_len=seq_len,
                            local_heads=local_heads,
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

    @torch.no_grad()
    def test_short_key_shards_match_unsharded_attention(self):
        """Key blocks shrink with the head dim, and two key blocks leave a half-full
        unsharded layer unsplit; a shard that splits there must still be padded."""
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
        for head_dim, query_len, key_len in ((256, 1025, 100), (128, 400, 192)):
            blocks = _GLOBAL_HEADS * -(-query_len // 64)
            if sdpa._flash_num_splits(blocks, key_len, head_dim, num_sms) > 1:
                continue  # the unsharded layer splits too, so nothing is padded
            query = _qkv(query_len, head_dim)[0]
            key, value = _qkv(key_len, head_dim)[1:]
            expected = F.scaled_dot_product_attention(
                *(x.transpose(1, 2) for x in (query, key, value)),
                scale=head_dim**-0.5,
            ).transpose(1, 2)
            for local_heads in (12, 6, 3):
                with self.subTest(head_dim=head_dim, local_heads=local_heads):
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

    def test_split_boundaries_per_sm_count(self):
        """Split decisions at the boundaries of the 132-SM and 148-SM GPUs."""
        cases = (
            # num_sms, local heads, query_len, key_len, head_dim, splits
            (132, 12, 1025, 1025, 128, True),
            (132, 12, 1089, 1025, 128, False),
            (148, 12, 1154, 1154, 128, True),
            (148, 12, 1217, 1217, 128, False),
            (148, 6, 1025, 100, 256, True),
            (148, 6, 1025, 64, 256, False),
            (148, 12, 400, 192, 128, True),
            (148, 24, 400, 192, 128, False),
        )
        for num_sms, heads, query_len, key_len, head_dim, splits in cases:
            with self.subTest(num_sms=num_sms, heads=heads, query_len=query_len):
                blocks = heads * -(-query_len // 64)
                num_splits = sdpa._flash_num_splits(blocks, key_len, head_dim, num_sms)
                self.assertEqual(num_splits > 1, splits)

    def test_padding_only_when_the_shard_alone_would_split(self):
        query = torch.empty(1, 6, 1025, 128, device="cuda", dtype=torch.bfloat16)
        exact_len = _impl(heads=6, global_num_heads=24)._unsplit_query_len(query, 1025)
        if exact_len == 1025:
            self.skipTest("too few SMs for these shapes to split")
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
                self.assertEqual(impl._unsplit_query_len(query, 1025), 1025)
        with self.subTest("fp32"):
            impl = _impl(heads=6, global_num_heads=24)
            self.assertEqual(impl._unsplit_query_len(query.float(), 1025), 1025)
        with self.subTest("lossless request"), _request("lossless"):
            impl = _impl(heads=6, global_num_heads=24)
            self.assertEqual(impl._unsplit_query_len(query, 1025), 1025)
        # Stages that pass forward_batch without quality= keep the context default
        # "exact"; the request's own tier must still decide.
        for quality, expect_padding in (("lossless", False), ("exact", True)):
            request = set_forward_context(
                current_timestep=0,
                attn_metadata=None,
                forward_batch=SimpleNamespace(
                    sampling_params=SimpleNamespace(quality=quality)
                ),
            )
            with self.subTest(f"{quality} request via forward_batch"), request:
                impl = _impl(heads=6, global_num_heads=24)
                padded = impl._unsplit_query_len(query, 1025) > 1025
                self.assertEqual(padded, expect_padding)
        with self.subTest("one impl caches the shape, not the request or dtype"):
            impl = _impl(heads=6, global_num_heads=24)
            for quality, expected in (
                ("lossless", 1025),
                ("exact", exact_len),
                ("lossless", 1025),
            ):
                with _request(quality):
                    self.assertEqual(impl._unsplit_query_len(query, 1025), expected)
            with _request("exact"):
                self.assertEqual(impl._unsplit_query_len(query.float(), 1025), 1025)
        with (
            self.subTest("no forward context"),
            patch.object(forward_context, "_forward_context", None),
        ):
            impl = _impl(heads=6, global_num_heads=24)
            self.assertEqual(impl._unsplit_query_len(query, 1025), 1025)
        with self.subTest("rocm"), patch.object(torch.version, "hip", "6.4"):
            impl = _impl(heads=6, global_num_heads=24)
            self.assertEqual(impl._unsplit_query_len(query, 1025), 1025)
        with (
            self.subTest("calls that cannot split skip the request lookup"),
            patch.object(sdpa, "_request_quality_is_exact", side_effect=AssertionError),
        ):
            impl = _impl(heads=6, global_num_heads=24)
            long_query = torch.empty(
                1, 6, 8192, 128, device="cuda", dtype=torch.bfloat16
            )
            self.assertEqual(impl._unsplit_query_len(long_query, 8192), 8192)
        with self.subTest("calls with every head skip the plan cache"):
            impl = _impl(global_num_heads=_GLOBAL_HEADS)
            full = torch.empty(
                1, _GLOBAL_HEADS, 1025, 128, device="cuda", dtype=torch.bfloat16
            )
            self.assertEqual(impl._unsplit_query_len(full, 1025), 1025)
            self.assertEqual(impl._split_padding_plans, {})
        with self.subTest("layers share the split search"):
            searches = sdpa._flash_num_splits.cache_info
            sdpa._flash_num_splits.cache_clear()
            _impl(heads=6, global_num_heads=24)._unsplit_query_len(query, 1025)
            first_layer = searches().misses
            _impl(heads=6, global_num_heads=24)._unsplit_query_len(query, 1025)
            self.assertGreater(first_layer, 0)
            self.assertEqual(searches().misses, first_layer)
        with self.subTest("short keys never split"):
            impl = _impl(heads=6, global_num_heads=24)
            self.assertEqual(impl._unsplit_query_len(query, 128), 1025)
        with self.subTest("full layer also splits"):
            short = query[:, :, :64]
            self.assertEqual(
                _impl(heads=6, global_num_heads=24)._unsplit_query_len(short, 1025), 64
            )

    @torch.no_grad()
    def test_padded_output_keeps_the_unpadded_layout(self):
        """A padded call returns the contiguous [B, S, H, D] layout of an unpadded one,
        which callers such as HunyuanVideo's single-stream block view as [B, S, H * D]."""
        probe = torch.empty(1, 6, 1025, 128, device="cuda", dtype=torch.bfloat16)
        impl = _impl(heads=6, global_num_heads=_GLOBAL_HEADS)
        if impl._unsplit_query_len(probe, 1025) == 1025:
            self.skipTest("too few SMs for these shapes to split")
        for batch in (1, 2):
            with self.subTest(batch=batch):
                query, key, value = _qkv(1025, 128, batch=batch)
                shard_probe = query[:, :, :6].transpose(1, 2)
                if impl._unsplit_query_len(shard_probe, 1025) == 1025:
                    self.skipTest("too few SMs for this batch to split")
                expected = F.scaled_dot_product_attention(
                    *(x.transpose(1, 2) for x in (query, key, value)),
                    scale=128**-0.5,
                ).transpose(1, 2)
                shards = [
                    impl.forward(q, k, v, None)
                    for q, k, v in zip(
                        query.split(6, dim=2),
                        key.split(6, dim=2),
                        value.split(6, dim=2),
                    )
                ]
                for shard in shards:
                    self.assertEqual(tuple(shard.shape), (batch, 1025, 6, 128))
                    self.assertTrue(shard.is_contiguous())
                    shard.view(batch, 1025, -1)
                torch.testing.assert_close(
                    torch.cat(shards, dim=2), expected, atol=0, rtol=0
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
