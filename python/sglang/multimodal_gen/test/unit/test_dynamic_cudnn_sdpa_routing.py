import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    DynamicCudnnSDPAImpl,
    SDPAImpl,
)


def _qkv(q_len: int, kv_len: int, heads: int = 32, head_dim: int = 64):
    def tensor(seq_len: int):
        return SimpleNamespace(
            shape=(1, seq_len, heads, head_dim),
            dtype=torch.bfloat16,
            device=SimpleNamespace(type="cuda"),
        )

    return tensor(q_len), tensor(kv_len), tensor(kv_len)


class TestDynamicCudnnSDPARouting(unittest.TestCase):
    def _uses_cudnn(self, q_len: int, kv_len: int) -> bool:
        impl = SimpleNamespace(
            causal=False, _cudnn_failed=False, _is_sm100=True, _is_sm103=False
        )
        return DynamicCudnnSDPAImpl._use_cudnn_sdpa(impl, *_qkv(q_len, kv_len))

    def test_blackwell_keeps_cudnn_when_both_sides_fill_a_tile(self):
        for q_len, kv_len in ((128, 128), (1536, 1024), (6144, 6144), (192, 6144)):
            self.assertTrue(self._uses_cudnn(q_len, kv_len), (q_len, kv_len))

    def test_blackwell_routes_a_side_shorter_than_a_tile_to_fa4(self):
        for q_len, kv_len in ((126, 1024), (1536, 126), (126, 1536), (32, 4096)):
            self.assertFalse(self._uses_cudnn(q_len, kv_len), (q_len, kv_len))

    def test_blackwell_keeps_cudnn_when_both_sides_are_short(self):
        self.assertTrue(self._uses_cudnn(126, 126))


class _IntoOnLongSegments(SDPAImpl):
    """forward doubles q; forward_into serves segments of 4+ rows."""

    def __init__(self):
        self.calls = []

    def forward(self, query, key, value, attn_metadata):
        self.calls.append(("forward", query.shape[1]))
        return query * 2

    def forward_into(self, query, key, value, out, attn_metadata):
        if query.shape[1] < 4:
            return False
        self.calls.append(("into", query.shape[1]))
        out.copy_(query * 2)
        return True


class TestVarlenSegmentsWriteInPlace(unittest.TestCase):
    def test_segments_use_forward_into_and_fall_back_to_forward(self):
        impl = _IntoOnLongSegments()
        q = torch.randn(9, 2, 8)
        out = impl.forward_varlen(
            q, q, q, cu_seqlens=torch.tensor([0, 6, 9]), max_seqlen=6
        )
        torch.testing.assert_close(out, q * 2, rtol=0, atol=0)
        self.assertEqual(impl.calls, [("into", 6), ("forward", 3)])

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 3),
        "FA4 self-attention route is sm_103 only",
    )
    def test_fa4_segment_written_in_place_is_bit_identical(self):
        used, total, heads = 16448, 16512, 4
        impl = DynamicCudnnSDPAImpl(
            num_heads=heads, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        g = torch.Generator(device="cuda").manual_seed(0)
        qkv = torch.randn(
            total, 3 * heads, 128, device="cuda", dtype=torch.bfloat16, generator=g
        )
        q, k, v = qkv.split(heads, dim=1)
        out = impl.forward_varlen(
            q,
            k,
            v,
            cu_seqlens=torch.tensor([0, used, total], device="cuda", dtype=torch.int32),
            max_seqlen=used,
            cu_seqlens_host=(0, used, total),
        )
        ref = torch.cat(
            [
                impl.forward(*(t[a:b].unsqueeze(0) for t in (q, k, v)), None)[0]
                for a, b in ((0, used), (used, total))
            ]
        )
        self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
