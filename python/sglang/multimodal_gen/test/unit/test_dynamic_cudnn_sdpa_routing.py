import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    DynamicCudnnSDPAImpl,
    SDPAImpl,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    get_forward_context,
    set_forward_context,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)


def _qkv(q_len: int, kv_len: int, heads: int = 32, head_dim: int = 64, kv_heads=None):
    def tensor(seq_len: int, num_heads: int):
        return SimpleNamespace(
            shape=(1, seq_len, num_heads, head_dim),
            dtype=torch.bfloat16,
            device=SimpleNamespace(type="cuda"),
        )

    kv_heads = heads if kv_heads is None else kv_heads
    return tensor(q_len, heads), tensor(kv_len, kv_heads), tensor(kv_len, kv_heads)


class TestDynamicCudnnSDPARouting(unittest.TestCase):
    def _uses_cudnn(
        self,
        q_len: int,
        kv_len: int,
        kv_heads=None,
        is_sm100: bool = True,
        *,
        graph: bool = False,
        quality: str = "lossless",
    ) -> bool:
        impl = SimpleNamespace(
            causal=False, _cudnn_failed=False, _is_sm100=is_sm100, _is_sm103=False
        )
        with (
            set_forward_context(0, None, quality=quality),
            enable_breakable_cuda_graph() if graph else nullcontext(),
        ):
            return DynamicCudnnSDPAImpl._use_cudnn_sdpa(
                impl, *_qkv(q_len, kv_len, kv_heads=kv_heads)
            )

    def test_blackwell_keeps_grouped_query_attention_on_cudnn(self):
        # Cosmos3: 32 query heads over 8 KV heads, self- and cross-attention.
        self.assertTrue(self._uses_cudnn(6144, 6144, kv_heads=8))
        self.assertTrue(self._uses_cudnn(6144, 512, kv_heads=8))

    def test_uneven_head_grouping_and_pre_blackwell_gqa_stay_on_fa(self):
        self.assertFalse(self._uses_cudnn(6144, 6144, kv_heads=5))
        self.assertFalse(self._uses_cudnn(1024, 1024, kv_heads=8, is_sm100=False))

    def test_blackwell_keeps_cudnn_when_both_sides_fill_a_tile(self):
        for q_len, kv_len in ((128, 128), (1536, 1024), (6144, 6144), (192, 6144)):
            self.assertTrue(self._uses_cudnn(q_len, kv_len), (q_len, kv_len))

    def test_blackwell_subtile_route_depends_on_execution_mode(self):
        for q_len, kv_len in ((126, 1024), (1536, 126), (126, 1536), (32, 4096)):
            self.assertTrue(self._uses_cudnn(q_len, kv_len), (q_len, kv_len))
            self.assertFalse(
                self._uses_cudnn(q_len, kv_len, graph=True), (q_len, kv_len)
            )
            # Graph breaks replay eagerly after leaving the capture scope.
            self.assertTrue(self._uses_cudnn(q_len, kv_len), (q_len, kv_len))

    def test_blackwell_keeps_cudnn_when_both_sides_are_short(self):
        self.assertTrue(self._uses_cudnn(126, 126))

    def test_exact_keeps_reference_subtile_route_in_both_modes(self):
        for graph in (False, True):
            for q_len, kv_len in ((26, 1536), (1536, 26), (126, 1536)):
                self.assertFalse(
                    self._uses_cudnn(q_len, kv_len, graph=graph, quality="exact")
                )

    def test_missing_context_keeps_reference_route(self):
        impl = SimpleNamespace(
            causal=False, _cudnn_failed=False, _is_sm100=True, _is_sm103=False
        )
        self.assertFalse(DynamicCudnnSDPAImpl._use_cudnn_sdpa(impl, *_qkv(126, 1536)))
        self.assertTrue(self._uses_cudnn(126, 1536, quality="high"))

    def test_quality_context_is_explicit_and_restores(self):
        batch = SimpleNamespace(extra={})
        with set_forward_context(0, None, forward_batch=batch, quality="lossless"):
            self.assertEqual(get_forward_context().quality, "lossless")
            with set_forward_context(0, None, quality="exact"):
                self.assertEqual(get_forward_context().quality, "exact")
            self.assertEqual(get_forward_context().quality, "lossless")
        with set_forward_context(0, None):
            self.assertEqual(get_forward_context().quality, "exact")


def _impl(capability: tuple[int, int], head_size: int = 128) -> DynamicCudnnSDPAImpl:
    # the constructor also pins the process-wide FA version; keep it untouched
    with (
        patch("torch.cuda.get_device_capability", return_value=capability),
        patch(
            "sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.set_fa_ver"
        ),
    ):
        return DynamicCudnnSDPAImpl(
            num_heads=2, head_size=head_size, causal=False, softmax_scale=0.125
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestDynamicCudnnSelection(unittest.TestCase):
    """Which kernel the constructed impl picks for each Blackwell variant."""

    def test_sm103_long_self_attention_takes_fa4(self):
        impl = _impl((10, 3))
        self.assertFalse(impl._use_cudnn_sdpa(*_qkv(16384, 16384, 2, 128)))
        self.assertFalse(impl._use_cudnn_sdpa(*_qkv(75600, 75600, 2, 128)))

    def test_sm103_short_cross_and_hd64_keep_cudnn(self):
        impl = _impl((10, 3))
        self.assertTrue(impl._use_cudnn_sdpa(*_qkv(8192, 8192, 2, 128)))
        self.assertTrue(impl._use_cudnn_sdpa(*_qkv(37800, 512, 2, 128)))
        impl64 = _impl((10, 3), head_size=64)
        self.assertTrue(impl64._use_cudnn_sdpa(*_qkv(18900, 18900, 2, 64)))

    def test_sm100_keeps_cudnn(self):
        impl = _impl((10, 0))
        self.assertTrue(impl._use_cudnn_sdpa(*_qkv(75600, 75600, 2, 128)))


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
