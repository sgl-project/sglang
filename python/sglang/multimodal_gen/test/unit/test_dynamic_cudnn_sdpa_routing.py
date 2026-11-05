import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    DynamicCudnnSDPAImpl,
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


if __name__ == "__main__":
    unittest.main()
