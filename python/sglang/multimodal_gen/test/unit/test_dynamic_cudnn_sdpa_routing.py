import unittest
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    DynamicCudnnSDPAImpl,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
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
    def _uses_cudnn(self, q_len: int, kv_len: int, *, graph: bool = False) -> bool:
        impl = SimpleNamespace(
            causal=False, _cudnn_failed=False, _is_sm100=True, _is_sm103=False
        )
        with enable_breakable_cuda_graph() if graph else nullcontext():
            return DynamicCudnnSDPAImpl._use_cudnn_sdpa(impl, *_qkv(q_len, kv_len))

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


if __name__ == "__main__":
    unittest.main()
