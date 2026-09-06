"""Pin tests for the draft-extend CUDA-graph backend capability.

``eagle_worker_v2`` gates draft-extend graph capture on
``AttentionBackend.supports_draft_extend_cuda_graph()``. It used to be an
isinstance allowlist that omitted ``FlashAttentionBackend`` even though fa3's
DRAFT_EXTEND_V2 graph machinery exists and is unit-tested (see
``test/registered/attention/unittests/dense/test_fa3.py``). Pins: the
FlashAttention backend declares support (fa3/fa4 EAGLE draft-extend runs under
CUDA graph), support implies the in-graph metadata rebuild the WAR read-done
publication relies on, and the sliding-window-pool combination stays on the
eager path.
"""

import unittest

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=2, stage="base-b", runner_config="1-gpu-small")


class TestDraftExtendGraphCapability(CustomTestCase):
    def test_flashattention_declares_support(self):
        backend = object.__new__(FlashAttentionBackend)
        backend.use_sliding_window_kv_pool = False
        self.assertTrue(backend.supports_draft_extend_cuda_graph())
        # Backends that never declared support keep the eager default.
        base = object.__new__(AttentionBackend)
        self.assertFalse(base.supports_draft_extend_cuda_graph())

    def test_support_implies_in_graph_metadata_rebuild(self):
        # The draft-extend graph runner keys its WAR read-done publication on
        # draft_extend_metadata_captured_in_graph(): FA rebuilds metadata inside
        # the captured graph (re-reading req_to_token at replay time), so
        # whenever it declares graph support it must also declare the in-graph
        # rebuild -- otherwise the scheduler's next-batch shared-buffer writes
        # could race the replay's reads.
        backend = object.__new__(FlashAttentionBackend)
        backend.use_sliding_window_kv_pool = False
        self.assertTrue(backend.draft_extend_metadata_captured_in_graph())

    def test_swa_pool_stays_eager(self):
        backend = object.__new__(FlashAttentionBackend)
        backend.use_sliding_window_kv_pool = True
        self.assertFalse(backend.supports_draft_extend_cuda_graph())


if __name__ == "__main__":
    unittest.main()
