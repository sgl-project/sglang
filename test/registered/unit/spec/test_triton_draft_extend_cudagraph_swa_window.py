"""Regression test for the EAGLE draft-extend cuda-graph SWA window buffers.

White-box unit test: it builds a bare ``TritonAttnBackend`` via ``__new__`` and
stubs only the cuda-graph buffers that ``_build_cuda_graph_forward_metadata``
reads, so the draft-extend metadata can be exercised on CPU without a captured
graph or a real model. It is coupled to that method's internals on purpose,
which is the price of testing the metadata shape in isolation.

The contract under test: on a hybrid sliding-window model the draft-extend
cuda-graph metadata must expose the swa-pool window buffers (a non-None
``window_kv_indices``) exactly like target_verify does. When it leaves them
None the SWA layers fall back to the full-pool kv_indices and read out of
bounds of the smaller swa K/V buffer. Non-SWA models must keep the window
fields None.
"""

import unittest

import torch

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS = 3


class TestTritonDraftExtendCudaGraphSWAWindow(CustomTestCase):
    def _build_backend(self, sliding_window_size):
        backend = TritonAttnBackend.__new__(TritonAttnBackend)
        backend.sliding_window_size = sliding_window_size
        backend.num_draft_tokens = 4
        backend.speculative_num_steps = 3
        backend.kv_indptr = torch.zeros(BS + 1, dtype=torch.int32)
        backend.qo_indptr = torch.zeros(BS + 1, dtype=torch.int32)
        backend.window_kv_indptr = torch.zeros(BS + 1, dtype=torch.int32)
        backend.cuda_graph_kv_indices = torch.zeros(16, dtype=torch.int64)
        backend.cuda_graph_window_kv_indices = torch.zeros(16, dtype=torch.int64)
        backend.cuda_graph_window_num_kv_splits = torch.zeros(BS, dtype=torch.int32)
        backend.cuda_graph_window_kv_offsets = torch.zeros(BS, dtype=torch.int32)
        return backend

    def test_draft_extend_swa_exposes_window_buffers(self):
        backend = self._build_backend(sliding_window_size=128)
        md = backend._build_cuda_graph_forward_metadata(
            BS, ForwardMode.DRAFT_EXTEND_V2, spec_info=None, swa_out_cache_loc=None
        )
        # The draft step must be handed the swa-pool window buffers instead of
        # None, so the SWA layers stay on the swa buffer rather than indexing it
        # with full-pool locations.
        self.assertIsNotNone(md.window_kv_indices)
        self.assertIs(md.window_kv_indices, backend.cuda_graph_window_kv_indices)
        self.assertIsNotNone(md.window_kv_indptr)
        self.assertIs(md.window_kv_offsets, backend.cuda_graph_window_kv_offsets)
        self.assertIsNotNone(md.window_num_kv_splits)

    def test_draft_extend_without_swa_keeps_window_none(self):
        for size in (None, 0):
            backend = self._build_backend(sliding_window_size=size)
            md = backend._build_cuda_graph_forward_metadata(
                BS, ForwardMode.DRAFT_EXTEND_V2, spec_info=None, swa_out_cache_loc=None
            )
            self.assertIsNone(md.window_kv_indices, msg=f"size={size}")
            self.assertIsNone(md.window_kv_indptr, msg=f"size={size}")
            self.assertIsNone(md.window_kv_offsets, msg=f"size={size}")
            self.assertIsNone(md.window_num_kv_splits, msg=f"size={size}")


if __name__ == "__main__":
    unittest.main()
