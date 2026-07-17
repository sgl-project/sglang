"""Regression tests for draft-extend CUDA graph bucket selection.

The runner is constructed via ``__new__`` and execution stops immediately after
bucket selection, allowing the request-count invariant to be tested on CPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _BucketSelected(Exception):
    pass


class TestEagleDraftExtendCudaGraphRunner(CustomTestCase):
    def test_graph_bucket_uses_raw_request_count(self):
        # 7 requests * 5 draft tokens = 35 scaled tokens:
        # execute() must select the replay bucket from the raw request count (7),
        # matching can_run_graph(), not from the scaled token count (35).
        for algorithm, is_eagle in (("eagle", True), ("standalone", False)):
            with self.subTest(algorithm=algorithm):
                runner = EAGLEDraftExtendCudaGraphRunner.__new__(
                    EAGLEDraftExtendCudaGraphRunner
                )
                runner.require_mlp_tp_gather = True
                runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
                runner.buffers = SimpleNamespace()
                runner.capture_bs = [1, 2, 4, 8]
                runner.captured_req_width = 5
                runner.model_runner = SimpleNamespace(
                    spec_algorithm=SimpleNamespace(
                        is_eagle=MagicMock(return_value=is_eagle)
                    )
                )
                # Stop after bucket selection; downstream buffers are irrelevant.
                runner._pad_to_bucket = MagicMock(side_effect=_BucketSelected)

                forward_batch = SimpleNamespace(
                    out_cache_loc=object(),
                    batch_size=7,
                    input_ids=torch.empty(35),
                    global_num_tokens_cpu=[35],
                    original_global_num_tokens_cpu=[7],
                )

                with self.assertRaises(_BucketSelected):
                    runner.execute(forward_batch)

                runner._pad_to_bucket.assert_called_once_with(7, runner.capture_bs)


if __name__ == "__main__":
    unittest.main()
