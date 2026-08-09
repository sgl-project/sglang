"""Regression tests for asynchronous state-capture result lifetimes."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.utils import _async_d2h
from sglang.srt.state_capturer.base import BaseTopkCapturer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class TestStateCapturer(CustomTestCase):
    def test_copy_stream_reads_snapshot_after_next_forward_reuses_buffers(self):
        device = torch.device("cuda")
        capturer = BaseTopkCapturer(
            num_tokens=16,
            max_batch_size=2,
            num_layers=2,
            topk_size=2,
            device=device,
            name="test",
        )

        first_topk = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32, device=device)
        capturer.capture(layer_id=0, topk_indices=first_topk)
        out_cache_loc = torch.tensor([5, 6], device=device)
        first_result = capturer.on_forward_end(
            forward_batch=SimpleNamespace(out_cache_loc=out_cache_loc),
            can_run_graph=False,
            cuda_graph_batch=None,
            no_copy_to_cpu=True,
        )
        self.assertIsNotNone(first_result)

        forward_stream = torch.cuda.current_stream(device)
        copy_stream = torch.cuda.Stream(device=device)
        copy_done = torch.cuda.Event()

        # This is the scheduler's initial dependency: result copies may start
        # after the first forward, without ordering them before the next one.
        copy_stream.wait_stream(forward_stream)

        # Force the adverse, but valid, cross-stream execution order in which
        # the next forward reuses the sources before the result copy reads them.
        capturer.capture(
            layer_id=0,
            topk_indices=torch.tensor(
                [[9, 10], [11, 12]], dtype=torch.int32, device=device
            ),
        )
        out_cache_loc.copy_(torch.tensor([7, 8], device=device))
        sources_reused = torch.cuda.Event()
        sources_reused.record(forward_stream)

        copy_stream.wait_event(sources_reused)
        with torch.cuda.stream(copy_stream):
            first_result.map_device_tensors(_async_d2h)
            copy_done.record()

        copy_done.synchronize()
        first_result.finalize()

        torch.testing.assert_close(
            capturer.host_cache.buffer[torch.tensor([5, 6]), 0],
            first_topk.cpu(),
        )
        torch.testing.assert_close(
            capturer.host_cache.buffer[torch.tensor([7, 8]), 0],
            torch.zeros((2, 2), dtype=torch.int32),
        )


if __name__ == "__main__":
    unittest.main()
