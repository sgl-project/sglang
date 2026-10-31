"""Received PP proxy tensors must be recorded on the forward stream before the
forward reads them, so the caching allocator cannot hand their memory to a
later recv while that forward is still queued.

The tensors are allocated by ``recv_tensor_dict`` on the PP comm / schedule
stream and read by ``run_batch`` on the forward stream. Once the scheduler
drops its reference, the allocator only knows about the stream that allocated
them, so without ``record_stream`` the block can be reused by the next recv.

Run inside the sglang container:

    python -m pytest test/registered/unit/managers/test_pp_proxy_record_stream.py -v
"""

import unittest
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors  # noqa: E402

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _fake_tensor(is_cpu: bool) -> MagicMock:
    """A tensor-shaped stand-in: CPU CI has no CUDA tensors to hand to the launcher."""
    tensor = MagicMock(spec=torch.Tensor)
    tensor.is_cpu = is_cpu
    return tensor


def _result():
    return SimpleNamespace(
        can_run_cuda_graph=True,
        spec_verify_out_cache_loc=None,
        fpm_timing=None,
    )


class TestPPProxyRecordStream(CustomTestCase):
    def _make_scheduler(self, run_batch):
        forward_stream = object()
        scheduler = SchedulerPPMixin()
        scheduler.forward_stream = forward_stream
        scheduler.forward_stream_ctx = nullcontext()
        scheduler._pp_wait_forward_dependencies = lambda: None
        scheduler.pp_group = SimpleNamespace(is_last_rank=False)
        scheduler.device_module = SimpleNamespace(
            Event=lambda: SimpleNamespace(record=lambda stream: None),
            current_stream=lambda: forward_stream,
        )
        scheduler.run_batch = run_batch
        return scheduler, forward_stream

    def _make_batch(self):
        return SimpleNamespace(
            reqs=[], spec_algorithm=SimpleNamespace(is_none=lambda: True)
        )

    def test_received_proxy_recorded_before_run_batch(self):
        order = []
        scheduler, forward_stream = self._make_scheduler(
            lambda batch, proxy: order.append("run_batch") or _result()
        )
        hidden, residual, cpu_meta = (
            _fake_tensor(False),
            _fake_tensor(False),
            _fake_tensor(True),
        )
        hidden.record_stream.side_effect = lambda stream: order.append("record")
        residual.record_stream.side_effect = lambda stream: order.append("record")
        proxy = PPProxyTensors(
            {
                "hidden_states": hidden,
                "residual": residual,
                "cpu_meta": cpu_meta,
                "__msg_type__": "proxy",
            }
        )
        scheduler._pp_launch_batch(0, self._make_batch(), proxy, [None], deque())

        # Every device tensor is recorded on the forward stream, exactly once,
        # and before the forward that reads it is submitted. CPU tensors and
        # the string message tag are skipped.
        hidden.record_stream.assert_called_once_with(forward_stream)
        residual.record_stream.assert_called_once_with(forward_stream)
        cpu_meta.record_stream.assert_not_called()
        self.assertEqual(order, ["record", "record", "run_batch"])

    def test_launch_without_proxy_tensors(self):
        # Rank 0 and prebuilt microbatches pass None; the launch must not touch it.
        seen = []
        scheduler, _ = self._make_scheduler(
            lambda batch, proxy: seen.append(proxy) or _result()
        )

        scheduler._pp_launch_batch(0, self._make_batch(), None, [None], deque())

        self.assertEqual(seen, [None])


if __name__ == "__main__":
    unittest.main()
