"""Received PP proxy tensors must be recorded on the forward stream before the
forward reads them, so the caching allocator cannot hand their memory to a
later recv while that forward is still queued."""

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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tensor(is_cpu):
    tensor = MagicMock(spec=torch.Tensor)
    tensor.is_cpu = is_cpu
    return tensor


class TestPPProxyRecordStream(CustomTestCase):
    def test_launch_records_received_proxy_on_forward_stream(self):
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
        order = []

        def run_batch(batch, pp_proxy_tensors):
            order.append("run_batch")
            return SimpleNamespace(
                can_run_cuda_graph=True,
                spec_verify_out_cache_loc=None,
                fpm_timing=None,
            )

        scheduler.run_batch = run_batch
        hidden, residual, cpu_meta = _tensor(False), _tensor(False), _tensor(True)
        hidden.record_stream.side_effect = lambda stream: order.append("record")
        proxy = PPProxyTensors(
            {
                "hidden_states": hidden,
                "residual": residual,
                "cpu_meta": cpu_meta,
                "__msg_type__": "proxy",
            }
        )
        batch = SimpleNamespace(
            reqs=[], spec_algorithm=SimpleNamespace(is_none=lambda: True)
        )

        scheduler._pp_launch_batch(0, batch, proxy, [None], deque())

        hidden.record_stream.assert_called_once_with(forward_stream)
        residual.record_stream.assert_called_once_with(forward_stream)
        cpu_meta.record_stream.assert_not_called()
        self.assertEqual(order, ["record", "run_batch"])


if __name__ == "__main__":
    unittest.main()
