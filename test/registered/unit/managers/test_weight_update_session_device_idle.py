"""A weight-update session opens only on a device that has finished the work already launched on it."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.io_struct import BeginWeightUpdateReqInput
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

# about a second on current GPUs
_FORWARD_CYCLES = 2_000_000_000


class TestWeightUpdateSessionOpensOnAnIdleDevice(CustomTestCase):
    def test_begin_returns_after_the_work_still_running_on_the_device(self):
        """Peers write the weights over RDMA as soon as begin returns; the forward an in_place pause leaves
        running on another stream would read them mid-write."""
        runner = SimpleNamespace(
            weight_updater=SimpleNamespace(begin_weight_update=lambda: None)
        )
        manager = SchedulerWeightUpdaterManager(
            tp_worker=SimpleNamespace(
                model_runner=runner,
                weight_update_runners=lambda: [("target", runner)],
                deserialize_own_rank=lambda payloads: [],
            ),
            draft_worker=None,
            tp_cpu_group=None,
            memory_saver_adapter=None,
            flush_cache=lambda **kwargs: True,
            is_fully_idle=lambda **kwargs: True,
            scheduler=SimpleNamespace(),
        )
        forward_stream = torch.cuda.Stream()
        with torch.cuda.stream(forward_stream):
            torch.cuda._sleep(_FORWARD_CYCLES)

        with patch("torch.distributed.barrier"):
            output = manager.begin_weight_update(BeginWeightUpdateReqInput())

        self.assertTrue(output.success)
        self.assertTrue(forward_stream.query())


if __name__ == "__main__":
    unittest.main()
