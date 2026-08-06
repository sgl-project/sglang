"""Unit tests for triton_load_watch — no server, no model loading."""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import triton_load_watch
from sglang.test.test_utils import CustomTestCase


@triton.jit
def _probe_kernel(x_ptr, C: tl.constexpr):
    # Each constexpr C is a distinct specialization -> a fresh device load.
    tl.store(x_ptr + tl.program_id(0), C)


class TestTritonLoadWatch(CustomTestCase):
    def tearDown(self):
        # The watch is process-global; disarm so later tests in the same
        # pytest process don't warn on their own first-use kernel loads.
        triton_load_watch._serving_started = False

    def test_load_after_ready_warns_and_crashes(self):
        triton_load_watch.install()
        x = torch.zeros(4, device="cuda", dtype=torch.int32)

        # Loads during init (before serving starts) are silent.
        with self.assertNoLogs(triton_load_watch.logger, level="WARNING"):
            _probe_kernel[(1,)](x, C=1)

        triton_load_watch.mark_serving_started()

        # First use of a new specialization after ready warns with the name.
        with (
            patch.object(
                torch.cuda, "mem_get_info", return_value=(128 << 20, 80 << 30)
            ),
            self.assertLogs(triton_load_watch.logger, level="WARNING") as logs,
        ):
            _probe_kernel[(1,)](x, C=2)
        self.assertTrue(any("free device mem" in line for line in logs.output))

        # Already-loaded specializations stay silent.
        with self.assertNoLogs(triton_load_watch.logger, level="WARNING"):
            _probe_kernel[(1,)](x, C=2)

        # Crash mode turns the next late load into a hard error.
        with envs.SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY.override(True):
            with self.assertRaises(RuntimeError):
                _probe_kernel[(1,)](x, C=3)


if __name__ == "__main__":
    unittest.main()
