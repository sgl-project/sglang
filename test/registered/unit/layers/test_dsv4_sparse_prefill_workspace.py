"""Sparse-prefill scratch replacement must fit without retaining its old owner.

The allocator limit belongs to a fresh subprocess, so it cannot affect other
CUDA tests or a serving process sharing the device.
"""

import json
import subprocess
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

_RESIZE_PROBE = """
import json
import sys
import torch
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import SparsePrefillWorkspace

case = sys.argv[1]
device = torch.device("cuda:0")
limit = 128 * 1024**2
torch.cuda.set_per_process_memory_fraction(
    limit / torch.cuda.get_device_properties(device).total_memory, device
)
torch.cuda.empty_cache()
base = torch.cuda.memory_allocated(device)
assert base < 8 * 1024**2, base
torch.cuda.reset_peak_memory_stats(device)
workspace = SparsePrefillWorkspace(device)
if case == "growth":
    workspace.get(64 * 1024).fill_(1)
else:
    workspace.get(96 * 1024, dtype=torch.uint8).fill_(1)
# Each new allocation is 96 MiB. Holding the old storage would exceed 128 MiB.
result = workspace.get(96 * 1024)
result.fill_(2)
assert result.shape == (96 * 1024, 1, 512)
assert result.dtype == torch.bfloat16
assert result[0, 0, 0].item() == result[-1, 0, -1].item() == 2
peak = torch.cuda.max_memory_allocated(device) - base
assert peak <= 100 * 1024**2, peak
print(json.dumps({"case": case, "peak_bytes": peak, "limit_bytes": limit}))
"""


class TestSparsePrefillWorkspace(CustomTestCase):
    def test_replacement_fits_without_old_scratch(self):
        """A larger scratch buffer fits, although old plus new would not."""
        for case in ("growth", "dtype"):
            with self.subTest(case=case):
                completed = subprocess.run(
                    [sys.executable, "-c", _RESIZE_PROBE, case],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                self.assertLess(result["peak_bytes"], result["limit_bytes"])

    def test_live_views_and_reuse(self):
        """Replacing the owner must preserve views held by an existing caller."""
        from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
            SparsePrefillWorkspace,
        )

        workspace = SparsePrefillWorkspace(torch.device("cuda:0"))
        old_view = workspace.get(32)
        old_view.fill_(1)
        new_view = workspace.get(64)
        new_view.fill_(2)
        torch.testing.assert_close(old_view, torch.ones_like(old_view))
        torch.testing.assert_close(new_view, torch.full_like(new_view, 2))
        smaller_view = workspace.get(16)
        self.assertEqual(smaller_view.data_ptr(), new_view.data_ptr())
        torch.testing.assert_close(smaller_view, new_view[:16])


if __name__ == "__main__":
    unittest.main()
