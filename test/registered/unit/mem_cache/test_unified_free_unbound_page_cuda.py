"""On CUDA, a free of an unbound unified page trips the device-side invariant check.

A device assert poisons the CUDA context, so each case runs in its own process.
"""

import os
import subprocess
import sys
import unittest

import torch
from test_unified_free_no_host_sync import _UNBOUND_FREE_CASES

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="nightly", runner_config="1-gpu-small")

_UNBOUND_FREE_ON_CUDA = """
import sys
import torch
sys.path.insert(0, {test_dir!r})
from test_unified_free_no_host_sync import _free_an_unbound_page
_free_an_unbound_page("cuda", {band_kind!r}, {case!r})
torch.cuda.synchronize()
"""


@unittest.skipUnless(torch.cuda.is_available(), "the CUDA free path")
class TestFreeOfAnUnboundPageOnCuda(CustomTestCase):
    def test_a_free_of_an_unbound_page_is_reported(self):
        """Every case exits nonzero and names the failed invariant; the cases
        run concurrently."""
        test_dir = os.path.dirname(os.path.abspath(__file__))
        runs = {}
        try:
            for band_kind, case, level in _UNBOUND_FREE_CASES:
                runs[(band_kind, case, level)] = subprocess.Popen(
                    [
                        sys.executable,
                        "-c",
                        _UNBOUND_FREE_ON_CUDA.format(
                            test_dir=test_dir, band_kind=band_kind, case=case
                        ),
                    ],
                    env={**os.environ, "SGLANG_INVARIANT_CHECK": str(int(level))},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            for (band_kind, case, level), run in runs.items():
                with self.subTest(band=band_kind, case=case, level=level.name):
                    output, _ = run.communicate(timeout=300)
                    self.assertNotEqual(run.returncode, 0, output)
                    self.assertIn(
                        "invariant-check FAILED [band.free_of_bound_page]", output
                    )
        finally:
            for run in runs.values():
                if run.poll() is None:
                    run.kill()
                    run.wait()
                run.stdout.close()


if __name__ == "__main__":
    unittest.main()
