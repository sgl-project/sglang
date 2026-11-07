"""The unified pool's page-safety stress harness on a CUDA device (weekly): 64
generated sequences of 300 operations whose page moves, zeroing and copies run
on real streams, with forwards and transfers as delayed kernels, so that the
waits the allocator issues are the only order between them. Each rank's
payload is checked against what its last writer wrote. A failure names its
seed: rerun it with
`python test/manual/mem_cache/unified_page_safety_stress.py --device cuda
--seeds <seed> --ops 300`.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_page_safety_harness import run_seed_cuda

register_cuda_ci(est_time=150, stage="weekly", runner_config="1-gpu-large")

_SEEDS = range(64)
_OPS = 300


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPageSafetySweepOnCuda(CustomTestCase):
    def test_sweep_keeps_pages_safe_on_cuda(self):
        failures = []
        for seed in _SEEDS:
            result = run_seed_cuda(seed, _OPS)
            if result.failure is not None:
                failures.append(
                    f"seed {seed} failed at operation {result.failed_at}: "
                    f"{result.failure}"
                )
        if failures:
            self.fail("\n".join(failures))


if __name__ == "__main__":
    unittest.main()
