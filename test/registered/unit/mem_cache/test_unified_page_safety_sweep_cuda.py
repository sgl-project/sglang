"""The unified pool's page-safety stress harness on real CUDA streams, where
the allocator's stream waits are the only order between page moves, copies and
delayed forwards. Rerun a failing seed with
`test/manual/mem_cache/unified_page_safety_stress.py --device cuda --seeds <seed>`.
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
