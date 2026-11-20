"""Nightly: test_cuda_graph_persistence.py's EAGLE3 + full-prefill run without env pins;
the `[Foundry e2e]` lines of both runs give the with / without comparison."""

import unittest

# Module import, not the class;
# unittest would collect an imported class and run the per-commit tests here too.
import test_cuda_graph_persistence as e2e

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=320, stage="nightly", runner_config="1-gpu-large")


class TestFoundryGraphPersistenceSpecPrefillNoPins(
    e2e.TestFoundryGraphPersistenceSpecPrefill
):
    # Measured: the pins cut per-shape prep by a quarter to 40 % on FlashInfer.
    engine_env = {}


if __name__ == "__main__":
    unittest.main(verbosity=3)
