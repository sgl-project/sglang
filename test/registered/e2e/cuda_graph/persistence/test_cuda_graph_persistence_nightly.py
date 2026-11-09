"""Nightly comparison run of --cuda-graph-persistence: the per-commit EAGLE3 +
full-backend-prefill configuration of test_cuda_graph_persistence.py without
the environment pins of Foundry's recipes. Same assertions; the printed
`[Foundry e2e] ... launch -> ready` and `LOAD capture loop` lines of the two
files give the with / without comparison (the pins cut SGLang's per-shape
capture-loop prep by a quarter to 40 % on the FlashInfer backend; see
recipe/sglang/set_env.sh in Foundry). Needs the optional dependency
sglang[foundry].

Run:  python3 test/registered/e2e/cuda_graph/persistence/test_cuda_graph_persistence_nightly.py
"""

import unittest

# Module import (not the class): unittest would collect an imported class and
# run the per-commit tests here as well.
import test_cuda_graph_persistence as e2e

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=320, stage="nightly", runner_config="1-gpu-large")


class TestFoundryGraphPersistenceSpecPrefillNoPins(
    e2e.TestFoundryGraphPersistenceSpecPrefill
):
    engine_env = {}


if __name__ == "__main__":
    unittest.main(verbosity=3)
