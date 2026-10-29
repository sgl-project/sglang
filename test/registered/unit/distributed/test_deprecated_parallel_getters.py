import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.distributed import communication_op, parallel_state
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDeprecatedGettersUnderCompile(CustomTestCase):
    def test_collective_compiles_with_fullgraph(self):
        """Deprecated parallel getters must not graph-break under torch.compile."""
        compiled = torch.compile(
            communication_op.tensor_model_parallel_all_reduce,
            backend="eager",
            fullgraph=True,
        )
        tp_group = SimpleNamespace(all_reduce=lambda x: x * 2)
        # A getter that has already warned skips the frame lookup.
        with (
            patch.object(parallel_state, "_ALREADY_WARNED", set()),
            get_parallel().override(tp_group=tp_group),
        ):
            out = compiled(torch.ones(2))
        self.assertTrue(torch.equal(out, torch.full((2,), 2.0)))


if __name__ == "__main__":
    unittest.main()
