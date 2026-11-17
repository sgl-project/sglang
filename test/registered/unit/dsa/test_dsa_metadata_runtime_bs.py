"""Unit tests for kernels/ops/attention/dsa/metadata.py"""

import ast
import inspect
import unittest

from sglang.kernels.ops.attention.dsa import metadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KERNELS = (
    "_fused_dsa_decode_metadata_kernel",
    "_fused_dsa_target_verify_metadata_kernel",
    "_fused_dsa_draft_extend_metadata_kernel",
)


class TestDSAMetadataRuntimeBatchSize(CustomTestCase):
    def test_bs_is_not_constexpr(self):
        """A constexpr `bs` makes Triton compile a new kernel for every batch size."""
        tree = ast.parse(inspect.getsource(metadata))
        fns = {
            n.name: n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in _KERNELS
        }
        self.assertEqual(set(fns), set(_KERNELS))
        for name, fn in fns.items():
            with self.subTest(kernel=name):
                bs = next(a for a in fn.args.args if a.arg == "bs")
                annotation = "" if bs.annotation is None else ast.unparse(bs.annotation)
                self.assertNotIn("constexpr", annotation)


if __name__ == "__main__":
    unittest.main()
