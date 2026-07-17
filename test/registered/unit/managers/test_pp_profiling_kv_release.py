import ast
import inspect
import pathlib
import unittest

from sglang.srt.managers import scheduler_pp_mixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _free_site_slice_bounds():
    source = pathlib.Path(inspect.getsourcefile(scheduler_pp_mixin)).read_text()
    tree = ast.parse(source)

    bounds = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if ast.unparse(node.value) != "self.req_to_token_pool.req_to_token":
            continue
        if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) != 2:
            continue
        index, token_range = node.slice.elts
        if ast.unparse(index) != "req.req_pool_idx":
            continue
        if not isinstance(token_range, ast.Slice):
            continue
        bounds.append(
            (
                None if token_range.lower is None else ast.unparse(token_range.lower),
                None if token_range.upper is None else ast.unparse(token_range.upper),
            )
        )
    return bounds


class TestPPProfilingKVRelease(CustomTestCase):
    def test_profiling_free_spans_the_whole_recorded_allocation(self):
        """The profiler frees [0, kv_allocated_len), not the prompt length it happens to have asked for."""
        self.assertEqual(_free_site_slice_bounds(), [(None, "req.kv.kv_allocated_len")])


if __name__ == "__main__":
    unittest.main()
