"""HC fused combine rows: which row counts SGLANG_HC_FUSED_COMBINE_MAX_ROWS hands to the fused boundary (CPU)."""

import importlib.util
import unittest

from sglang.srt.environ import envs
from sglang.srt.layers import hyperconnection
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _load_with(max_rows):
    """A private copy of the module imported under SGLANG_HC_FUSED_COMBINE_MAX_ROWS=max_rows (read at import)."""
    spec = importlib.util.spec_from_file_location("_hc_copy", hyperconnection.__file__)
    module = importlib.util.module_from_spec(spec)
    if max_rows is None:
        spec.loader.exec_module(module)
    else:
        with envs.SGLANG_HC_FUSED_COMBINE_MAX_ROWS.override(max_rows):
            spec.loader.exec_module(module)
    return module


def _rows_taken(module):
    layer = object.__new__(module.GatedResidual)
    layer._fused_combine_ok = True
    return [rows for rows in range(1, 300) if layer._fused_combine_rows(rows)]


class TestFusedCombineRows(CustomTestCase):
    def test_default_is_off(self):
        if envs.SGLANG_HC_FUSED_COMBINE_MAX_ROWS.is_set():
            self.skipTest("SGLANG_HC_FUSED_COMBINE_MAX_ROWS is set in this environment")
        module = _load_with(None)
        self.assertLess(module._FUSED_COMBINE_MAX_ROWS, module._FUSED_COMBINE_MIN_ROWS)
        self.assertEqual(_rows_taken(module), [])

    def test_rows_start_above_the_gate_rows_and_stop_at_the_kernel_bound(self):
        """Rows 1-32 keep their own gate paths; the fused norm kernel admits at most 128 rows."""
        self.assertEqual(_rows_taken(_load_with(64)), list(range(33, 65)))
        self.assertEqual(_rows_taken(_load_with(128)), list(range(33, 129)))
        self.assertEqual(_rows_taken(_load_with(1000)), list(range(33, 129)))
        self.assertEqual(_rows_taken(_load_with(32)), [])


if __name__ == "__main__":
    unittest.main()
