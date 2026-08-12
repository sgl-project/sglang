import sys
import unittest
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import MagicMock, patch

from sglang.srt.distributed.device_communicators.symm_mem_gather import SymmMemGather
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_MARKER_MODULE = "sglang.srt.distributed.device_communicators.symm_mem_marker"


class TestSymmMemGather(unittest.TestCase):
    def test_incomplete_snapshot_retries(self):
        gatherer, marker = self._gatherer(ready_values=[0, 1])

        with (
            patch.dict(sys.modules, {_MARKER_MODULE: marker}),
            patch("torch.cuda.stream", return_value=nullcontext()),
        ):
            gatherer.gather(MagicMock())

        self.assertEqual(marker.snapshot_rows_acquire.call_count, 2)

    @staticmethod
    def _gatherer(ready_values):
        gatherer = object.__new__(SymmMemGather)
        gatherer._slot = 0
        gatherer._generation = 0
        gatherer._stream = MagicMock()
        gatherer._host_in = MagicMock()
        gatherer._staging = MagicMock()
        gatherer._host_out = MagicMock()
        gatherer._host_ready = MagicMock()
        gatherer._host_ready.all.return_value.item.side_effect = ready_values
        gatherer._region = [MagicMock()]
        gatherer._marker_region = [MagicMock()]
        gatherer._row_snapshot = MagicMock()
        gatherer._ready = MagicMock()
        gatherer._peer_row_ptrs = [[MagicMock()]]
        gatherer._peer_marker_ptrs = [[MagicMock()]]

        marker = ModuleType(_MARKER_MODULE)
        marker.copy_row_and_publish = MagicMock()
        marker.snapshot_rows_acquire = MagicMock()
        return gatherer, marker


if __name__ == "__main__":
    unittest.main()
