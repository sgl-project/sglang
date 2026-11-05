"""MLX hybrid admission through the shared acquire entry point.

The MLX request table binds a request's auxiliary-state slot in its own
``alloc``, with the row, so the state storage it carries has nothing left to
bind. Admission used to fail on the first hybrid request, after the row and
the slot were taken, because the generic acquire read ReplaySSM fields the MLX
pool does not have. MLX-gated because the table's module imports mlx.core.
"""

from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.test.ci.ci_register import register_mlx_ci
from sglang.test.test_utils import CustomTestCase

register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_MLX = importlib.util.find_spec("mlx") is not None

if _HAS_MLX:
    from sglang.srt.hardware_backend.mlx.kv_cache.auxiliary_state import (
        MlxAuxiliaryStateReqToTokenPool,
    )


def _req():
    return SimpleNamespace(kv=ReqKvInfo())


@unittest.skipUnless(_HAS_MLX, "requires mlx")
class TestMlxAuxiliaryStateAcquire(CustomTestCase):
    def setUp(self):
        self.table = MlxAuxiliaryStateReqToTokenPool(
            size=4,
            max_context_len=16,
            device="cpu",
            enable_memory_saver=False,
            auxiliary_state_size=4,
        )
        self.slots = self.table.auxiliary_state_pool

    def test_first_admission_binds_the_row_and_one_slot(self):
        req = _req()
        rows = acquire_rows(self.table, self.table.state, [req])
        self.assertEqual(len(rows), 1)
        self.assertTrue(req.kv.holds_mamba)
        self.assertEqual(self.slots.available_size(), 3)
        self.assertEqual(
            int(self.table.get_auxiliary_state_indices(rows)[0]),
            int(req.kv.mamba_pool_idx),
        )

    def test_a_request_holding_a_slot_keeps_it(self):
        req = _req()
        held = self.slots.alloc(1)[0]
        req.kv.mamba_pool_idx = held
        rows = acquire_rows(self.table, self.table.state, [req])
        self.assertEqual(int(req.kv.mamba_pool_idx), int(held))
        self.assertEqual(self.slots.available_size(), 3)
        self.assertEqual(
            int(self.table.get_auxiliary_state_indices(rows)[0]), int(held)
        )


if __name__ == "__main__":
    unittest.main()
