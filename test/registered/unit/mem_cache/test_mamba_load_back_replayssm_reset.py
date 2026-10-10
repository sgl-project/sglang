import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _component(write_pos, slot):
    comp = object.__new__(MambaComponent)
    allocator = MagicMock()
    allocator.alloc.return_value = torch.tensor([slot], dtype=torch.int64)
    pool = SimpleNamespace(
        mamba_allocator=allocator,
        mamba_pool=SimpleNamespace(replayssm_write_pos=write_pos),
        translate_mamba_indices=lambda idx: idx,
    )
    comp.cache = SimpleNamespace(req_to_token_pool=pool)
    comp.tree_core = SimpleNamespace(
        component_has_host_value_only=lambda node_id, component_type: True
    )
    return comp


def _req():
    return SimpleNamespace(kv=SimpleNamespace(holds_mamba=False, mamba_pool_idx=None))


class TestMambaLoadBackReplaySSMReset(CustomTestCase):
    def test_load_back_slot_starts_with_empty_ring(self):
        write_pos = torch.tensor([0, 0, 5, 3], dtype=torch.int32)
        comp = _component(write_pos, slot=2)
        req = _req()
        result = comp.prepare_load_back(node_id=1, req=req)
        self.assertEqual(result.allocated_mamba_slot.tolist(), [2])
        self.assertEqual(int(req.kv.mamba_pool_idx), 2)
        self.assertEqual(write_pos.tolist(), [0, 0, 0, 3])

    def test_without_replayssm_is_noop(self):
        comp = _component(None, slot=1)
        req = _req()
        result = comp.prepare_load_back(node_id=1, req=req)
        self.assertEqual(result.allocated_mamba_slot.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
