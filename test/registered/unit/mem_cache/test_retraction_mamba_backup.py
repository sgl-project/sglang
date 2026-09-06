import unittest

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MAMBA_STATE = object()


class _MambaPool:
    def __init__(self):
        self.loaded = None

    def get_cpu_copy(self, indices):
        return MAMBA_STATE

    def load_cpu_copy(self, state, indices):
        self.loaded = state


class _Allocator:
    def __init__(self, carries_mamba: bool):
        self._kv = type("_KV", (), {"cpu_copy_carries_mamba": carries_mamba})()
        self.loaded_kv = None

    def get_kvcache(self):
        return self._kv

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        return "kv"

    def load_cpu_copy(
        self, cpu_tensors, indices, mamba_indices=None, req_pool_index=None
    ):
        self.loaded_kv = cpu_tensors


def _req_and_pool():
    req = object.__new__(Req)
    req.kv = ReqKvInfo(req_pool_idx=0)
    req.origin_input_ids = [1, 2]
    req.output_ids = [3]
    req.kv.mamba_pool_idx = torch.tensor(1)

    pool = object.__new__(HybridReqToTokenPool)
    pool.req_to_token = torch.zeros(1, 8, dtype=torch.int64)
    pool.mamba_pool = _MambaPool()
    return req, pool


class TestRetractionMambaBackup(unittest.TestCase):
    def test_state_travels_when_kv_pool_leaves_it_behind(self):
        """A sliding-window KV pool accepts mamba_indices and ignores them, so a
        retracted request whose recurrent state is not backed up separately
        resumes on whatever state the reused slot happens to hold."""
        req, pool = _req_and_pool()
        allocator = _Allocator(carries_mamba=False)

        req.offload_kv_cache(pool, allocator)
        self.assertIs(req.kv.retraction_backup.mamba_cpu, MAMBA_STATE)

        req.load_kv_cache(pool, allocator)
        self.assertIs(pool.mamba_pool.loaded, MAMBA_STATE)

    def test_state_is_not_copied_twice_when_the_kv_pool_carries_it(self):
        req, pool = _req_and_pool()
        allocator = _Allocator(carries_mamba=True)

        req.offload_kv_cache(pool, allocator)
        self.assertIsNone(req.kv.retraction_backup.mamba_cpu)

        req.load_kv_cache(pool, allocator)
        self.assertIsNone(pool.mamba_pool.loaded)


if __name__ == "__main__":
    unittest.main()
