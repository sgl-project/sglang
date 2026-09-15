import unittest

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

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


DCP_SIZE = 4
DCP_RANK = 1
ROWS = 8


def _bare_mla_pool() -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.layer_num = 2
    pool.cpu_offloading_chunk_size = 3
    pool.kv_buffer = [
        (torch.arange(ROWS, dtype=torch.float32) + 100 * layer).view(ROWS, 1, 1)
        for layer in range(pool.layer_num)
    ]
    return pool


def _dcp():
    return get_parallel().override(
        dcp_enabled=True, attn_dcp_size=DCP_SIZE, attn_dcp_rank=DCP_RANK
    )


class TestRetractionDcpBackup(unittest.TestCase):
    """`req_to_token` names KV slots in the widened DCP id space while
    `kv_buffer` holds only this rank's rows; a widened id used as a row index
    reads past the buffer or copies another token's row."""

    def test_restore_lands_on_new_owned_rows(self):
        pool = _bare_mla_pool()
        before = [buf.clone() for buf in pool.kv_buffer]
        old_widened = torch.arange(0, 12, dtype=torch.int64)
        new_widened = torch.arange(12, 24, dtype=torch.int64)

        with _dcp():
            pool.load_cpu_copy(pool.get_cpu_copy(old_widened), new_widened)

        old_rows = old_widened[DCP_RANK::DCP_SIZE] // DCP_SIZE
        new_rows = new_widened[DCP_RANK::DCP_SIZE] // DCP_SIZE
        self.assertEqual(new_rows.tolist(), [3, 4, 5])
        untouched = torch.tensor([r for r in range(ROWS) if r not in new_rows.tolist()])
        for layer in range(pool.layer_num):
            torch.testing.assert_close(
                pool.kv_buffer[layer][new_rows], before[layer][old_rows]
            )
            torch.testing.assert_close(
                pool.kv_buffer[layer][untouched], before[layer][untouched]
            )

    def test_resolved_pool_takes_ids_as_rows(self):
        pool = _bare_mla_pool()
        pool.write_loc_is_dcp_resolved = True
        rows = torch.arange(ROWS, dtype=torch.int64)

        with _dcp():
            kv_cpu = pool.get_cpu_copy(rows)

        for layer in range(pool.layer_num):
            torch.testing.assert_close(torch.cat(kv_cpu[layer]), pool.kv_buffer[layer])


if __name__ == "__main__":
    unittest.main()
