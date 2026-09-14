"""Retraction CPU backup/restore on a DCP-sharded MLA pool.

`req_to_token` names KV slots in the widened DCP id space while `kv_buffer`
holds only this rank's rows; a widened id used as a row index reads past the
buffer or copies another token's row.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZE = 4
DCP_RANK = 1
ROWS = 8
LAYERS = 2


def _bare_mla_pool() -> MLATokenToKVPool:
    pool = object.__new__(MLATokenToKVPool)
    pool.layer_num = LAYERS
    pool.cpu_offloading_chunk_size = 3
    pool.kv_buffer = [
        (torch.arange(ROWS, dtype=torch.float32) + 100 * layer).view(ROWS, 1, 1)
        for layer in range(LAYERS)
    ]
    return pool


def _owned_rows(widened: torch.Tensor) -> torch.Tensor:
    return widened[widened % DCP_SIZE == DCP_RANK] // DCP_SIZE


def _dcp():
    return get_parallel().override(
        dcp_enabled=True, attn_dcp_size=DCP_SIZE, attn_dcp_rank=DCP_RANK
    )


class TestRetractionDcpBackup(CustomTestCase):
    def test_backup_gathers_only_this_ranks_rows(self):
        pool = _bare_mla_pool()
        widened = torch.arange(20, dtype=torch.int64)

        with _dcp():
            kv_cpu = pool.get_cpu_copy(widened)

        expected_rows = _owned_rows(widened)
        self.assertEqual(expected_rows.tolist(), [0, 1, 2, 3, 4])
        for layer in range(LAYERS):
            gathered = torch.cat(kv_cpu[layer])
            torch.testing.assert_close(gathered, pool.kv_buffer[layer][expected_rows])

    def test_restore_lands_on_new_owned_rows(self):
        pool = _bare_mla_pool()
        before = [buf.clone() for buf in pool.kv_buffer]
        old_widened = torch.arange(0, 12, dtype=torch.int64)
        new_widened = torch.arange(12, 24, dtype=torch.int64)

        with _dcp():
            kv_cpu = pool.get_cpu_copy(old_widened)
            pool.load_cpu_copy(kv_cpu, new_widened)

        old_rows = _owned_rows(old_widened)
        new_rows = _owned_rows(new_widened)
        self.assertEqual(new_rows.tolist(), [3, 4, 5])
        untouched = torch.tensor([r for r in range(ROWS) if r not in new_rows.tolist()])
        for layer in range(LAYERS):
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

        for layer in range(LAYERS):
            torch.testing.assert_close(torch.cat(kv_cpu[layer]), pool.kv_buffer[layer])


if __name__ == "__main__":
    unittest.main()
