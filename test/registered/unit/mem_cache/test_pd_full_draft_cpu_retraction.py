import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    RetractionBackup,
    retraction_backup,
    retraction_discard,
    retraction_restore,
)
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    MambaPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def make_pools(dcp_size):
    # Use real copy/restore methods with small CPU tensors. Physical target
    # rows are sharded; full draft rows retain the logical token-id domain.
    target = object.__new__(MLATokenToKVPool)
    target.layer_num = 2
    target.cpu_offloading_chunk_size = 7
    target.kv_buffer = [
        (torch.arange(128, dtype=torch.float32) + 1000 * layer).view(128, 1, 1)
        for layer in range(2)
    ]
    draft = object.__new__(MHATokenToKVPool)
    draft.use_hnd = False
    draft.layer_num = 3
    draft.cpu_offloading_chunk_size = 11
    draft.k_buffer = [
        (torch.arange(128 * dcp_size, dtype=torch.float32) + 3000 * layer).view(
            -1, 1, 1
        )
        for layer in range(3)
    ]
    draft.v_buffer = [buffer + 1500 for buffer in draft.k_buffer]
    mamba = object.__new__(MambaPool)
    mamba._slot_siblings = []
    mamba.mamba_cache = SimpleNamespace(
        conv=[torch.arange(24, dtype=torch.float32).view(2, 4, 3)],
        temporal=torch.arange(40, dtype=torch.float32).view(2, 4, 5),
    )
    hybrid = object.__new__(HybridLinearKVPool)
    hybrid.full_kv_pool = target
    hybrid.mamba_pool = mamba
    hybrid._mamba_translate = lambda indices: indices
    return target, draft, mamba, hybrid


class TestFullDraftCpuRetraction(CustomTestCase):
    def test_full_draft_registers_but_compact_mapping_does_not(self):
        for compact in (False, True):
            with self.subTest(compact=compact):
                _, draft, _, hybrid = make_pools(1)
                allocator = TokenToKVPoolAllocator(
                    128, torch.float32, "cpu", hybrid, need_sort=False
                )
                worker = object.__new__(DFlashWorkerV2)
                worker.use_compact_draft_cache = compact
                worker._draft_worker = SimpleNamespace(alloc_memory_pool=Mock())
                worker.draft_model_runner = SimpleNamespace(token_to_kv_pool=draft)
                req_pool = object()
                worker.alloc_memory_pool(
                    req_to_token_pool=req_pool,
                    token_to_kv_pool_allocator=allocator,
                )
                passed_req_pool = (
                    worker._draft_worker.alloc_memory_pool.call_args.kwargs[
                        "req_to_token_pool"
                    ]
                )
                self.assertIs(passed_req_pool, None if compact else req_pool)
                self.assertIs(
                    allocator.cpu_retraction_draft_pool,
                    None if compact else draft,
                )

    def test_poison_and_relocate_preserves_target_mamba_and_full_draft(self):
        for dcp_size in (1, 2, 8):
            for rank in range(dcp_size):
                for length in (0, 1, 7, 17, 63):
                    for with_draft in (False, True):
                        with self.subTest(
                            dcp_size=dcp_size,
                            rank=rank,
                            length=length,
                            with_draft=with_draft,
                        ):
                            self._round_trip(dcp_size, rank, length, with_draft)

    def _round_trip(self, dcp_size, rank, length, with_draft):
        target, draft, mamba, hybrid = make_pools(dcp_size)
        allocator = TokenToKVPoolAllocator(
            128 * dcp_size, torch.float32, "cpu", hybrid, need_sort=False
        )
        if with_draft:
            allocator.cpu_retraction_draft_pool = draft
        req = Req("test", "", [1] * (length + 1), SamplingParams(max_new_tokens=16))
        req.kv.req_pool_idx = 0
        req.kv.mamba_pool_idx = torch.tensor([1])
        # Two fragments; destination relocation preserves each logical offset.
        old = torch.cat((torch.arange(32), torch.arange(64, 96)))[:length]
        new = old + 32 * dcp_size
        req_pool = SimpleNamespace(req_to_token=old.unsqueeze(0).clone())
        target_expected = [buf.clone() for buf in target.kv_buffer]
        draft_expected = [
            buf[old].clone() for buf in (*draft.k_buffer, *draft.v_buffer)
        ]
        mamba_expected = [
            buf[:, 1].clone()
            for buf in (*mamba.mamba_cache.conv, mamba.mamba_cache.temporal)
        ]
        with get_parallel().override(
            dcp_enabled=dcp_size > 1, attn_dcp_size=dcp_size, attn_dcp_rank=rank
        ):
            self.assertTrue(
                retraction_backup(req, None, req_pool, allocator, "cpu_tensor")
            )
            for buf in (*target.kv_buffer, *draft.k_buffer, *draft.v_buffer):
                buf.fill_(-100)
            for buf in (*mamba.mamba_cache.conv, mamba.mamba_cache.temporal):
                buf.fill_(-200)
            req_pool.req_to_token = new.unsqueeze(0)
            req.kv.mamba_pool_idx = torch.tensor([2])
            retraction_restore(req, None, req_pool, allocator, "cpu_tensor")
        self.assertIsNone(req.kv.retraction_backup)
        mask = old.remainder(dcp_size) == rank
        source_rows = old[mask] // dcp_size
        destination_rows = new[mask] // dcp_size
        for actual, expected in zip(target.kv_buffer, target_expected, strict=True):
            torch.testing.assert_close(actual[destination_rows], expected[source_rows])
            untouched = torch.ones(actual.shape[0], dtype=torch.bool)
            untouched[destination_rows] = False
            self.assertTrue(torch.all(actual[untouched] == -100))
        for actual, expected in zip(
            (*draft.k_buffer, *draft.v_buffer), draft_expected, strict=True
        ):
            if with_draft:
                torch.testing.assert_close(actual[new], expected)
            else:
                self.assertTrue(torch.all(actual == -100))
        for actual, expected in zip(
            (*mamba.mamba_cache.conv, mamba.mamba_cache.temporal),
            mamba_expected,
            strict=True,
        ):
            torch.testing.assert_close(actual[:, 2], expected)
            self.assertTrue(torch.all(actual[:, 1] == -200))

    def test_host_pool_backend_keeps_its_own_transfer_path(self):
        req = SimpleNamespace(
            seqlen=3,
            kv=SimpleNamespace(retraction_backup=None),
            offload_kv_cache=Mock(side_effect=AssertionError("CPU path called")),
            load_kv_cache=Mock(side_effect=AssertionError("CPU path called")),
        )
        backup = RetractionBackup(host_indices=torch.tensor([1, 2]))
        cache = SimpleNamespace(
            retraction_backup=Mock(return_value=backup), retraction_restore=Mock()
        )
        self.assertTrue(retraction_backup(req, cache, None, None, "host_pool"))
        retraction_restore(req, cache, None, None, "host_pool")
        cache.retraction_restore.assert_called_once_with(req, backup)
        self.assertIsNone(req.kv.retraction_backup)

    def test_discard_drops_cpu_draft_snapshot(self):
        req = SimpleNamespace(
            kv=SimpleNamespace(
                retraction_backup=RetractionBackup(draft_cpu=torch.ones(3))
            )
        )
        retraction_discard(req, None, "cpu_tensor")
        self.assertIsNone(req.kv.retraction_backup)


if __name__ == "__main__":
    unittest.main()
