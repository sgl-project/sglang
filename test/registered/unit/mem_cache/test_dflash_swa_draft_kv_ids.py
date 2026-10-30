"""A compact-window DFLASH draft pool on the target's sliding-window ids
(SGLANG_ENABLE_DFLASH_SWA_DRAFT_KV): the draft runner's write ids are the
target's SWA translation, under the target's own SWA space key, and the
target allocators without a paged full->SWA map are refused.

CPU-only.
"""

from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.test.test_utils import CustomTestCase


def _target(full_to_swa, **overrides):
    pool = SimpleNamespace()
    swa_space = IdSpace(
        key=(IdSpaceKind.SLIDING_WINDOW, id(pool)),
        write=lambda ids: full_to_swa[ids.to(torch.int64)],
    )
    pool.id_spaces = lambda: {
        IdSpaceKind.FULL: IdSpace(key=(IdSpaceKind.FULL, id(pool))),
        IdSpaceKind.SLIDING_WINDOW: swa_space,
    }
    fields = dict(
        has_window_pool=True,
        hands_out_virtual_ids=False,
        swa_req_ring=False,
        get_kvcache=lambda: pool,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields), swa_space


def _runner(allocator, pool):
    return RunnerIds(
        req_to_token=torch.zeros((4, 16), dtype=torch.int32),
        token_to_kv_pool_allocator=allocator,
        token_to_kv_pool=pool,
        page_size=1,
        device="cpu",
        is_draft_worker=True,
    )


class TestDFlashSWADraftKVIds(CustomTestCase):
    def setUp(self):
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="tokenizer")

    def test_draft_writes_target_swa_ids(self):
        full_to_swa = torch.arange(64, dtype=torch.int64) * 3 + 1
        allocator, swa_space = _target(full_to_swa)
        draft = _runner(allocator, SimpleNamespace(follows_target_swa_ids=True))
        space = draft.space(IdSpaceKind.FULL)
        self.assertEqual(space.key, swa_space.key)
        self.assertIsNone(draft.space(IdSpaceKind.SLIDING_WINDOW))
        loc = torch.tensor([5, 9, 30])
        torch.testing.assert_close(space.write(loc), full_to_swa[loc])

    def test_plain_draft_keeps_its_own_ids(self):
        allocator, _ = _target(torch.arange(8))
        own = {IdSpaceKind.FULL: IdSpace(key=(IdSpaceKind.FULL, None))}
        draft = _runner(allocator, SimpleNamespace(id_spaces=lambda: own))
        self.assertIs(draft.space(IdSpaceKind.FULL), own[IdSpaceKind.FULL])

    def test_refuses_targets_without_paged_map(self):
        draft_pool = SimpleNamespace(follows_target_swa_ids=True)
        for overrides in (
            {"has_window_pool": False},
            {"swa_req_ring": True},
        ):
            allocator, _ = _target(torch.arange(8), **overrides)
            with self.subTest(**overrides), self.assertRaises(ValueError):
                _runner(allocator, draft_pool)


if __name__ == "__main__":
    unittest.main()
