"""FlashAttentionBackend helpers for a frozen-KV MTP draft, whose own pool has
no window sub-pool but whose window layers read the target's SWA sub-pool."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _Space:
    def __init__(self, mapping):
        self.mapping = mapping

    def ids(self, x):
        return self.mapping[x]


def _backend(*, runner_space, swa_pool, swa_space, own_pool):
    b = object.__new__(FlashAttentionBackend)
    b.runner_ids = SimpleNamespace(
        space=lambda kind: runner_space,
        ids_in=lambda kind, x: x + 1000,
    )
    b._swa_kv_pool = swa_pool
    b._swa_space = swa_space
    b.token_to_kv_pool = own_pool
    return b


class TestFrozenDraftSwaHelpers(CustomTestCase):
    def test_draft_translates_through_target_space(self):
        mapping = torch.tensor([0, 7, 8, 9])
        b = _backend(
            runner_space=None,
            swa_pool=SimpleNamespace(full_to_swa_index_mapping=mapping),
            swa_space=_Space(mapping),
            own_pool=SimpleNamespace(),
        )
        ids = torch.tensor([1, 3])
        self.assertTrue(torch.equal(b._swa_ids(ids), torch.tensor([7, 9])))

    def test_runner_with_window_space_keeps_its_own_path(self):
        b = _backend(
            runner_space=object(),
            swa_pool=None,
            swa_space=None,
            own_pool=SimpleNamespace(),
        )
        self.assertTrue(
            torch.equal(b._swa_ids(torch.tensor([1])), torch.tensor([1001]))
        )

    def test_mapping_comes_from_resolved_pool_not_swapped_pool(self):
        mapping = torch.arange(4)
        b = _backend(
            runner_space=None,
            swa_pool=SimpleNamespace(full_to_swa_index_mapping=mapping),
            swa_space=None,
            own_pool=SimpleNamespace(full_to_swa_index_mapping=None),
        )
        self.assertIs(b._swa_full_to_swa_index_mapping(), mapping)

    def test_plain_runner_reads_its_own_pool_mapping(self):
        mapping = torch.arange(4)
        b = _backend(
            runner_space=object(),
            swa_pool=None,
            swa_space=None,
            own_pool=SimpleNamespace(full_to_swa_index_mapping=mapping),
        )
        self.assertIs(b._swa_full_to_swa_index_mapping(), mapping)

    def test_no_window_write_space_gives_no_write_locations(self):
        b = _backend(
            runner_space=None,
            swa_pool=object(),
            swa_space=None,
            own_pool=SimpleNamespace(),
        )
        self.assertIsNone(b._get_swa_write_locations(SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
