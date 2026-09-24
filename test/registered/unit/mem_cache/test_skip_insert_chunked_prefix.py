"""A chunked request that skips radix insertion must still advance its prefix.

Fake-bootstrap (warmup) requests set skip_radix_cache_insert. Before the fix,
maybe_cache_unfinished_req returned without touching prefix_indices, so a
request longer than one prefill chunk re-extended its first chunk forever and
leaked that chunk's KV (K3 prefill_shapes warmups at 24K/32K with 16K chunks).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _req(skip: bool, filled: int, holds_kv: bool = True):
    return SimpleNamespace(
        skip_radix_cache_insert=skip,
        kv=SimpleNamespace(holds_kv=holds_kv, req_pool_idx=1),
        prefix_indices=torch.empty(0, dtype=torch.int64),
        get_fill_ids=lambda: list(range(filled)),
    )


class TestSkipInsertChunkedPrefix(CustomTestCase):
    def setUp(self):
        self.tree = Mock()
        self.tree.req_to_token_pool.req_to_token = torch.arange(2 * 64).reshape(2, 64)

    def test_skipped_insert_keeps_own_kv_as_prefix(self):
        req = _req(skip=True, filled=48)
        maybe_cache_unfinished_req(req, self.tree, chunked=True)
        self.tree.cache_unfinished_req.assert_not_called()
        torch.testing.assert_close(req.prefix_indices, torch.arange(64, 112))
        self.assertEqual(req.prefix_indices.dtype, torch.int64)

    def test_skipped_insert_without_kv_is_a_no_op(self):
        req = _req(skip=True, filled=48, holds_kv=False)
        maybe_cache_unfinished_req(req, self.tree, chunked=True)
        self.assertEqual(len(req.prefix_indices), 0)

    def test_normal_request_goes_through_the_tree(self):
        req = _req(skip=False, filled=48)
        maybe_cache_unfinished_req(req, self.tree, chunked=True)
        self.tree.cache_unfinished_req.assert_called_once_with(req, chunked=True)


if __name__ == "__main__":
    unittest.main()
