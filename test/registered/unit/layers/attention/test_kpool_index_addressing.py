"""K-pool index addressing: pooled key j of a request lives at index slot
loc(4j) // 4, so index page ids are the request's logical page ids."""

import unittest

import torch

from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
    build_pooled_page_table_64,
    compute_pooled_write_locs,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

POOL_SIZE = 4
PHYSICAL_PAGE_SIZE = 64
LOGICAL_PAGE_SIZE = PHYSICAL_PAGE_SIZE * POOL_SIZE


def _req_to_token(logical_pages: list[int]) -> torch.Tensor:
    offsets = torch.arange(LOGICAL_PAGE_SIZE, dtype=torch.int64)
    return torch.cat([page * LOGICAL_PAGE_SIZE + offsets for page in logical_pages])


def _physical_page_table(req_to_token: torch.Tensor) -> torch.Tensor:
    return (req_to_token[::PHYSICAL_PAGE_SIZE] // PHYSICAL_PAGE_SIZE).to(torch.int32)


def _write_locs(req_to_token: torch.Tensor, pool_ids: torch.Tensor) -> torch.Tensor:
    return compute_pooled_write_locs(
        _physical_page_table(req_to_token), pool_ids, POOL_SIZE
    )


class TestKpoolIndexAddressing(CustomTestCase):
    def test_write_slot_is_group_head_loc_over_pool_size(self):
        req_to_token = _req_to_token([7, 2, 11])
        pool_ids = torch.arange(req_to_token.numel() // POOL_SIZE)
        torch.testing.assert_close(
            _write_locs(req_to_token, pool_ids),
            req_to_token[::POOL_SIZE] // POOL_SIZE,
        )

    def test_index_page_table_is_logical_page_table(self):
        req_to_token = _req_to_token([7, 2, 11])
        index_pages = build_pooled_page_table_64(
            _physical_page_table(req_to_token), POOL_SIZE
        )
        self.assertEqual(index_pages.tolist(), [7, 2, 11])

    def test_suffix_writes_stay_out_of_a_shared_prefix_page(self):
        # B reuses A's first logical page and writes only its own suffix keys.
        prefix_page, a_page, b_page = 5, 9, 3
        a = _req_to_token([prefix_page, a_page])
        b = _req_to_token([prefix_page, b_page])
        keys_per_page = LOGICAL_PAGE_SIZE // POOL_SIZE

        b_suffix = torch.arange(keys_per_page, 2 * keys_per_page)
        b_suffix_pages = _write_locs(b, b_suffix) // keys_per_page
        self.assertEqual(set(b_suffix_pages.tolist()), {b_page})

        prefix = torch.arange(keys_per_page)
        torch.testing.assert_close(_write_locs(a, prefix), _write_locs(b, prefix))


if __name__ == "__main__":
    unittest.main()
