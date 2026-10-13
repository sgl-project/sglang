"""PD DSA indexer payloads must skip the prefix decode already holds.

With --disaggregation-decode-enable-radix-cache, decode promises prefill a
page-aligned decode_prefix_len and prefill sends KV only past it. The DSA
indexer state used to be sent over [0, seq_len) on both sides, so the indexer
rows of a cached prefix were rewritten (device hit) or landed on stale page ids
(prefix still being restored from L2/L3). Both sides now send the pages of
[decode_prefix_len, seq_len), which still pair up page by page.

    python -m pytest test/registered/unit/disaggregation/test_dsa_suffix_pages.py -v
"""

import unittest

import torch

from sglang.srt.mem_cache.common import suffix_page_indices
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 4


def _rows(*pages_per_row):
    # One req_to_token row per request: each page's slots, in order.
    return torch.tensor(
        [
            [page * PAGE_SIZE + offset for page in pages for offset in range(PAGE_SIZE)]
            for pages in pages_per_row
        ]
    )


class TestDsaSuffixPages(unittest.TestCase):
    def test_without_a_decode_prefix_every_page_is_sent(self):
        req_to_token = _rows([10, 2, 5, 8], [0, 1, 3, 4])
        pages = suffix_page_indices(req_to_token, 0, 0, 16, PAGE_SIZE)
        self.assertEqual(pages.tolist(), [10, 2, 5, 8])

    def test_a_decode_prefix_skips_its_pages(self):
        req_to_token = _rows([10, 2, 5, 8], [0, 1, 3, 4])
        pages = suffix_page_indices(req_to_token, 0, 8, 16, PAGE_SIZE)
        self.assertEqual(pages.tolist(), [5, 8])

    def test_a_partial_last_page_is_sent(self):
        req_to_token = _rows([10, 2, 5, 8], [0, 1, 3, 4])
        pages = suffix_page_indices(req_to_token, 0, 4, 14, PAGE_SIZE)
        self.assertEqual(pages.tolist(), [2, 5, 8])

    def test_prefill_and_decode_lists_pair_page_by_page(self):
        # Prefill and decode hold the request on different physical pages and
        # pool rows; with the same decode_prefix_len their lists must have the
        # same length so the transfer pairs them positionally.
        prefill = _rows([0, 0, 0, 0], [11, 12, 13, 14])
        decode = _rows([31, 32, 33, 34])
        for prefix_len in (0, 4, 8, 12):
            with self.subTest(prefix_len=prefix_len):
                sent = suffix_page_indices(prefill, 1, prefix_len, 15, PAGE_SIZE)
                received = suffix_page_indices(decode, 0, prefix_len, 15, PAGE_SIZE)
                self.assertEqual(len(sent), len(received))
                first = prefix_len // PAGE_SIZE
                self.assertEqual(sent.tolist(), [11, 12, 13, 14][first:])
                self.assertEqual(received.tolist(), [31, 32, 33, 34][first:])


if __name__ == "__main__":
    unittest.main()
