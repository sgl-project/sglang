"""MLA DFLASH draft window resolution, checked against a brute-force coverage oracle."""

import unittest

from sglang.srt.speculative.dflash_utils import resolve_mla_draft_compact_window
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

BLOCK = 8


def _mla_draft(layer_types, sliding_window=None):
    return {
        "num_hidden_layers": len(layer_types),
        "dflash_config": {"attention_mode": "mla"},
        "layer_types": list(layer_types),
        "sliding_window": sliding_window,
    }


def _compact_rows(prefix_len: int, window: int, page_size: int) -> range:
    # Mirrors DFlashWorkerV2._compute_compact_draft_seq_lens.
    visible = min(prefix_len, window)
    start = prefix_len - visible
    start -= start % page_size
    return range(start, prefix_len + BLOCK)


def _compact_changes_nothing(hf_window: int, window: int, page_size: int) -> bool:
    window_left = hf_window - 1
    for prefix_len in range(0, 3 * max(window, hf_window) + 2 * page_size):
        rows = _compact_rows(prefix_len, window, page_size)
        for p in range(prefix_len, prefix_len + BLOCK):
            if max(p - window_left, 0) < rows.start:
                return False
    return True


class TestResolveMlaDraftCompactWindow(CustomTestCase):
    def test_all_sliding_drops_exactly_when_the_compact_window_covers(self):
        window = 64
        for hf_window in (16, 63, 64, 65, 66, 80):
            for page_size in (1, 16):
                cfg = _mla_draft(["sliding_attention"] * 5, hf_window)
                with self.subTest(hf_window=hf_window, page_size=page_size):
                    if _compact_changes_nothing(hf_window, window, page_size):
                        self.assertIsNone(
                            resolve_mla_draft_compact_window(
                                cfg,
                                window_size=window,
                                dcp_size=16,
                                page_size=page_size,
                            )
                        )
                    else:
                        with self.assertRaises(ValueError):
                            resolve_mla_draft_compact_window(
                                cfg,
                                window_size=window,
                                dcp_size=16,
                                page_size=page_size,
                            )


if __name__ == "__main__":
    unittest.main()
