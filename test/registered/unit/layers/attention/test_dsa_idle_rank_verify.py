"""An attention-DP rank with no requests reaches the DSA backend's eager metadata as a
0-row TARGET_VERIFY (hybrid-SSM models relabel IDLE under SUM_LEN padding); the
page-table width no longer reduces over the empty seq_lens."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa_backend import _eager_max_seqlen_k
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch(seq_lens, with_cpu_mirror=True):
    seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    return SimpleNamespace(
        seq_lens=seq_lens, seq_lens_cpu=seq_lens.clone() if with_cpu_mirror else None
    )


class TestDsaIdleRankVerify(unittest.TestCase):
    def test_idle_rank_verify_has_zero_rows_under_sum_len(self):
        # A decode step with one busy DP rank of 4 picks SUM_LEN, so an idle rank keeps 0
        # tokens and is relabeled TARGET_VERIFY with 0 // num_draft_tokens rows; at DP2 the
        # same step picks MAX_LEN (every rank padded to the busy rank's rows).
        from sglang.srt.layers import dp_attention
        from sglang.srt.layers.dp_attention import DpPaddingMode

        backend = SimpleNamespace(is_pplx=lambda: False, is_deepep_v2=lambda: False)
        draft = 4
        for width, tokens, expected in (
            (4, [draft, 0, 0, 0], DpPaddingMode.SUM_LEN),
            (2, [draft, 0], DpPaddingMode.MAX_LEN),
        ):
            with (
                patch.object(dp_attention, "dp_gather_width", return_value=width),
                patch(
                    "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                    return_value=backend,
                ),
            ):
                mode = DpPaddingMode.get_dp_padding_mode(False, tokens)
            self.assertEqual(mode, expected)

    def test_empty_batch(self):
        self.assertEqual(_eager_max_seqlen_k(_batch([]), 4, 1024), 5)
        self.assertEqual(_eager_max_seqlen_k(_batch([], False), 4, 1024), 5)

    def test_non_empty_batch_unchanged(self):
        self.assertEqual(_eager_max_seqlen_k(_batch([7, 130, 9]), 4, 1024), 134)
        self.assertEqual(_eager_max_seqlen_k(_batch([7, 130, 9], False), 0, 1024), 1024)


if __name__ == "__main__":
    unittest.main()
