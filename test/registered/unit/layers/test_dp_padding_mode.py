from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.runtime_context import get_flags


@contextmanager
def dp_context(dp_size: int, hybrid_ssm: bool, max_len_with_idle: bool):
    dp = get_flags().dp
    saved = (dp.hybrid_ssm, dp.max_len_with_idle)
    dp.hybrid_ssm = hybrid_ssm
    dp.max_len_with_idle = max_len_with_idle
    try:
        with patch(
            "sglang.srt.layers.dp_attention.get_attention_dp_size",
            return_value=dp_size,
        ):
            yield
    finally:
        dp.hybrid_ssm, dp.max_len_with_idle = saved


def get_mode(
    global_num_tokens,
    *,
    is_extend_in_batch: bool,
    dp_size: int = 4,
    hybrid_ssm: bool = False,
    max_len_with_idle: bool = False,
) -> DpPaddingMode:
    with dp_context(
        dp_size=dp_size, hybrid_ssm=hybrid_ssm, max_len_with_idle=max_len_with_idle
    ):
        return DpPaddingMode.get_dp_padding_mode(is_extend_in_batch, global_num_tokens)


class TestGetDpPaddingMode(unittest.TestCase):

    def test_uniform_extend_batch_uses_max_len(self):
        """A uniform extend batch pads nothing, so MAX_LEN is safe and preferred."""
        self.assertEqual(
            get_mode([128, 128, 128, 128], is_extend_in_batch=True),
            DpPaddingMode.MAX_LEN,
        )

    def test_skewed_extend_batch_uses_sum_len(self):
        """A skewed extend batch would materialize pad rows, so it stays SUM_LEN."""
        self.assertEqual(
            get_mode([128, 64, 128, 128], is_extend_in_batch=True),
            DpPaddingMode.SUM_LEN,
        )

    def test_extend_batch_with_idle_rank_uses_sum_len(self):
        """An idle rank makes the batch skewed, which must not select MAX_LEN."""
        self.assertEqual(
            get_mode([128, 0, 128, 128], is_extend_in_batch=True),
            DpPaddingMode.SUM_LEN,
        )

    def test_empty_extend_batch_uses_sum_len(self):
        """An all-zero batch has zero communication cost either way; MAX_LEN would
        drive the rank into the fabricated-row conversion with 0 tokens."""
        self.assertEqual(get_mode([0], is_extend_in_batch=True), DpPaddingMode.SUM_LEN)

    def test_skewed_decode_batch_still_uses_the_cost_heuristic(self):
        """The uniformity restriction applies to extend batches only."""
        self.assertEqual(
            get_mode([128, 64, 128, 128], is_extend_in_batch=False),
            DpPaddingMode.MAX_LEN,
        )

    def test_single_rank_extend_batch_uses_the_cost_heuristic(self):
        """With dp_size == 1 max_len equals sum_len, so MAX_LEN stays available."""
        self.assertEqual(
            get_mode([128], is_extend_in_batch=True, dp_size=1),
            DpPaddingMode.MAX_LEN,
        )

    def test_hybrid_ssm_uniform_extend_batch_uses_sum_len(self):
        """Hybrid-SSM extend batches keep the forced mode even when uniform."""
        self.assertEqual(
            get_mode([128, 128, 128, 128], is_extend_in_batch=True, hybrid_ssm=True),
            DpPaddingMode.SUM_LEN,
        )

    def test_hybrid_ssm_with_idle_rank_uses_max_len_when_flagged(self):
        """max_len_with_idle families still materialize idle ranks via MAX_LEN."""
        self.assertEqual(
            get_mode(
                [128, 0, 128, 128],
                is_extend_in_batch=True,
                hybrid_ssm=True,
                max_len_with_idle=True,
            ),
            DpPaddingMode.MAX_LEN,
        )

    def test_hybrid_ssm_without_idle_rank_uses_sum_len_when_flagged(self):
        """max_len_with_idle only applies when some rank is actually idle."""
        self.assertEqual(
            get_mode(
                [128, 64, 128, 128],
                is_extend_in_batch=True,
                hybrid_ssm=True,
                max_len_with_idle=True,
            ),
            DpPaddingMode.SUM_LEN,
        )


if __name__ == "__main__":
    unittest.main()
