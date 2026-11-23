"""Tail DP metadata must remain optional when no layer gathers tokens."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.models.deepseek_common import decoder_swa_bounded_replay as tail
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def batch(gather, valid):
    scalar = torch.tensor(valid, dtype=torch.int32)
    return SimpleNamespace(
        global_num_tokens_cpu=[8, 8] if gather else None,
        global_num_tokens_gpu=torch.tensor([8, 8]) if gather else None,
        global_num_tokens_padded_cpu=[8, 8] if gather else None,
        global_num_tokens_for_logprob_cpu=None,
        global_num_tokens_for_logprob_gpu=None,
        global_dp_buffer_len=16 if gather else None,
        dp_padding_mode=DpPaddingMode.MAX_LEN,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
        global_num_token_non_padded=scalar,
        global_num_token_non_padded_cpu=valid,
        num_token_non_padded=scalar,
    )


class TestTailDpCounts(unittest.TestCase):
    def test_no_gather_preserves_absence_but_bounds_moe_rows(self):
        for counts, valid in (([4, 4], 8), ([2, 4], 8), ([4, 4], 0)):
            with self.subTest(counts=counts, valid=valid):
                fb = batch(False, valid)
                before = vars(fb).copy()
                with (
                    get_parallel().override(attn_dp_rank=1, attn_dp_size=2),
                    patch.object(tail, "set_dp_buffer_len_from_batch") as publish,
                ):
                    saved = tail.enter_late_layer_dp(fb, counts)
                    for name in before:
                        if name not in (
                            "global_num_token_non_padded",
                            "global_num_token_non_padded_cpu",
                            "num_token_non_padded",
                        ):
                            self.assertIs(getattr(fb, name), before[name])
                    self.assertEqual(fb.num_token_non_padded.item(), min(valid, 4))
                    self.assertEqual(fb.global_num_token_non_padded_cpu, min(valid, 4))
                    tail.exit_late_layer_dp(fb, saved)
                    publish.assert_not_called()
                for name, value in before.items():
                    self.assertIs(getattr(fb, name), value)

    def test_gather_counts_are_replaced_and_restored(self):
        for counts in ([4, 4], [2, 4]):
            with self.subTest(counts=counts):
                fb = batch(True, 8)
                before = vars(fb).copy()
                with (
                    get_parallel().override(attn_dp_rank=1, attn_dp_size=2),
                    patch.object(tail, "set_dp_buffer_len_from_batch") as publish,
                    patch.object(
                        tail,
                        "async_h2d",
                        side_effect=lambda x, **kw: torch.tensor(x, **kw),
                    ),
                ):
                    saved = tail.enter_late_layer_dp(fb, counts)
                    self.assertEqual(fb.global_num_tokens_gpu.tolist(), counts)
                    self.assertEqual(fb.global_num_tokens_cpu, counts)
                    self.assertEqual(fb.global_dp_buffer_len, sum(counts))
                    self.assertEqual(fb.dp_padding_mode, DpPaddingMode.SUM_LEN)
                    self.assertEqual(fb.num_token_non_padded.item(), 4)
                    self.assertEqual(publish.call_count, 1)
                    tail.exit_late_layer_dp(fb, saved)
                    self.assertEqual(publish.call_count, 2)
                for name, value in before.items():
                    self.assertIs(getattr(fb, name), value)


if __name__ == "__main__":
    unittest.main()
