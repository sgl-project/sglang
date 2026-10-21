"""PDMux prefill uses NCCL gather without changing decode or other layouts."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import dp_attention as dp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDMuxDpGatherv(unittest.TestCase):
    def test_policy_and_decode_layout_guards(self):
        parallel = SimpleNamespace(
            attn_dp_enabled=True, attn_tp_size=1, tp_size=4, attn_dp_size=4
        )
        with (
            patch.object(dp, "get_parallel", return_value=parallel),
            patch.object(dp, "get_flags"),
            patch.object(dp, "get_exec") as execution,
            patch.object(dp, "world_dp_gather_enabled", return_value=False),
            patch.object(dp, "_USE_DP_GATHERV", False),
            patch.object(dp._DpGatheredBufferWrapper, "is_dp_max_padding") as max_len,
            patch.dict(os.environ),
        ):
            execution.return_value.moe.elastic_ep_backend = None
            for pdmux, override, expected in (
                (False, None, False),
                (True, None, True),
                (True, "0", False),
                (False, "1", True),
            ):
                with self.subTest(pdmux=pdmux, override=override):
                    os.environ.pop("SGLANG_DP_USE_GATHERV", None)
                    if override is not None:
                        os.environ["SGLANG_DP_USE_GATHERV"] = override
                    dp.initialize_dp_attention_flags(
                        SimpleNamespace(enable_pdmux=pdmux)
                    )
                    max_len.return_value = False
                    self.assertEqual(dp.is_dp_gatherv_active(), expected)
                    max_len.return_value = True
                    self.assertFalse(dp.is_dp_gatherv_active())
            max_len.return_value = False
            parallel.attn_tp_size = 2
            self.assertFalse(dp.is_dp_gatherv_active())
            parallel.attn_tp_size = 1
            parallel.attn_dp_size = 2
            self.assertFalse(dp.is_dp_gatherv_active())

    def test_gather_preserves_rows_and_zeros_idle_rank_padding(self):
        sizes = [0, 4, 0, 4]
        source = [torch.arange(n * 8).reshape(n, 8).float() for n in (0, 3, 0, 4)]
        expected = torch.cat(
            [
                torch.cat([x, torch.zeros(rows - len(x), 8)])
                for rows, x in zip(sizes, source)
            ]
        )
        for rank in range(4):
            output = torch.full((8, 8), float("nan"))

            def all_gatherv(local, *, sizes, output):
                start = sum(sizes[:rank])
                torch.testing.assert_close(local, expected[start : start + sizes[rank]])
                output.copy_(expected)

            with (
                patch.object(dp, "dp_slot_in", return_value=rank),
                patch.object(dp, "_use_dp_gather_fp8", return_value=False),
                patch.object(
                    dp,
                    "get_parallel",
                    return_value=SimpleNamespace(
                        tp_group=SimpleNamespace(all_gatherv=all_gatherv)
                    ),
                ),
            ):
                dp._dp_gather_via_all_gatherv(output, source[rank], None, False, sizes)
            torch.testing.assert_close(output, expected)


if __name__ == "__main__":
    unittest.main()
