"""Single-rank token synchronization must not launch a collective."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers import sampler

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSamplerTokenSync(CustomTestCase):
    def test_sync_conditions_preserve_tokens_and_multi_rank_behavior(self):
        for size in (1, 2):
            for forced in (False, True):
                for grammar in (False, True):
                    with self.subTest(size=size, forced=forced, grammar=grammar):
                        group = object()
                        instance = SimpleNamespace(
                            tp_sync_world_size=size, tp_sync_group=group
                        )
                        tokens = torch.tensor([7, 42], dtype=torch.int64)
                        info = SimpleNamespace(grammars=[object()] if grammar else None)
                        with (
                            patch.object(sampler, "SYNC_TOKEN_IDS_ACROSS_TP", forced),
                            patch.object(torch.distributed, "all_reduce") as reduce,
                        ):
                            sampler.Sampler._sync_token_ids_across_tp(
                                instance, tokens, info
                            )
                        if size > 1 and (forced or grammar):
                            reduce.assert_called_once_with(
                                tokens, op=torch.distributed.ReduceOp.MIN, group=group
                            )
                        else:
                            reduce.assert_not_called()
                        torch.testing.assert_close(tokens, torch.tensor([7, 42]))


if __name__ == "__main__":
    unittest.main()
