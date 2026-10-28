# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.clover_utils import (
    clover_accept_tokens,
    clover_correct_drafts,
)
from sglang.srt.speculative.clover_worker import CloverWorker
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCloverAcceptGraph(CustomTestCase):
    def test_replay_with_new_candidates_and_logits(self):
        for rule in ("threshold", "exact"):
            with self.subTest(rule=rule):
                self._check_replay(rule)

    def _check_replay(self, rule):
        worker = CloverWorker.__new__(CloverWorker)
        worker.accept_rule = rule
        worker.accept_ratio, worker.accept_delta = 0.6, 0.3
        worker._accept_graph = worker._accept_graph_key = None
        worker._sync = lambda x: x
        logits = torch.empty(1, 16, 128, device="cuda")
        with patch(
            "sglang.srt.speculative.clover_worker.get_parallel",
            return_value=SimpleNamespace(tp_size=1),
        ):
            for iteration in range(6):
                if iteration == 3:
                    logits = torch.empty_like(logits)  # force graph recapture
                logits.normal_()
                candidates = torch.randint(128, (1, 16), device="cuda")
                if iteration % 2 == 0:
                    candidates[:, 1:] = logits[:, :-1].argmax(-1)
                count, boundary = worker._accept_boundary(logits, candidates, True)
                expected = clover_correct_drafts(
                    logits, candidates, 0.6, 0.3, accept_rule=rule
                )
                torch.testing.assert_close(count, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    boundary,
                    logits[torch.arange(1, device="cuda"), expected.long()],
                    rtol=0,
                    atol=0,
                )
                count, bonus, lens, tokens = worker._accept_boundary(
                    logits, candidates, True, greedy=True
                )
                expected_bonus = logits[
                    torch.arange(1, device="cuda"), expected.long()
                ].argmax(-1)
                torch.testing.assert_close(count, expected, rtol=0, atol=0)
                torch.testing.assert_close(bonus, expected_bonus, rtol=0, atol=0)
                torch.testing.assert_close(lens, expected + 1, rtol=0, atol=0)
                torch.testing.assert_close(
                    tokens,
                    clover_accept_tokens(candidates, expected, expected_bonus),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
