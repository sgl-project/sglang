"""CPU checks for the FDFO + LowConfidence overlap draft.

The algorithm result and the FutureMap block relay must agree whether or not
the scheduler overlaps the next batch with result processing.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.mixin.scheduler import DllmManager
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _low_confidence():
    return LowConfidence(
        SimpleNamespace(
            block_size=4,
            mask_id=0,
            first_done_first_out_mode=True,
            algorithm_config={"threshold": 0.5},
        )
    )


def _logits_for(input_ids: torch.Tensor, vocab: int) -> torch.Tensor:
    logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab), -1e4)
    # Masked positions (token 0) predict token 1 with probability ~1.
    logits[:, :, 1] = torch.where(input_ids == 0, 10.0, logits[:, :, 1])
    return logits.reshape(-1, vocab)


class _Runner:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def forward(self, forward_batch, pp_proxy_tensors=None):
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=self.logits),
            can_run_graph=False,
        )


class TestFdfoLowConfidenceOverlap(unittest.TestCase):
    def test_run_fdfo_matches_step_and_stays_on_device(self):
        algo = _low_confidence()
        input_ids = torch.tensor([[0, 0, 3, 4], [1, 2, 3, 4]], dtype=torch.int64)
        logits = _logits_for(input_ids, vocab=6)

        direct = SimpleNamespace(
            batch_size=2,
            input_ids=input_ids.clone().reshape(-1),
        )
        algo.step(direct, logits, [None, None])

        batched = SimpleNamespace(
            batch_size=2,
            input_ids=input_ids.clone().reshape(-1),
        )
        result = algo._run_fdfo(_Runner(logits), batched, None)

        self.assertIsInstance(result.next_token_ids, torch.Tensor)
        self.assertEqual(result.next_token_ids.shape, (2, 4))
        self.assertIsInstance(result.dllm_done, torch.Tensor)
        self.assertEqual(result.dllm_done.dtype, torch.bool)
        self.assertEqual(result.next_token_ids.tolist(), direct.input_ids.view(2, 4).tolist())
        self.assertEqual(result.dllm_done.tolist(), [False, True])

    def test_step_falls_back_to_highest_confidence_position(self):
        algo = _low_confidence()
        input_ids = torch.tensor([[0, 0, 3, 4], [1, 2, 3, 4]], dtype=torch.int64)
        logits = torch.zeros((2, 4, 6))
        # Both masked positions stay under the 0.5 threshold. Position 1 is higher.
        logits[0, 0, 1] = 0.2
        logits[0, 1, 2] = 0.4
        batch = SimpleNamespace(batch_size=2, input_ids=input_ids.clone().reshape(-1))

        done = algo.step(batch, logits.reshape(-1, 6), [None, None])

        self.assertEqual(done.tolist(), [False, True])
        self.assertEqual(batch.input_ids.view(2, 4).tolist(), [[0, 2, 3, 4], [1, 2, 3, 4]])

    def test_future_map_overwrites_only_rows_with_a_block(self):
        future_map = FutureMap(
            device=torch.device("cpu"),
            spec_algo=SimpleNamespace(),
            req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros((4, 8))),
            needs_cpu_seq_lens=False,
        )
        future_map.stash_dllm_block_tokens(
            torch.tensor([2]),
            torch.tensor([[8, 8, 8, 8]]),
        )
        batch = SimpleNamespace(
            is_dllm=lambda: True,
            req_pool_indices=torch.tensor([1, 2]),
            input_ids=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64),
        )
        future_map.resolve_dllm_block_tokens(batch)
        self.assertEqual(
            batch.input_ids.tolist(),
            [0, 0, 0, 0, 8, 8, 8, 8],
        )

        future_map.dllm_block_tokens_buf[2] = -1
        batch.input_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
        future_map.resolve_dllm_block_tokens(batch)
        self.assertEqual(batch.input_ids.tolist(), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_init_next_round_keeps_geometry_until_block_done(self):
        manager = DllmManager(
            SimpleNamespace(max_running_requests=2, first_done_first_out_mode=True)
        )

        class _Req:
            def __init__(self, done):
                self.dllm_block_done = done
                self.dllm_incomplete_ids = array("q", [1, 2, 3, 4])
                self.inited = 0

            def init_next_round_input(self):
                self.inited += 1

        open_req = _Req(False)
        done_req = _Req(True)
        manager.staging_queue = [open_req, done_req]
        manager.init_next_round()

        self.assertEqual(open_req.inited, 0)
        self.assertFalse(open_req.dllm_block_done)
        self.assertEqual(done_req.inited, 1)
        # Marker stays set so an in-flight extra step can skip a second emit.
        self.assertTrue(done_req.dllm_block_done)
        self.assertEqual(manager.staging_queue, [])


if __name__ == "__main__":
    unittest.main()
