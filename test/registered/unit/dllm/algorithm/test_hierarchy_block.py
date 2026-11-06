import unittest
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm import get_algorithm_cls
from sglang.srt.dllm.algorithm.hierarchy_block import HierarchyBlock
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.scheduler import SchedulerDllmMixin
from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MASK = 99
VOCAB = 128
HIGH = 20.0
LOW = 0.5


def _algorithm(block_size=8, sub_block_size=4, threshold=0.9):
    config = DllmConfig(
        algorithm="HierarchyBlock",
        algorithm_config={"sub_block_size": sub_block_size, "threshold": threshold},
        block_size=block_size,
        mask_id=MASK,
        max_running_requests=4,
        carries_block_state=True,
    )
    return HierarchyBlock(config)


def _batch(rows, prompt_mask=None):
    input_ids = torch.tensor(rows, dtype=torch.long)
    return SimpleNamespace(
        batch_size=input_ids.shape[0],
        input_ids=input_ids.view(-1).clone(),
        dllm_prompt_mask=prompt_mask,
    )


class _Model:
    def __init__(self):
        self.causal = True

    def set_block_attention_causal(self, causal):
        self.causal = causal


class _OracleRunner:
    """Logits at block position p peak at targets[:, p + 1] (token shift)."""

    def __init__(self, targets, peaks, fail_on_forward=None):
        self.targets = torch.tensor(targets)
        self.peaks = torch.as_tensor(peaks, dtype=torch.float32)
        self.model = _Model()
        self.forwards = []
        self.fail_on_forward = fail_on_forward

    def forward(self, forward_batch, pp_proxy_tensors=None):
        if self.fail_on_forward == len(self.forwards):
            raise RuntimeError("forward failed")
        self.forwards.append((self.model.causal, forward_batch.input_ids.clone()))
        batch_size, width = self.targets.shape
        block_size = width - 1
        logits = torch.zeros(batch_size, block_size, VOCAB)
        peak = self.peaks.expand(batch_size, block_size)
        logits.scatter_(-1, self.targets[:, 1:].unsqueeze(-1), peak.unsqueeze(-1))
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits.view(-1, VOCAB)),
            can_run_graph=False,
        )


class TestHierarchyBlockRun(unittest.TestCase):
    def test_registered(self):
        self.assertIs(get_algorithm_cls("HierarchyBlock"), HierarchyBlock)

    def test_sub_blocks_resolve_in_order_and_kv_is_written_causally(self):
        targets = [[10, 11, 12, 13, 14, 15, 16, 17, 18]]
        runner = _OracleRunner(targets, HIGH)
        batch = _batch([[MASK] * 8])

        _, next_ids, _, states, _ = _algorithm().run(runner, batch, [10])

        causal_flags = [causal for causal, _ in runner.forwards]
        self.assertEqual(causal_flags, [False, False, True])
        # Sub-block 1 stays masked until sub-block 0 is resolved.
        self.assertEqual(runner.forwards[1][1][4:].tolist(), [MASK] * 4)
        self.assertEqual(next_ids[0].tolist(), targets[0][:8])
        self.assertEqual(states, [18])
        self.assertTrue(runner.model.causal)

    def test_low_confidence_still_unmasks_one_token_per_step(self):
        targets = [[10, 11, 12, 13, 14, 15, 16, 17, 18]]
        runner = _OracleRunner(targets, LOW)
        batch = _batch([[MASK] * 8])

        _, next_ids, _, _, _ = _algorithm().run(runner, batch, [10])

        # Seven masks after seeding: seven denoise forwards, then one causal.
        self.assertEqual(len(runner.forwards), 8)
        self.assertEqual([c for c, _ in runner.forwards].count(True), 1)
        self.assertEqual(next_ids[0].tolist(), targets[0][:8])

    def test_prompt_only_block_still_returns_carried_state(self):
        targets = [[1, 2, 3, 4, 5, 6, 7, 8, 42]]
        runner = _OracleRunner(targets, HIGH)
        batch = _batch([targets[0][:8]])

        _, next_ids, _, states, _ = _algorithm().run(runner, batch, [None])

        self.assertEqual([c for c, _ in runner.forwards], [True])
        self.assertEqual(next_ids, [])
        self.assertEqual(states, [42])

    def test_requests_in_a_batch_keep_their_own_state(self):
        targets = [
            [10, 11, 12, 13, 14, 15, 16, 17, 18],
            [1, 2, 3, 20, 21, 22, 23, 24, 25],
        ]
        runner = _OracleRunner(targets, [[HIGH] * 8, [LOW] * 8])
        batch = _batch([[MASK] * 8, [1, 2, 3] + [MASK] * 5])

        _, next_ids, _, states, _ = _algorithm().run(runner, batch, [10, 77])

        self.assertEqual(next_ids[0].tolist(), targets[0][:8])
        # Position 0 holds a prompt token, so the carried token is not used.
        self.assertEqual(next_ids[1].tolist(), targets[1][3:8])
        self.assertEqual(states, [18, 25])
        self.assertEqual([c for c, _ in runner.forwards].count(True), 1)
        self.assertTrue(runner.forwards[-1][0])

    def test_literal_mask_in_prompt_is_not_denoised(self):
        targets = [[5, MASK, 6, 30, 31, 32, 33, 34, 35]]
        runner = _OracleRunner(targets, HIGH)
        prompt_mask = torch.tensor([[True, True, True] + [False] * 5])
        batch = _batch([[5, MASK, 6] + [MASK] * 5], prompt_mask=prompt_mask)

        _, next_ids, _, _, _ = _algorithm().run(runner, batch, [None])

        self.assertEqual(batch.input_ids[1].item(), MASK)
        self.assertEqual(next_ids[0].tolist(), targets[0][3:8])

    def test_attention_is_causal_again_after_a_failed_denoise_forward(self):
        runner = _OracleRunner([[10] * 9], HIGH, fail_on_forward=0)
        with self.assertRaises(RuntimeError):
            _algorithm().run(runner, _batch([[MASK] * 8]), [10])
        self.assertTrue(runner.model.causal)

    def test_sub_block_size_must_divide_block_size(self):
        with self.assertRaises(ValueError):
            _algorithm(block_size=8, sub_block_size=3)


class TestHierarchyBlockScheduling(unittest.TestCase):
    def _req(self, committed_len, state, carries=True):
        dllm_config = SimpleNamespace(
            block_size=4,
            requires_separate_context_encoding=False,
            carries_block_state=carries,
        )
        return SimpleNamespace(
            dllm_config=dllm_config,
            return_logprob=False,
            logprob_start_len=-1,
            is_dllm=lambda: True,
            origin_input_ids=list(range(committed_len)),
            output_ids=[],
            dllm_algo_state=state,
        )

    def _max_prefix_len(self, req):
        input_len = len(req.origin_input_ids) + req.dllm_config.block_size
        return Req._compute_max_prefix_len(req, input_len)

    def test_prefix_match_recomputes_last_block_when_state_is_missing(self):
        # Block-aligned commit, no carried token: reuse stops one block early.
        self.assertEqual(self._max_prefix_len(self._req(8, None)), 4)
        self.assertEqual(self._max_prefix_len(self._req(8, 7)), 8)
        self.assertEqual(self._max_prefix_len(self._req(8, None, carries=False)), 8)
        # A partial last block is recomputed anyway.
        self.assertEqual(self._max_prefix_len(self._req(10, None)), 8)

    def test_block_state_is_stored_on_every_request(self):
        reqs = [SimpleNamespace(dllm_algo_state=None) for _ in range(2)]
        SchedulerDllmMixin._store_dllm_block_state(reqs, [3, 4])
        self.assertEqual([r.dllm_algo_state for r in reqs], [3, 4])
        with self.assertRaises(AssertionError):
            SchedulerDllmMixin._store_dllm_block_state(reqs, None)


if __name__ == "__main__":
    unittest.main()
