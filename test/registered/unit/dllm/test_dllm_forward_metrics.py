"""Actual dLLM forward accounting across sync, FDFO and prefill."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.dllm.algorithm.low_confidence import LowConfidence
from sglang.srt.dllm.config import DllmConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _config(fdfo=False):
    return DllmConfig("LowConfidence", {"threshold": 0.95}, 4, -1, 8, fdfo)


class _ForwardOracle:
    """Fast row unmasks all at once; slow row unmasks one token each step."""

    def __init__(self):
        self.calls = []

    def forward(self, batch, pp_proxy_tensors=None):
        self.calls.append((list(batch.rids), batch.input_ids.clone()))
        logits = torch.zeros(batch.batch_size, 4, 3)
        for i, rid in enumerate(batch.rids):
            logits[i, :, 0] = 12 if rid == "fast" else 0
        return SimpleNamespace(
            logits_output=SimpleNamespace(full_logits=logits.flatten(0, 1)),
            can_run_graph=False,
        )


def _batch(rids, ids=None):
    return SimpleNamespace(
        rids=rids,
        batch_size=len(rids),
        dllm_prompt_mask=None,
        input_ids=(
            torch.full((len(rids) * 4,), -1, dtype=torch.long)
            if ids is None
            else torch.tensor(ids, dtype=torch.long)
        ),
    )


class TestActualForwardCount(unittest.TestCase):
    def test_sync_charges_finished_row_for_actual_batched_work(self):
        runner = _ForwardOracle()
        batch = _batch(["fast", "slow"])
        result = LowConfidence(_config()).run(runner, batch)
        self.assertEqual(len(result), 6, "Algorithm result must expose actual work")
        self.assertEqual(result.forward_count, len(runner.calls))
        self.assertEqual(result.forward_count, 5)  # Four unmask rounds plus final KV.
        self.assertFalse((runner.calls[1][1][:4] == -1).any())
        self.assertTrue((runner.calls[1][1][4:] == -1).any())
        self.assertTrue(all(rids == ["fast", "slow"] for rids, _ in runner.calls))

    def test_fdfo_stops_charging_row_after_it_leaves(self):
        runner = _ForwardOracle()
        algorithm = LowConfidence(_config(fdfo=True))
        batch = _batch(["fast", "slow"])
        counts = {"fast": 0, "slow": 0}
        states = None
        for _ in range(8):
            if not batch.rids:
                break
            result = algorithm.run(runner, batch, states)
            self.assertEqual(result.forward_count, 1)
            for rid in batch.rids:
                counts[rid] += result.forward_count
            keep = [
                i
                for i, accepted in enumerate(result.accept_length_per_req_cpu)
                if not accepted
            ]
            rows = batch.input_ids.view(-1, 4)[keep].flatten().tolist()
            states = [result.dllm_algo_state[i] for i in keep]
            batch = _batch([batch.rids[i] for i in keep], rows)
        self.assertFalse(batch.rids, "FDFO did not finish within eight rounds")
        self.assertEqual(counts, {"fast": 2, "slow": 5})
        self.assertEqual(
            counts,
            {rid: sum(rid in rids for rids, _ in runner.calls) for rid in counts},
        )

    def test_pure_prefill_counts_even_without_output(self):
        runner = _ForwardOracle()
        result = LowConfidence(_config()).run(runner, _batch(["fast"], [1] * 4))
        self.assertEqual(result.next_token_ids, [])
        self.assertEqual(result.forward_count, 1)
        self.assertEqual(len(runner.calls), 1)

    def test_fdfo_counts_multiple_calls_and_resets_each_run(self):
        class ExtraForwardAlgorithm(LowConfidence):
            def prepare_inputs(self, model_runner, forward_batch, states):
                # Model a round that performs extra real work before denoising.
                self._forward(model_runner, forward_batch)

        runner = _ForwardOracle()
        algorithm = ExtraForwardAlgorithm(_config(fdfo=True))
        for _ in range(2):
            previous_calls = len(runner.calls)
            result = algorithm.run(runner, _batch(["fast"]))
            actual_calls = len(runner.calls) - previous_calls
            self.assertGreater(actual_calls, 1)
            self.assertEqual(result.forward_count, actual_calls)


if __name__ == "__main__":
    unittest.main()
