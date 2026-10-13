import argparse
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.environ import envs
from sglang.srt.sampling.draft_sampling import DraftSamplingParams, build_draft_probs
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.spec_utils import sample_draft_proposal
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def params(temperatures, top_ks, top_ps):
    return DraftSamplingParams(
        torch.tensor(temperatures, dtype=torch.float32),
        torch.tensor(top_ks, dtype=torch.int32),
        torch.tensor(top_ps, dtype=torch.float32),
    )


class TestDraftSampling(unittest.TestCase):
    def setUp(self):
        # CPU exercises the multinomial fallback; Gumbel's row argmax is GPU-only.
        self.enterContext(envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False))

    def test_overrides_replace_target_values_but_keep_greedy_rows(self):
        info = SimpleNamespace(
            temperatures=torch.tensor([[0.5], [1.5], [1.0]]),
            top_ks=torch.tensor([2, TOP_K_ALL, 1], dtype=torch.int32),
            top_ps=torch.tensor([0.8, 1.0, 1.0]),
        )
        target_top_ks = info.top_ks.clone()
        for temperature, top_k, top_p in itertools.product(
            (None, 0.0, 0.7), (None, -1, 1, 3), (None, 0.6)
        ):
            spec = SimpleNamespace(
                speculative_draft_temperature=temperature,
                speculative_draft_top_k=top_k,
                speculative_draft_top_p=top_p,
            )
            with (
                self.subTest(temperature=temperature, top_k=top_k, top_p=top_p),
                patch("sglang.srt.runtime_context.get_spec", return_value=spec),
            ):
                draft = DraftSamplingParams.from_sampling_info(info)
            expected_ks = target_top_ks.tolist()
            if top_k is not None:
                k = TOP_K_ALL if top_k == -1 else top_k
                expected_ks = [1 if t <= 1 else k for t in expected_ks]
            if temperature == 0:
                expected_ks = [1, 1, 1]
            self.assertEqual(draft.top_ks.tolist(), expected_ks)
            expected_t = [0.5, 1.5, 1.0] if not temperature else [temperature] * 3
            torch.testing.assert_close(draft.temperatures, torch.tensor(expected_t))
            expected_p = [0.8, 1.0, 1.0] if top_p is None else [top_p] * 3
            torch.testing.assert_close(draft.top_ps, torch.tensor(expected_p))
            self.assertTrue(torch.equal(info.top_ks, target_top_ks))

    def test_top_k_precedes_top_p(self):
        logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
        # Top-k changes the mass basis: 0.4 / (0.4 + 0.3) exceeds 0.55.
        torch.testing.assert_close(
            build_draft_probs(logits, params([1.0], [2], [0.55])),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def test_greedy_rows_propose_the_first_argmax_with_point_mass_q(self):
        logits = torch.tensor([[2.0, 2.0, 1.0], [0.2, 0.8, 0.6]])
        q, sampled_probs, tokens = sample_draft_proposal(
            logits, params([1.0, 1.0], [1, 1], [1.0, 1.0])
        )
        torch.testing.assert_close(q, torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        self.assertEqual(tokens.flatten().tolist(), [0, 1])
        torch.testing.assert_close(sampled_probs, torch.ones(2, 1))

    def test_sample_frequencies_match_returned_truncated_q(self):
        torch.manual_seed(100)
        batch_size = 20000
        # top-k keeps the first three tokens; top-p=0.7 then keeps two.
        logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log().repeat(batch_size, 1)
        draft = params([1.0] * batch_size, [3] * batch_size, [0.7] * batch_size)
        q, sampled_probs, tokens = sample_draft_proposal(logits, draft)
        expected = torch.tensor([4 / 7, 3 / 7, 0.0, 0.0])
        torch.testing.assert_close(q, expected.expand_as(q))
        torch.testing.assert_close(sampled_probs, q.gather(1, tokens))
        frequencies = torch.bincount(tokens.flatten(), minlength=4) / batch_size
        torch.testing.assert_close(frequencies, expected, atol=0.015, rtol=0)

    def test_single_graph_retains_final_q_for_each_step(self):
        runner = object.__new__(MultiLayerEagleDraftExtendCudaGraphRunner)
        runner.prune_draft_extend_logits = True
        runner.buffers = SimpleNamespace(
            sampling_params=params([1.0, 1.0], [1, 2], [1.0, 1.0]),
            draft_probs=torch.full((2, 2, 3), -1.0),
        )
        ret = SimpleNamespace(
            next_token_logits=torch.tensor([[0.0, 2.0, 1.0], [1.0, 0.0, 2.0]])
        )
        expected = torch.tensor(
            [[0.0, 1.0, 0.0], [1 / (1 + torch.e), 0.0, torch.e / (1 + torch.e)]]
        )
        for step in range(2):
            runner.step = step
            runner._sample_draft_proposal(ret, 2)
            stored_q = runner.buffers.draft_probs[:, step]
            torch.testing.assert_close(stored_q, expected)
            torch.testing.assert_close(ret.topk_p, stored_q.gather(1, ret.topk_index))

    def test_empty_batch(self):
        q = build_draft_probs(
            torch.empty(0, 4, 16), DraftSamplingParams.greedy(0, "cpu")
        )
        self.assertEqual(q.shape, (0, 4, 16))


class TestDraftSamplingArgs(unittest.TestCase):
    def test_cli_validation(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        values = {
            "temperature": (["0", "0.7"], ["-1", "nan", "inf"]),
            "top-k": (["-1", "1"], ["0", "-2"]),
            "top-p": (["1.0", "0.5"], ["0", "-0.1", "1.1", "nan", "inf"]),
        }
        for name, (valid, invalid) in values.items():
            flag = f"--speculative-draft-{name}"
            for value in valid + invalid:
                args = ServerArgs.from_cli_args(
                    parser.parse_args(["--model-path", "dummy", f"{flag}={value}"])
                )
                with self.subTest(flag=flag, value=value):
                    if value in valid:
                        handle_speculative_decoding(args)
                    else:
                        with self.assertRaisesRegex(ValueError, flag):
                            handle_speculative_decoding(args)


if __name__ == "__main__":
    unittest.main()
