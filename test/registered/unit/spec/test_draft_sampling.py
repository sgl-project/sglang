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


def sampling_info():
    return SimpleNamespace(
        temperatures=torch.tensor([[0.5], [1.5]]),
        top_ks=torch.tensor([2, TOP_K_ALL], dtype=torch.int32),
        top_ps=torch.tensor([0.8, 1.0]),
    )


def draft_config(temperature=None, top_k=None, top_p=None):
    return SimpleNamespace(
        speculative_draft_temperature=temperature,
        speculative_draft_top_k=top_k,
        speculative_draft_top_p=top_p,
    )


def params(temperatures, top_ks, top_ps):
    return DraftSamplingParams(
        torch.tensor(temperatures, dtype=torch.float32),
        torch.tensor(top_ks, dtype=torch.int32),
        torch.tensor(top_ps, dtype=torch.float32),
    )


def resolve(info, **overrides):
    with patch(
        "sglang.srt.runtime_context.get_spec", return_value=draft_config(**overrides)
    ):
        return DraftSamplingParams.from_sampling_info(info)


class TestDraftSampling(unittest.TestCase):
    def setUp(self):
        # CPU exercises the multinomial fallback; Gumbel's row argmax is GPU-only.
        self.enterContext(envs.SGLANG_OPT_USE_GUMBEL_SAMPLE.override(False))

    def test_inherit_and_override_without_mutating_target(self):
        info = sampling_info()
        target = {
            "temperatures": info.temperatures.reshape(-1).clone(),
            "top_ks": info.top_ks.clone(),
            "top_ps": info.top_ps.clone(),
        }
        for temperature, top_k, top_p in itertools.product(
            (None, 0.0, 0.7), (None, -1, 1, 3), (None, 0.6, 1.0)
        ):
            with self.subTest(temperature=temperature, top_k=top_k, top_p=top_p):
                draft = resolve(info, temperature=temperature, top_k=top_k, top_p=top_p)
                expected_k = TOP_K_ALL if top_k == -1 else top_k
                if temperature == 0:
                    # Draft temperature 0 is encoded as greedy (top_k = 1).
                    temperature, expected_k = None, 1
                for name, override in (
                    ("temperatures", temperature),
                    ("top_ks", expected_k),
                    ("top_ps", top_p),
                ):
                    expected = (
                        target[name]
                        if override is None
                        else torch.full_like(target[name], override)
                    )
                    torch.testing.assert_close(getattr(draft, name), expected)
                torch.testing.assert_close(info.top_ks, target["top_ks"])

    def test_target_greedy_rows_stay_greedy_with_overrides(self):
        info = sampling_info()
        info.top_ks[0] = 1
        for top_k in (-1, 1, 4, 1 << 40):
            with self.subTest(top_k=top_k):
                draft = resolve(info, temperature=2.0, top_k=top_k, top_p=1.0)
                self.assertEqual(
                    draft.top_ks.tolist(),
                    [1, TOP_K_ALL if top_k == -1 else min(top_k, TOP_K_ALL)],
                )
                self.assertEqual(draft.greedy_mask.tolist(), [True, top_k == 1])

    def test_disabled_cutoffs_keep_full_model_support(self):
        info = sampling_info()
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).repeat(2, 1)
        inherited = build_draft_probs(logits, resolve(info))
        untruncated = build_draft_probs(logits, resolve(info, top_k=-1, top_p=1.0))
        self.assertEqual(torch.count_nonzero(inherited[0]).item(), 1)
        torch.testing.assert_close(
            untruncated, torch.softmax(logits / info.temperatures, dim=-1)
        )

    def test_top_k_precedes_top_p(self):
        logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
        # Top-k changes the mass basis: 0.4 / (0.4 + 0.3) exceeds 0.55.
        torch.testing.assert_close(
            build_draft_probs(logits, params([1.0], [2], [0.55])),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

    def test_candidate_support_and_request_broadcast(self):
        logits = torch.tensor([[[[2.0, 1.0], [0.0, 3.0]]], [[[2.0, 1.0], [0.0, 3.0]]]])
        q = build_draft_probs(logits, params([1.0, 2.0], [1, 50], [1.0, 1.0]))
        self.assertEqual(q.shape, logits.shape)
        torch.testing.assert_close(q[0], torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
        torch.testing.assert_close(q[1], torch.softmax(logits[1] / 2.0, dim=-1))

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

    def test_rejection_correction_preserves_target_with_sparse_or_greedy_q(self):
        target = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        for temperature, top_k in ((1.0, 1), (0.5, 2), (2.0, 2)):
            proposal = build_draft_probs(
                torch.tensor([[3.0, 2.0, 1.0, 0.0]]),
                params([temperature], [top_k], [0.8]),
            )[0].double()
            accepted_mass = torch.minimum(target, proposal)
            rejected_mass = 1.0 - accepted_mass.sum()
            residual = (target - proposal).clamp_min(0)
            output = accepted_mass + rejected_mass * residual / residual.sum()
            torch.testing.assert_close(output, target, rtol=1e-6, atol=1e-8)

    def test_graph_staging_writes_in_place(self):
        buffers = DraftSamplingParams.greedy(4, "cpu")
        addresses = [t.data_ptr() for t in (buffers.temperatures, buffers.top_ks)]
        with patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=draft_config(0.25, -1, 0.5),
        ):
            buffers.copy_from(sampling_info(), 2)
        torch.testing.assert_close(buffers.temperatures[:2], torch.tensor([0.25] * 2))
        self.assertEqual(buffers.top_ks[:2].tolist(), [TOP_K_ALL] * 2)
        torch.testing.assert_close(buffers.top_ps[:2], torch.tensor([0.5] * 2))
        self.assertEqual(
            addresses, [t.data_ptr() for t in (buffers.temperatures, buffers.top_ks)]
        )

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
    def test_default_and_cli(self):
        defaults = ServerArgs(model_path="dummy")
        for name in ("temperature", "top_k", "top_p"):
            self.assertIsNone(getattr(defaults, f"speculative_draft_{name}"))
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = ServerArgs.from_cli_args(
            parser.parse_args(
                [
                    "--model-path",
                    "dummy",
                    "--speculative-draft-temperature",
                    "0.7",
                    "--speculative-draft-top-k",
                    "-1",
                    "--speculative-draft-top-p",
                    "1.0",
                ]
            )
        )
        self.assertEqual(args.speculative_draft_temperature, 0.7)
        self.assertEqual(args.speculative_draft_top_k, -1)
        self.assertEqual(args.speculative_draft_top_p, 1.0)
        handle_speculative_decoding(args)

    def test_invalid_cli_values(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        invalid_values = {
            "temperature": ("-1", "nan", "inf"),
            "top-k": ("0", "-2"),
            "top-p": ("0", "-0.1", "1.1", "nan", "inf"),
        }
        for name, values in invalid_values.items():
            for value in values:
                flag = f"--speculative-draft-{name}"
                args = ServerArgs.from_cli_args(
                    parser.parse_args(["--model-path", "dummy", f"{flag}={value}"])
                )
                with (
                    self.subTest(flag=flag, value=value),
                    self.assertRaisesRegex(ValueError, flag),
                ):
                    handle_speculative_decoding(args)


if __name__ == "__main__":
    unittest.main()
