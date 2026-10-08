import argparse
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.sampling.draft_sampling import DraftSamplingParams, build_draft_probs
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class TestDraftSampling(unittest.TestCase):
    def test_inherit_and_override_without_mutating_target(self):
        info = sampling_info()
        originals = {
            name: getattr(info, name).clone()
            for name in ("temperatures", "top_ks", "top_ps")
        }
        for temperature, top_k, top_p in itertools.product(
            (None, 0.0, 0.7), (None, -1, 1, 3), (None, 0.6, 1.0)
        ):
            with (
                self.subTest(temperature=temperature, top_k=top_k, top_p=top_p),
                patch(
                    "sglang.srt.runtime_context.get_spec",
                    return_value=draft_config(temperature, top_k, top_p),
                ),
            ):
                params = DraftSamplingParams.from_sampling_info(info)
                effective_k = TOP_K_ALL if top_k == -1 else top_k
                for name, override in (
                    ("temperatures", temperature),
                    ("top_ks", effective_k),
                    ("top_ps", top_p),
                ):
                    expected = (
                        originals[name]
                        if override is None
                        else torch.full_like(originals[name], override)
                    )
                    torch.testing.assert_close(getattr(params, name), expected)
                    torch.testing.assert_close(getattr(info, name), originals[name])

    def test_disabled_cutoffs_keep_full_model_support(self):
        info = sampling_info()
        logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).repeat(2, 1)
        with patch("sglang.srt.runtime_context.get_spec", return_value=draft_config()):
            inherited = build_draft_probs(
                logits, DraftSamplingParams.from_sampling_info(info)
            )
        with patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=draft_config(top_k=-1, top_p=1.0),
        ):
            untruncated = build_draft_probs(
                logits, DraftSamplingParams.from_sampling_info(info)
            )
        self.assertEqual(torch.count_nonzero(inherited[0]).item(), 1)
        torch.testing.assert_close(
            untruncated, torch.softmax(logits / info.temperatures, dim=-1)
        )

    def test_target_greedy_rows_stay_greedy_with_overrides(self):
        info = sampling_info()
        info.top_ks[0] = 1
        for top_k in (-1, 1, 4, 1 << 40):
            with (
                self.subTest(top_k=top_k),
                patch(
                    "sglang.srt.runtime_context.get_spec",
                    return_value=draft_config(2.0, top_k, 1.0),
                ),
            ):
                params = DraftSamplingParams.from_sampling_info(info)
            self.assertEqual(
                params.top_ks.tolist(),
                [1, TOP_K_ALL if top_k == -1 else min(top_k, TOP_K_ALL)],
            )
            self.assertEqual(params.greedy_mask.tolist(), [True, top_k == 1])
            probs = build_draft_probs(
                torch.tensor([[2.0, 1.0, 0.0]]).repeat(2, 1), params
            )
            torch.testing.assert_close(probs[0], torch.tensor([1.0, 0.0, 0.0]))

    def test_top_k_precedes_top_p(self):
        params = DraftSamplingParams(
            torch.ones(1, 1), torch.tensor([2]), torch.tensor([0.55])
        )
        logits = torch.tensor([[0.4, 0.3, 0.2, 0.1]]).log()
        # Top-k changes the mass basis: 0.4 / (0.4 + 0.3) exceeds 0.55.
        torch.testing.assert_close(
            build_draft_probs(logits, params), torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        )

    def test_candidate_support_and_request_broadcast(self):
        logits = torch.tensor([[[[2.0, 1.0], [0.0, 3.0]]], [[[2.0, 1.0], [0.0, 3.0]]]])
        params = DraftSamplingParams(
            torch.tensor([[0.0], [2.0]]),
            torch.tensor([50, 50]),
            torch.ones(2),
        )
        q = build_draft_probs(logits, params)
        self.assertEqual(q.shape, logits.shape)
        torch.testing.assert_close(q[0], torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
        torch.testing.assert_close(q[1], torch.softmax(logits[1] / 2.0, dim=-1))
        torch.testing.assert_close(q.sum(-1), torch.ones_like(q[..., 0]))

    def test_greedy_proposal_remains_one_hot_with_tied_logits(self):
        params = DraftSamplingParams(
            torch.tensor([[0.0], [1.0]]), torch.tensor([TOP_K_ALL, 1]), torch.ones(2)
        )
        torch.testing.assert_close(
            build_draft_probs(torch.tensor([[2.0, 2.0, 1.0]]).repeat(2, 1), params),
            torch.tensor([[1.0, 0.0, 0.0]]).repeat(2, 1),
        )

    def test_graph_staging_resets_padding_and_keeps_addresses(self):
        params = DraftSamplingParams.create(4, "cpu")
        addresses = [
            tensor.data_ptr()
            for tensor in (params.temperatures, params.top_ks, params.top_ps)
        ]
        with patch(
            "sglang.srt.runtime_context.get_spec",
            return_value=draft_config(0.25, -1, 0.5),
        ):
            params.copy_from(sampling_info(), 2)
            params.copy_from(sampling_info(), 1)
        torch.testing.assert_close(
            params.temperatures[:, 0], torch.tensor([0.25, 1.0, 1.0, 1.0])
        )
        self.assertEqual(params.top_ks.tolist(), [TOP_K_ALL] * 4)
        torch.testing.assert_close(params.top_ps, torch.tensor([0.5, 1.0, 1.0, 1.0]))
        self.assertEqual(
            addresses,
            [
                tensor.data_ptr()
                for tensor in (params.temperatures, params.top_ks, params.top_ps)
            ],
        )

    def test_rejection_correction_preserves_target_with_sparse_or_greedy_q(self):
        target = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        for temperature in (0.0, 0.5, 2.0):
            params = DraftSamplingParams(
                torch.tensor([[temperature]]), torch.tensor([2]), torch.tensor([0.8])
            )
            proposal = build_draft_probs(torch.tensor([[3.0, 2.0, 1.0, 0.0]]), params)[
                0
            ].double()
            accepted_mass = torch.minimum(target, proposal)
            rejected_mass = 1.0 - accepted_mass.sum()
            residual = (target - proposal).clamp_min(0)
            output = accepted_mass + rejected_mass * residual / residual.sum()
            torch.testing.assert_close(output, target, rtol=1e-6, atol=1e-8)

    def test_empty_batch(self):
        q = build_draft_probs(
            torch.empty(0, 4, 16), DraftSamplingParams.create(0, "cpu")
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

    def test_invalid_temperature(self):
        for temperature in (-1.0, float("nan"), float("inf")):
            args = ServerArgs(
                model_path="dummy", speculative_draft_temperature=temperature
            )
            with (
                self.subTest(temperature=temperature),
                self.assertRaisesRegex(ValueError, "finite and non-negative"),
            ):
                handle_speculative_decoding(args)

    def test_invalid_cutoff_cli_values(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        invalid_values = {
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

    def test_invalid_programmatic_top_k(self):
        for top_k in (0.5, 2.0, True, float("nan"), float("inf")):
            args = ServerArgs(model_path="dummy", speculative_draft_top_k=top_k)
            with (
                self.subTest(top_k=top_k),
                self.assertRaisesRegex(ValueError, "--speculative-draft-top-k"),
            ):
                handle_speculative_decoding(args)


if __name__ == "__main__":
    unittest.main()
