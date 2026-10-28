# SPDX-License-Identifier: Apache-2.0
"""Clover acceptance alignment, V2 state lifetime and config contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.model_overrides.clover_moe import _clover_overrides
from sglang.srt.arg_groups.overrides import declare_resolution, resolving_view
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.configs.clover_moe import CloverMoeConfig
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.clover_config import (
    configure_clover,
    validate_clover_request,
)
from sglang.srt.speculative.clover_info import CloverDraftInput, CloverVerifyInput
from sglang.srt.speculative.clover_utils import (
    clover_accept_tokens,
    clover_correct_drafts,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCloverAcceptance(CustomTestCase):
    def test_first_rejection_and_bonus_alignment(self):
        candidates = torch.tensor([[7, 1, 2, 3]]).repeat(4, 1)
        logits = torch.full((4, 4, 8), -20.0)
        logits[:, :, 6] = 20.0
        # The rows accept 0, 1, 2 and 3 drafts. A match after rejection must
        # never be emitted; logits at j predict the input token at j+1.
        for row in range(4):
            for j in range(row):
                logits[row, j].fill_(-20)
                logits[row, j, j + 1] = 20
        logits[0, 2].fill_(-20)
        logits[0, 2, 3] = 20
        counts = clover_correct_drafts(logits, candidates)
        self.assertEqual(counts.tolist(), [0, 1, 2, 3])
        bonus = logits[torch.arange(4), counts.long()].argmax(-1)
        output = clover_accept_tokens(candidates, counts, bonus)
        self.assertEqual(
            [row[: n + 1].tolist() for row, n in zip(output, counts.tolist())],
            [[6], [1, 6], [1, 2, 6], [1, 2, 3, 6]],
        )

    def test_exact_rejects_non_argmax_and_resolves_ties(self):
        logits = torch.tensor([[[0.6, 0.4], [0.5, 0.5], [0.1, 0.9]]]).log()
        # Threshold accepts the non-argmax proposal; exact must reject it.
        candidates = torch.tensor([[1, 1, 0]])
        self.assertEqual(clover_correct_drafts(logits, candidates, 0.4, 10).item(), 2)
        self.assertEqual(
            clover_correct_drafts(logits, candidates, 0.4, 10, "exact").item(), 0
        )
        # A tie accepts only the same index chosen by greedy argmax.
        candidates = torch.tensor([[1, 0, 1], [1, 0, 0]])
        counts = clover_correct_drafts(
            logits.expand(2, -1, -1), candidates, accept_rule="exact"
        )
        self.assertEqual(counts.tolist(), [1, 2])

    def test_entropy_and_zero_probabilities(self):
        logits = torch.tensor([[[0.0, -torch.inf], [0.0, -torch.inf]]])
        candidates = torch.tensor([[1, 0]])
        self.assertEqual(clover_correct_drafts(logits, candidates).item(), 1)
        uniform = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
        candidates = torch.tensor([[0, 2, 3, 4]])
        self.assertEqual(clover_correct_drafts(uniform, candidates).item(), 3)

    def test_state_filter_merge_and_owned_storage(self):
        bonus = torch.tensor([11, 22, 33])
        state = CloverDraftInput.from_tokens(bonus, torch.tensor([10, 20, 30]))
        bonus.fill_(99)
        state.filter_batch(torch.tensor([2, 0]), [2, 0])
        state.merge_batch(
            CloverDraftInput.from_tokens(torch.tensor([44]), torch.tensor([40]))
        )
        self.assertEqual(state.bonus_tokens.tolist(), [33, 11, 44])
        self.assertEqual(state.new_seq_lens.tolist(), [30, 10, 40])
        self.assertTrue(state.is_draft_input())
        verify = CloverVerifyInput(None, None, 4)
        self.assertTrue(verify.is_verify_input())
        self.assertEqual(verify.num_tokens_per_req, 4)

    def test_algorithm_spelling_survives_model_and_spec_resolution(self):
        # Model overrides run before the speculative hook. Both must agree on
        # the spelling read by the preview fast-plan gate after publication.
        from sglang.srt.runtime_context import (
            get_spec,
            publish,
            reset_context,
        )

        model = SimpleNamespace(
            hf_config=CloverMoeConfig(architectures=["CloverMoeModelLM"])
        )
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            linear_attn_backend="triton",
            speculative_algorithm="ClOvEr",
        )
        declare_resolution(
            args, "clover_model", **_clover_overrides(args, model.hf_config)
        )
        with patch(
            "sglang.srt.speculative.clover_config.model_config_of",
            return_value=model,
        ):
            handle_speculative_decoding(args)
        self.assertEqual(resolving_view(args).speculative_algorithm, "CLOVER")
        self.assertEqual(args.speculative_algorithm, "ClOvEr")
        try:
            publish(args, role="scheduler")
            self.assertEqual(get_spec().speculative_algorithm, "CLOVER")
            self.assertIs(
                SpeculativeAlgorithm.from_string(get_spec().speculative_algorithm),
                SpeculativeAlgorithm.CLOVER,
            )
        finally:
            reset_context()

    def test_model_requires_explicit_clover_without_overriding_algorithm(self):
        from sglang.srt.arg_groups.arg_utils import resolvable_fields

        self.assertNotIn("speculative_algorithm", resolvable_fields(ServerArgs))
        model = CloverMoeConfig()
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            linear_attn_backend="triton",
            speculative_algorithm=None,
        )
        with self.assertRaisesRegex(
            ValueError, "requires --speculative-algorithm clover"
        ):
            _clover_overrides(args, model)
        self.assertIsNone(resolving_view(args).speculative_algorithm)
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            linear_attn_backend="triton",
            speculative_algorithm="CLOVER",
        )
        result = _clover_overrides(args, model)
        self.assertNotIn("speculative_algorithm", result)
        self.assertEqual(resolving_view(args).speculative_algorithm, "CLOVER")

    def test_validated_replica_dp_and_ep_configs(self):
        for kwargs in (
            {"dp_size": 2},
            {"tp_size": 2, "ep_size": 2},
            {"tp_size": 2, "ep_size": 2, "moe_runner_backend": "triton"},
        ):
            with self.subTest(**kwargs):
                args = ServerArgs(
                    model_path="dummy",
                    device="cuda",
                    linear_attn_backend="triton",
                    speculative_algorithm="CLOVER",
                    **kwargs,
                )
                declare_resolution(
                    args, "clover_model", **_clover_overrides(args, CloverMoeConfig())
                )
                cfg = resolving_view(args)
                self.assertEqual(cfg.dp_size, kwargs.get("dp_size", 1))
                self.assertEqual(cfg.ep_size, kwargs.get("ep_size", 1))
                if cfg.ep_size > 1:
                    self.assertEqual(cfg.moe_runner_backend, "triton")

    def test_parallel_gate_keeps_unvalidated_dispatch_disabled(self):
        # Replica DP must not enable attention DP, and later A2A resolution
        # must not silently select an EP implementation outside the tested path.
        for kwargs, error in (
            ({"pp_size": 2}, "pipeline parallel"),
            ({"attn_dp_size": 2, "tp_size": 2}, "attention DP/CP"),
            ({"attn_cp_size": 2, "tp_size": 2}, "attention DP/CP"),
            ({"dcp_size": 2, "tp_size": 2}, "DCP"),
            ({"moe_a2a_backend": "deepep", "tp_size": 2}, "moe-a2a-backend none"),
            (
                {
                    "ep_size": 2,
                    "tp_size": 2,
                    "moe_runner_backend": "flashinfer_cutlass",
                },
                "moe-runner-backend triton",
            ),
        ):
            with self.subTest(**kwargs):
                args = ServerArgs(
                    model_path="dummy",
                    device="cuda",
                    linear_attn_backend="triton",
                    speculative_algorithm="CLOVER",
                    **kwargs,
                )
                with self.assertRaisesRegex(ValueError, error):
                    _clover_overrides(args, CloverMoeConfig())


class TestCloverConfig(CustomTestCase):
    def configure(self, **kwargs):
        args = ServerArgs(model_path="dummy", speculative_algorithm="clover", **kwargs)
        model = SimpleNamespace(
            hf_config=CloverMoeConfig(architectures=["CloverMoeModelLM"])
        )
        with patch(
            "sglang.srt.speculative.clover_config.model_config_of", return_value=model
        ):
            configure_clover(args)
        return args

    def test_resolved_defaults_preserve_input(self):
        args = self.configure()
        cfg = resolving_view(args)
        self.assertEqual(cfg.speculative_num_draft_tokens, CloverMoeConfig().block_size)
        self.assertTrue(cfg.disable_overlap_schedule)
        self.assertFalse(cfg.enable_mixed_chunk)
        self.assertIsNone(args.speculative_num_draft_tokens)
        self.assertEqual(
            resolving_view(
                self.configure(speculative_num_draft_tokens=8)
            ).speculative_num_draft_tokens,
            8,
        )

    def test_penalty_defaults_preserve_explicit_server_preferences(self):
        defaults = resolving_view(self.configure()).preferred_sampling_params
        self.assertEqual(
            defaults, {"presence_penalty": 0.9, "repetition_penalty": 1.05}
        )
        args = self.configure(
            preferred_sampling_params='{"presence_penalty": 0, "temperature": 0.6}'
        )
        self.assertEqual(
            resolving_view(args).preferred_sampling_params,
            {"presence_penalty": 0, "repetition_penalty": 1.05, "temperature": 0.6},
        )

    def test_verify_backend_selection(self):
        for backend in ("triton", "flashinfer"):
            with self.subTest(backend=backend):
                args = self.configure(linear_attn_verify_backend=backend)
                self.assertEqual(
                    resolving_view(args).linear_attn_verify_backend, backend
                )
                self.assertEqual(args.linear_attn_verify_backend, backend)

    def test_invalid_config(self):
        for kwargs in (
            {"speculative_num_draft_tokens": 1},
            {"speculative_num_draft_tokens": 64},
            {"speculative_clover_accept_rule": "unknown"},
            {"speculative_clover_accept_ratio": float("nan")},
            {"speculative_clover_accept_ratio": 1.1},
            {"speculative_clover_accept_delta": -1.0},
            {"speculative_clover_accept_delta": float("inf")},
            {"dllm_algorithm": "Clover"},
            {"linear_attn_verify_backend": "cutedsl"},
            {"enable_linear_replayssm_spec": True},
            {"speculative_draft_model_path": "another-model"},
            {"enable_watermark": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.configure(**kwargs)

    def test_request_sampling_contract(self):
        fields = dict(
            tokenizer=None,
            return_logprob=False,
            return_sampling_mask=False,
            custom_logit_processor=None,
            input_embeds=None,
            positional_embed_overrides=None,
            return_hidden_states=False,
        )
        req = SimpleNamespace(
            **fields,
            sampling_params=SamplingParams(
                temperature=0.8,
                top_p=0.9,
                top_k=20,
                custom_params={"dllm_think_end_bias": 0},
            ),
        )
        self.assertIsNone(validate_clover_request(req))
        for field, value in (
            ("min_p", 0.1),
            ("regex", "[0-9]+"),
        ):
            req = SimpleNamespace(
                **fields, sampling_params=SamplingParams(**{field: value})
            )
            self.assertIn(field, validate_clover_request(req))
        req = SimpleNamespace(
            **{**fields, "return_logprob": True}, sampling_params=SamplingParams()
        )
        self.assertIn("return_logprob", validate_clover_request(req))


if __name__ == "__main__":
    unittest.main()
