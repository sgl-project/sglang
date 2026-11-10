"""GR serving policy and native beam selection, without model loading."""

import argparse
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator, _rows_topk_logprobs
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.managers.io_struct import GenerateReqInput, TokenizedGenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestGRRequestPolicy(CustomTestCase):
    def test_flag_defaults_cli_and_metadata(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        for flags, enabled in (([], False), (["--enable-gr"], True)):
            with self.subTest(enabled=enabled):
                args = ServerArgs.from_cli_args(
                    parser.parse_args(["--model-path", "dummy", *flags])
                )
                self.assertIs(args.enable_gr, enabled)
                self.assertIs(args.resolved_dict()["enable_gr"], enabled)
        self.assertTrue(ServerArgs(model_path="dummy", enable_gr=True).enable_gr)

    def test_request_preparation_enforces_gr(self):
        for enabled, requested in (
            (True, None),
            (True, False),
            (True, True),
            (False, False),
        ):
            with self.subTest(enabled=enabled, requested=requested):
                kwargs = {"max_new_tokens": 3}
                if requested is not None:
                    kwargs["ignore_eos"] = requested
                obj = GenerateReqInput(
                    input_ids=[1, 2], sampling_params=kwargs, rid="gr"
                )
                manager = SimpleNamespace(
                    preferred_sampling_params=None,
                    sampling_params_class=SamplingParams,
                    tokenized_generate_req_type=TokenizedGenerateReqInput,
                    tokenizer=None,
                    model_config=SimpleNamespace(vocab_size=256),
                    rid_to_state={"gr": SimpleNamespace(time_stats=Mock())},
                )
                with (
                    patch(
                        "sglang.srt.managers.tokenizer_manager.get_exec",
                        return_value=SimpleNamespace(
                            features=SimpleNamespace(
                                enable_watermark=False,
                                watermark_key=None,
                                watermark_context_window=1,
                                watermark_default_enabled=False,
                                watermark_enforce_all=False,
                            )
                        ),
                    ),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.get_serving",
                        return_value=SimpleNamespace(enable_gr=enabled),
                    ),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.get_disagg",
                        return_value=SimpleNamespace(
                            disaggregation_transfer_backend="mooncake"
                        ),
                    ),
                ):
                    result = TokenizerManager._create_tokenized_object(
                        manager, obj, "", [1, 2]
                    )
                self.assertIs(
                    result.sampling_params.ignore_eos, enabled or bool(requested)
                )
                self.assertEqual(result.sampling_params.max_new_tokens, 3)
                # Normalization must not mutate caller-owned request parameters.
                self.assertEqual(obj.sampling_params, kwargs)

    def test_invalid_length(self):
        for length in (None, 0, -1, 3.5, True):
            with self.subTest(length=length):
                params = SamplingParams(max_new_tokens=length)
                with self.assertRaisesRegex(
                    ValueError, "positive integer max_new_tokens"
                ):
                    params.apply_gr_policy()

    def test_explicit_stop_settings_rejected(self):
        for kwargs in (
            {"stop_token_ids": [0]},
            {"stop": "done"},
            {"stop_regex": "done.*"},
            {"stop": [""]},
            {"stop_regex": [""]},
        ):
            with self.subTest(kwargs=kwargs):
                params = SamplingParams(max_new_tokens=3, **kwargs)
                with self.assertRaisesRegex(ValueError, "does not support"):
                    params.apply_gr_policy()
        params = SamplingParams(max_new_tokens=3, stop=[], stop_token_ids=[])
        params.apply_gr_policy()
        self.assertTrue(params.ignore_eos)


class TestGRAdmission(CustomTestCase):
    def admit(self, enabled, width, *, page_size=1, pool_size=512, **kwargs):
        params = SamplingParams(beam_width=width, n=width, max_new_tokens=3, **kwargs)
        req = SimpleNamespace(
            sampling_params=params,
            session_id=None,
            lora_id=None,
            origin_input_ids=[11] * 16,
            eos_token_ids=[0],
            tokenizer=None,
        )
        recv = SimpleNamespace(
            session_params=None,
            return_logprob=False,
            return_hidden_states=False,
            return_sampling_mask=False,
            return_routed_experts=False,
        )
        coordinator = BeamCoordinator(
            model_config=SimpleNamespace(vocab_size=256, is_encoder_decoder=False),
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            dllm_enabled=False,
            max_req_len=8192,
            req_to_token_pool=SimpleNamespace(size=pool_size, device="cpu"),
            token_to_kv_pool_allocator=None,
            tree_cache=SimpleNamespace(
                supports_swa=lambda: False, supports_mamba=lambda: False
            ),
            future_map=None,
        )
        values = {
            "get_serving": SimpleNamespace(enable_gr=enabled),
            "get_disagg": SimpleNamespace(disaggregation_mode="null"),
            "get_schedule": SimpleNamespace(page_size=page_size),
            "get_memory": SimpleNamespace(
                enable_hisparse=False, enable_hierarchical_cache=False
            ),
            "get_parallel": SimpleNamespace(pp_size=1),
            "is_dp_attention_enabled": False,
        }
        with ExitStack() as stack:
            for name, value in values.items():
                stack.enter_context(
                    patch(
                        "sglang.srt.beam_search.coordinator." + name, return_value=value
                    )
                )
            error = coordinator.validate_and_init(req, recv)
        return req, error

    def test_gr_accepts_full_vocabulary_width_and_disables_eos(self):
        for width in (16, 128, 129, 256):
            with self.subTest(width=width):
                req, error = self.admit(True, width, ignore_eos=False)
                self.assertIsNone(error)
                self.assertEqual(req.beam_group.num_candidates, width)
                self.assertEqual(req.beam_group.stop_token_ids.numel(), 0)
                self.assertTrue(req.sampling_params.ignore_eos)

    def test_ordinary_admission_and_stop_ids_preserved(self):
        req, error = self.admit(False, 128)
        self.assertIsNone(error)
        self.assertEqual(req.beam_group.num_candidates, 256)
        self.assertEqual(req.beam_group.stop_token_ids.tolist(), [0])
        for width in (129, 256):
            with self.subTest(width=width):
                _, error = self.admit(False, width, ignore_eos=True)
                self.assertIn("too large for the vocabulary", error)

    def test_returned_scores_use_gr_cumulative_or_default_length_penalty(self):
        # Exercise admission, real selection, finalization and HTTP serialization.
        from sglang.srt.beam_search.output import _build_beam_search_out_dict

        for enabled in (False, True):
            for depth in (1, 3, 5):
                with self.subTest(enabled=enabled, depth=depth):
                    req, error = self.admit(enabled, 2, ignore_eos=True)
                    self.assertIsNone(error)
                    group = req.beam_group
                    group.max_new_tokens = depth
                    for step in range(depth):
                        rows = group.frontier_cum_logprobs.numel()
                        count = group.num_candidates
                        scores = (
                            -torch.arange(1, count + 1, dtype=torch.float32).expand(
                                rows, -1
                            )
                            / 8
                        )
                        tokens = torch.arange(1, count + 1).expand(rows, -1)
                        if step == depth - 1:
                            group.advance_final_frontier(
                                select_final_topk(
                                    group.frontier_cum_logprobs, scores, tokens, 2
                                ),
                                step,
                            )
                        else:
                            group.advance_frontier(
                                joint_select(
                                    group.frontier_cum_logprobs,
                                    scores,
                                    tokens,
                                    group.stop_token_ids,
                                    2,
                                ),
                                step,
                            )
                        group.commit_pending()
                    results = group.finalize()
                    self.assertEqual(len(results), 2)
                    for result in results:
                        expected = (
                            result.cum_logprob
                            if enabled
                            else result.cum_logprob / depth
                        )
                        self.assertEqual(result.beam_score, expected)
                    output = SimpleNamespace(
                        sequences=[
                            SimpleNamespace(
                                tokens=result.tokens,
                                cum_logprob=result.cum_logprob,
                                beam_score=result.beam_score,
                                finish_reason={"type": "length"},
                            )
                            for result in results
                        ]
                    )
                    serialized = _build_beam_search_out_dict(output, {}, None)
                    self.assertEqual(
                        [
                            beam["meta_info"]["sequence_score"]
                            for beam in serialized["beam_results"]
                        ],
                        [
                            result.cum_logprob
                            if enabled
                            else result.cum_logprob / depth
                            for result in results
                        ],
                    )

    def test_gr_retains_other_limits(self):
        for width, kwargs, message in (
            (257, {}, "too large for the vocabulary"),
            (256, {"pool_size": 128}, "req-to-token slots"),
            (16, {"page_size": 16}, "page-size 1"),
            (16, {"stop_token_ids": [0]}, "does not support"),
        ):
            with self.subTest(width=width, kwargs=kwargs):
                _, error = self.admit(True, width, **kwargs)
                self.assertIn(message, error)


class TestGRCandidateSelection(CustomTestCase):
    def test_full_vocabulary_oracle(self):
        generator = torch.Generator().manual_seed(20261012)
        for width, vocab in ((2, 4), (3, 4), (4, 4), (16, 256), (128, 256), (256, 256)):
            for rows in (1, width):
                for final in (False, True):
                    with self.subTest(width=width, rows=rows, final=final):
                        logits = torch.randn(rows, vocab, generator=generator)
                        cumulative = -torch.arange(rows, dtype=torch.float32) * 0.07
                        # Independent oracle ranks all parent-token extensions.
                        oracle = (
                            torch.log_softmax(logits, -1) + cumulative[:, None]
                        ).flatten()
                        scores, indices = oracle.topk(width)
                        row_scores, tokens = _rows_topk_logprobs(
                            [logits[:1], logits[1:]], width
                        )
                        if final:
                            selected = select_final_topk(
                                cumulative, row_scores, tokens, width
                            )
                            actual_tokens, actual_scores = (
                                selected.tokens,
                                selected.cum_logprobs,
                            )
                        else:
                            selected = joint_select(
                                cumulative,
                                row_scores,
                                tokens,
                                torch.empty(0, dtype=torch.int64),
                                width,
                            )
                            actual_tokens, actual_scores = (
                                selected.next_tokens,
                                selected.new_cum_logprobs,
                            )
                            self.assertEqual(int(selected.num_survivors), width)
                            self.assertEqual(int(selected.num_finished), 0)
                        self.assertEqual(
                            selected.parent_idx.tolist(), (indices // vocab).tolist()
                        )
                        self.assertEqual(
                            actual_tokens.tolist(), (indices % vocab).tolist()
                        )
                        torch.testing.assert_close(
                            actual_scores, scores, atol=1e-6, rtol=1e-6
                        )

    def test_eos_token_continues_until_fixed_length(self):
        group = BeamGroup(beam_width=2, num_candidates=2, max_new_tokens=3)
        for step in range(3):
            rows = len(group.leaves)
            # Token zero is the model EOS, but is a normal token under GR.
            logits = torch.tensor([[5.0, 1.0, -2.0, -3.0]]).repeat(rows, 1)
            scores, tokens = _rows_topk_logprobs([logits], group.num_candidates)
            if step < 2:
                selected = joint_select(
                    group.frontier_cum_logprobs, scores, tokens, group.stop_token_ids, 2
                )
                self.assertFalse(group.advance(selected))
            else:
                self.assertTrue(
                    group.advance_final(
                        select_final_topk(
                            group.frontier_cum_logprobs, scores, tokens, 2
                        )
                    )
                )
        results = group.finalize()
        self.assertEqual(results[0].tokens, [0, 0, 0])
        self.assertTrue(all(len(result.tokens) == 3 for result in results))
        self.assertTrue(all(result.matched_token is None for result in results))

    def test_candidate_count_contract(self):
        self.assertEqual(BeamGroup(beam_width=4, max_new_tokens=3).num_candidates, 8)
        with self.assertRaisesRegex(ValueError, "at least beam_width"):
            BeamGroup(beam_width=4, num_candidates=3, max_new_tokens=3)


if __name__ == "__main__":
    unittest.main()
