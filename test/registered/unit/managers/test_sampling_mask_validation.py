import asyncio
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    BatchTokenizedGenerateReqInput,
    GenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.observability.req_time_stats import APIServerReqTimeStats
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.custom_logit_processor import (
    DisallowedTokensLogitsProcessor,
    Qwen3ThinkingBudgetLogitProcessor,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestSamplingMaskValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_custom_logit_processor=True
        )
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 128
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = False
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )

    def _validate(
        self,
        processor,
        return_sampling_mask=True,
        return_sampling_logprobs=False,
        return_logprob=False,
        return_flat_raw_sampling_mask=False,
    ):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={"top_k": 10},
            custom_logit_processor=processor,
            return_logprob=return_logprob,
            return_sampling_mask=return_sampling_mask,
            return_sampling_logprobs=return_sampling_logprobs,
            return_flat_raw_sampling_mask=return_flat_raw_sampling_mask,
        )
        self.manager._validate_one_request(req, req.input_ids)

    def test_top_logprobs_num_must_be_a_valid_count_when_logprobs_requested(self):
        """A malformed or oversized candidate count must not reach the scheduler."""
        self.manager.model_config.vocab_size = 32
        for top_logprobs_num, return_logprob, accepted in (
            (None, True, True),
            (0, True, True),
            (32, True, True),
            (33, True, False),
            (-3, True, False),
            ([33], True, False),
            (33, False, True),
        ):
            with self.subTest(top_logprobs_num=top_logprobs_num, rl=return_logprob):
                req = GenerateReqInput(
                    input_ids=[1, 2, 3],
                    sampling_params={},
                    return_logprob=return_logprob,
                    top_logprobs_num=top_logprobs_num,
                )
                if accepted:
                    self.manager._validate_one_request(req, req.input_ids)
                else:
                    with self.assertRaisesRegex(ValueError, "top_logprobs"):
                        self.manager._validate_one_request(req, req.input_ids)

    def test_token_indices_to_pool_must_be_prompt_positions(self):
        prompt_only = {"max_new_tokens": 0}
        for indices, sampling_params, accepted in (
            ([0, 2], prompt_only, True),
            ([], prompt_only, False),
            ([3], prompt_only, False),
            ([0], {}, False),
        ):
            with self.subTest(indices=indices, sampling_params=sampling_params):
                req = GenerateReqInput(
                    input_ids=[1, 2, 3],
                    sampling_params=sampling_params,
                    token_indices_to_pool=indices,
                )
                if accepted:
                    self.manager._validate_one_request(req, req.input_ids)
                else:
                    with self.assertRaisesRegex(ValueError, "token_indices_to_pool"):
                        self.manager._validate_one_request(req, req.input_ids)

    def test_logprob_start_len_must_be_at_least_minus_one(self):
        for logprob_start_len, return_logprob, accepted in (
            (None, True, True),
            (-1, True, True),
            (0, True, True),
            (2, True, True),
            (-2, True, False),
            (-5, True, False),
            ([0], True, False),
            (-5, False, True),
        ):
            with self.subTest(logprob_start_len=logprob_start_len, rl=return_logprob):
                req = GenerateReqInput(
                    input_ids=[1, 2, 3],
                    sampling_params={},
                    return_logprob=return_logprob,
                    logprob_start_len=logprob_start_len,
                )
                if accepted:
                    self.manager._validate_one_request(req, req.input_ids)
                else:
                    with self.assertRaisesRegex(ValueError, "logprob_start_len"):
                        self.manager._validate_one_request(req, req.input_ids)

    def test_mixed_batch_forwards_zero_count_for_logprob_disabled_items(self):
        """Disabled items must not contribute their raw count to a shared topk.

        Each enabled item keeps its own count.
        """
        self.manager.model_config.vocab_size = 32
        self.manager.preferred_sampling_params = None
        self.manager.sampling_params_class = SamplingParams
        self.manager.tokenizer = None
        self.manager.fake_bootstrap_room_counter = 0
        req = GenerateReqInput(
            input_ids=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
            sampling_params={},
            return_logprob=[True, False, False, True],
            top_logprobs_num=[5, 33, "many", 7],
        )
        req.normalize_batch_and_arguments()
        self.manager.rid_to_state = {
            rid: SimpleNamespace(time_stats=APIServerReqTimeStats()) for rid in req.rid
        }
        forwarded_counts = []
        for i in range(4):
            self.manager._validate_one_request(req[i], req[i].input_ids)
            tokenized_request = self.manager._create_tokenized_object(
                req[i], "", req[i].input_ids
            )
            forwarded_counts.append(tokenized_request.top_logprobs_num)
        self.assertEqual(forwarded_counts, [5, 0, 0, 7])

    def test_accepts_hard_exclusion(self):
        class BoundMask(DisallowedTokensLogitsProcessor):
            def __call__(self, logits, custom_param_list=None):
                logits[..., [2]] = -float("inf")
                return logits

        for processor in (
            DisallowedTokensLogitsProcessor.to_str(),
            BoundMask.to_str(),
        ):
            with self.subTest(processor=processor):
                self._validate(processor)

    def test_rejects_unsupported_or_malformed_processors(self):
        for processor in (
            Qwen3ThinkingBudgetLogitProcessor.to_str(),
            "invalid processor",
            '{"callable": "00"}',
        ):
            with self.subTest(processor=processor):
                with self.assertRaisesRegex(
                    ValueError, "only supports DisallowedTokensLogitsProcessor"
                ):
                    self._validate(processor)

    def test_other_processors_still_work_without_sampling_masks(self):
        self._validate(
            Qwen3ThinkingBudgetLogitProcessor.to_str(), return_sampling_mask=False
        )

    def test_sampling_logprobs_need_no_mask_but_need_output_logprobs(self):
        self._validate(
            None,
            return_sampling_mask=False,
            return_sampling_logprobs=True,
            return_logprob=True,
        )
        with self.assertRaisesRegex(
            ValueError, "return_sampling_logprobs requires return_logprob"
        ):
            self._validate(None, return_sampling_logprobs=True)
        with self.assertRaisesRegex(
            ValueError, "only supports DisallowedTokensLogitsProcessor"
        ):
            self._validate(
                Qwen3ThinkingBudgetLogitProcessor.to_str(),
                return_sampling_mask=False,
                return_sampling_logprobs=True,
                return_logprob=True,
            )

    def _tokenize(self, sampling_params, *, watermarked=False, **request_fields):
        self.manager.model_config.vocab_size = 32
        self.manager.preferred_sampling_params = None
        self.manager.sampling_params_class = SamplingParams
        # min_new_tokens needs a tokenizer; nothing here reads it.
        self.manager.tokenizer = SimpleNamespace()
        self.manager.fake_bootstrap_room_counter = 0
        req = GenerateReqInput(
            input_ids=[1, 2, 3], sampling_params=sampling_params, **request_fields
        )
        req.normalize_batch_and_arguments()
        self.manager.rid_to_state = {
            req.rid: SimpleNamespace(time_stats=APIServerReqTimeStats())
        }
        with patch(
            "sglang.srt.managers.tokenizer_manager._validate_watermark_request",
            return_value=watermarked,
        ):
            return self.manager._create_tokenized_object(req, "", req.input_ids)

    def test_unfiltered_sampling_output_must_describe_what_was_sampled(self):
        """An unfiltered request is not captured: it returns its output logprobs
        as sampling logprobs and empty mask rows. A watermark forces the token, so
        both are wrong; SGLANG_RETURN_ORIGINAL_LOGPROB only changes the logprob.
        A filtered request is captured from the sampler and stays admitted."""
        logprobs = {"return_logprob": True, "return_sampling_logprobs": True}
        mask = {"return_sampling_mask": True}
        for watermarked, original_logprob, sampling_params, flags, rejected in (
            (True, False, {}, logprobs, True),
            (True, False, {}, mask, True),
            (True, False, {"top_k": 10}, {**logprobs, **mask}, False),
            (False, True, {}, logprobs, True),
            (False, True, {}, mask, False),
            (False, True, {"top_k": 10}, {**logprobs, **mask}, False),
        ):
            with (
                self.subTest(
                    watermarked=watermarked, params=sampling_params, flags=flags
                ),
                envs.SGLANG_RETURN_ORIGINAL_LOGPROB.override(original_logprob),
            ):
                if rejected:
                    with self.assertRaisesRegex(
                        ValueError, "without top-k/top-p/min-p"
                    ):
                        self._tokenize(
                            sampling_params, watermarked=watermarked, **flags
                        )
                else:
                    self._tokenize(sampling_params, watermarked=watermarked, **flags)

    def test_unfiltered_sampling_output_is_refused_where_mlx_commits_argmax(self):
        """Without --mlx-enable-sampling the MLX runner commits argmax whatever the
        temperature is, so an unfiltered request's output logprobs and empty mask
        rows would describe a draw that never happened."""
        logprobs = {"return_logprob": True, "return_sampling_logprobs": True}
        mask = {"return_sampling_mask": True}
        for mlx, mlx_sampling, sampling_params, rejected in (
            (True, False, {"temperature": 0.7}, True),
            (True, True, {"temperature": 0.7}, False),
            (False, False, {"temperature": 0.7}, False),
            (True, False, {"top_k": 10}, False),
        ):
            for flags in (logprobs, mask):
                with (
                    self.subTest(
                        mlx=mlx, mlx_sampling=mlx_sampling, params=sampling_params
                    ),
                    patch(
                        "sglang.srt.managers.tokenizer_manager.use_mlx",
                        return_value=mlx,
                    ),
                    get_context().override_server_args(
                        mlx_enable_sampling=mlx_sampling
                    ),
                ):
                    if rejected:
                        with self.assertRaisesRegex(ValueError, "mlx-enable-sampling"):
                            self._tokenize(sampling_params, **flags)
                    else:
                        self._tokenize(sampling_params, **flags)

    def test_unfiltered_mask_is_refused_when_the_support_is_narrowed(self):
        """An unfiltered request's empty mask rows stand for the whole vocabulary.
        A grammar, a custom logit processor, logit_bias, min_new_tokens or strict
        thinking leaves the sampler fewer tokens, so the mask must be refused. Its
        sampling logprobs stay exact, and a finite top_k captures the real rows."""
        disallowed = {
            "custom_logit_processor": DisallowedTokensLogitsProcessor.to_str()
        }
        for sampling_params, request_fields, server_args in (
            ({"json_schema": '{"type": "object"}'}, {}, {}),
            ({"regex": "a+"}, {}, {}),
            ({"ebnf": 'root ::= "a"'}, {}, {}),
            ({"structural_tag": "{}"}, {}, {}),
            ({}, disallowed, {}),
            ({"logit_bias": {"3": -100.0}}, {}, {}),
            ({"min_new_tokens": 2}, {}, {}),
            ({}, {}, {"enable_strict_thinking": True}),
        ):
            with (
                self.subTest(
                    params=sampling_params, fields=request_fields, server=server_args
                ),
                get_context().override_server_args(**server_args),
            ):
                with self.assertRaisesRegex(ValueError, "Set a finite top_k"):
                    self._tokenize(
                        sampling_params, return_sampling_mask=True, **request_fields
                    )
                self._tokenize(
                    sampling_params,
                    return_logprob=True,
                    return_sampling_logprobs=True,
                    **request_fields,
                )
                self._tokenize(
                    {**sampling_params, "top_k": 10},
                    return_sampling_mask=True,
                    **request_fields,
                )
        self._tokenize({"temperature": 0.7}, return_sampling_mask=True)

    def test_dispatch_records_whether_sampling_is_filtered(self):
        """The response reads a request's sampling output by whether its sampling
        is filtered. A parallel-sampling copy gets a fresh state without being
        tokenized again, so the value must reach the state at dispatch, on the
        single and the batched path, and a request body cannot supply it."""
        sent = []
        self.manager.cuda_vmm_feature_transport = SimpleNamespace(
            prepare_for_dispatch_async=AsyncMock(return_value=[]),
            cancel_for_dispatch=lambda items: None,
        )
        self.manager.positional_embeds_transport = SimpleNamespace(
            prepare_for_dispatch=lambda tokenized_objs: [],
            cancel_for_dispatch=lambda proxies: None,
        )
        self.manager.canary_info_hook = None
        self.manager.encoder_dispatch_ready = {}
        self.manager.batch_tokenized_generate_req_type = BatchTokenizedGenerateReqInput
        self.manager._dispatch_to_scheduler = sent.append

        def fresh_state(tokenized):
            state = ReqState([], False, asyncio.Event(), None, tokenized.time_stats)
            self.manager.rid_to_state = {tokenized.rid: state}
            return state

        for sampling_params, filtered in (({"top_k": 10}, True), ({}, False)):
            with self.subTest(sampling_params=sampling_params):
                tokenized = self._tokenize(sampling_params, return_sampling_mask=True)
                sample = copy.copy(tokenized)
                sample.rid = f"{tokenized.rid}-sample"
                state = fresh_state(sample)
                asyncio.run(self.manager._send_one_request(sample))
                self.assertIs(state.filters_sampling, filtered)

                state = fresh_state(tokenized)
                asyncio.run(self.manager._send_batch_request([tokenized]))
                self.assertIs(state.filters_sampling, filtered)
        self.assertEqual(len(sent), 4)
        with self.assertRaises(TypeError):
            GenerateReqInput(input_ids=[1], filters_sampling=True)

    def test_removed_sampling_logprobs_mode_is_rejected(self):
        for mode in ("selected", "support", [None, None]):
            with self.subTest(mode=mode):
                req = GenerateReqInput(
                    text=["a", "b"],
                    return_sampling_mask=True,
                    sampling_logprobs_mode=mode,
                )
                with self.assertRaisesRegex(
                    ValueError, "sampling_logprobs_mode was removed"
                ):
                    req.normalize_batch_and_arguments()

    def test_flat_raw_sampling_mask_requires_sampling_mask(self):
        self._validate(None, return_flat_raw_sampling_mask=True)
        with self.assertRaisesRegex(
            ValueError, "return_flat_raw_sampling_mask can only be set when"
        ):
            self._validate(
                None, return_sampling_mask=False, return_flat_raw_sampling_mask=True
            )


if __name__ == "__main__":
    unittest.main()
