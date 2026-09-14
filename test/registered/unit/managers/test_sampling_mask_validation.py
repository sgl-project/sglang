import unittest
from unittest.mock import patch

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.custom_logit_processor import (
    CustomLogitProcessor,
    DisallowedTokensLogitsProcessor,
    Qwen3ThinkingBudgetLogitProcessor,
)
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

    def _validate(self, processor, return_sampling_mask=True):
        req = GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={"top_k": 10},
            custom_logit_processor=processor,
            return_sampling_mask=return_sampling_mask,
        )
        self.manager._validate_one_request(req, req.input_ids)

    def test_accepts_hard_exclusion(self):
        class BoundMask(DisallowedTokensLogitsProcessor):
            def __call__(self, logits, custom_param_list=None):
                logits[..., [2]] = -float("inf")
                return logits

        for processor in (
            None,
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

    def test_disabled_processors_are_not_deserialized(self):
        with (
            get_context().override_server_args(enable_custom_logit_processor=False),
            patch.object(CustomLogitProcessor, "from_str") as deserialize,
        ):
            with self.assertRaisesRegex(ValueError, "--enable-custom-logit-processor"):
                self._validate(DisallowedTokensLogitsProcessor.to_str())
            deserialize.assert_not_called()

    def test_other_processors_still_work_without_sampling_masks(self):
        self._validate(
            Qwen3ThinkingBudgetLogitProcessor.to_str(), return_sampling_mask=False
        )


if __name__ == "__main__":
    unittest.main()
