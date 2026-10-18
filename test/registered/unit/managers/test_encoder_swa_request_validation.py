"""Request boundaries for encoder SWA replay's CUDA image support."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context, override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEncoderSwaRequestValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_encoder_swa_bounded_replay=True,
        )
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 128
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = False
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(joint_head_config=None)
        self.manager.server_args = SimpleNamespace(enable_custom_logit_processor=False)
        self.manager._validate_token_ids_logprob = Mock()

    def _validate(self, **kwargs):
        req = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={}, **kwargs)
        self.manager._validate_one_request(req, req.input_ids)

    @override_platform(is_cuda=True, is_hip=False)
    def test_cuda_accepts_text_and_images(self):
        self._validate()
        self._validate(image_data=["image.png"])

    @override_platform(is_cuda=False, is_hip=True)
    def test_rocm_still_rejects_images(self):
        self._validate()
        with self.assertRaisesRegex(ValueError, "image requests only on CUDA"):
            self._validate(image_data=["image.png"])

    @override_platform(is_cuda=True, is_hip=False)
    def test_unsupported_inputs_and_cached_logprobs_remain_rejected(self):
        for field in ("video_data", "audio_data", "input_embeds"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "does not support"),
            ):
                self._validate(**{field: [1]})
        with self.assertRaisesRegex(ValueError, "cached prompt logprobs"):
            self._validate(return_logprob=True, logprob_start_len=0)


if __name__ == "__main__":
    unittest.main()
