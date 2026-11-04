import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTokenizerManagerModelTypeValidation(CustomTestCase):
    def _make_tokenizer_manager(self, is_generation):
        override = get_context().override_server_args(
            enable_custom_logit_processor=False
        )
        override.install()
        self.addCleanup(override.restore)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = 128
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = is_generation
        manager.model_config = SimpleNamespace(joint_head_config=None)
        manager._validate_token_ids_logprob = Mock()
        return manager

    def test_generation_request_on_embedding_server_is_rejected(self):
        """A generation request on an embedding server reached the model and
        failed an assertion in its forward, which took the scheduler down."""
        manager = self._make_tokenizer_manager(is_generation=False)
        request = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={})

        with self.assertRaisesRegex(ValueError, "cannot generate"):
            manager._validate_one_request(request, [1, 2, 3])

    def test_embedding_request_on_generation_server_is_rejected(self):
        manager = self._make_tokenizer_manager(is_generation=True)
        request = EmbeddingReqInput(input_ids=[1, 2, 3], sampling_params={})

        with self.assertRaisesRegex(ValueError, "--is-embedding"):
            manager._validate_one_request(request, [1, 2, 3])

    def test_matching_request_types_pass(self):
        for is_generation, request in (
            (True, GenerateReqInput(input_ids=[1, 2, 3], sampling_params={})),
            (False, EmbeddingReqInput(input_ids=[1, 2, 3], sampling_params={})),
        ):
            with self.subTest(is_generation=is_generation):
                self._make_tokenizer_manager(is_generation)._validate_one_request(
                    request, [1, 2, 3]
                )


if __name__ == "__main__":
    unittest.main()
