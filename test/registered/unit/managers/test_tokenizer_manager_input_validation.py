import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTokenizerManagerInputValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(
            enable_custom_logit_processor=False
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
        self.manager._validate_token_ids_logprob = Mock()

    def test_prompt_without_tokens_is_rejected(self):
        """A prompt that tokenizes to nothing must fail validation; forwarding a
        zero-token request crashes the scheduler."""
        request = GenerateReqInput(text="", sampling_params={})

        with self.assertRaisesRegex(ValueError, "input is empty"):
            self.manager._validate_one_request(request, [])

    def test_embedding_without_tokens_is_rejected(self):
        """/v1/embeddings accepts a batch of token-id lists; an empty inner list
        reached the model and raised a CUDA device-side assert."""
        self.manager.is_generation = False
        request = EmbeddingReqInput(input_ids=[])

        with self.assertRaisesRegex(ValueError, "input is empty"):
            self.manager._validate_one_request(request, [])

    def test_embedding_empty_input_ids_is_rejected_before_batching(self):
        """/encode with input_ids=[] indexed input_ids[0] while deriving the
        batch size, an IndexError the server returned as a 500."""
        with self.assertRaisesRegex(ValueError, "input_ids cannot be empty"):
            EmbeddingReqInput(input_ids=[]).normalize_batch_and_arguments()

    def test_session_request_may_leave_prompt_to_session(self):
        request = GenerateReqInput(
            input_ids=[], sampling_params={}, session_params={"id": "s1"}
        )

        self.manager._validate_one_request(request, [])


if __name__ == "__main__":
    unittest.main()
