"""Unit tests for /generate request validation in TokenizerManager (issue #1266)."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_VOCAB_SIZE = 100


class TestGenerateRequestValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 128
        self.manager.num_reserved_tokens = 0
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = True
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None, vocab_size=_VOCAB_SIZE
        )

    def _validate(self, req, input_ids=None):
        # Mirror the HTTP path: the request is normalized before validation.
        req.normalize_batch_and_arguments()
        if input_ids is None:
            input_ids = req.input_ids
        self.manager._validate_one_request(req, input_ids)

    def test_accepts_input_ids_at_vocab_boundaries(self):
        self._validate(GenerateReqInput(input_ids=[0, _VOCAB_SIZE - 1]))

    def test_rejects_out_of_vocab_input_ids(self):
        for input_ids, bad_id in (
            ([1000000000, -5], -5),
            ([1, _VOCAB_SIZE], _VOCAB_SIZE),
            ([-1, 2], -1),
        ):
            with self.subTest(input_ids=input_ids):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"input_ids contains out-of-vocabulary token id {bad_id}; "
                    rf"valid range is \[0, {_VOCAB_SIZE}\)",
                ):
                    self._validate(GenerateReqInput(input_ids=input_ids))

    def test_rejects_non_integer_input_ids(self):
        # A non-integer between integer extremes must not slip past a
        # min/max-only check.
        for input_ids in ([1, "2"], [1, 2.5, 3], [1, None], [1, True, 3]):
            with self.subTest(input_ids=input_ids):
                with self.assertRaisesRegex(ValueError, "input_ids must be integers"):
                    self._validate(GenerateReqInput(input_ids=input_ids))

    def test_text_prompt_ids_are_not_checked_against_vocab(self):
        # Ids produced by the tokenizer or a multimodal processor are trusted;
        # only client-supplied input_ids are range-checked.
        self._validate(GenerateReqInput(text="hi"), input_ids=[_VOCAB_SIZE + 5])

    def test_rejects_non_integer_max_new_tokens(self):
        for value in ("ten", 1.5, [4]):
            with self.subTest(max_new_tokens=value):
                req = GenerateReqInput(
                    text="hi", sampling_params={"max_new_tokens": value}
                )
                with self.assertRaisesRegex(
                    ValueError, "max_new_tokens must be an integer"
                ):
                    self._validate(req, input_ids=[1, 2])

    def test_accepts_integer_max_new_tokens(self):
        req = GenerateReqInput(text="hi", sampling_params={"max_new_tokens": 8})
        self._validate(req, input_ids=[1, 2])


if __name__ == "__main__":
    unittest.main()
