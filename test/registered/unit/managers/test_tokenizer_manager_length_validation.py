"""Unit tests for TokenizerManager request length validation.

`max_new_tokens` is an output upper bound, not a reservation of the context
window. A request whose prompt leaves less room than the cap asks for must be
accepted and clamped by the scheduler; only a cap larger than the whole window
is rejected. A prompt that does not fit still fails, so every accepted request
has room for at least one output token.
"""

import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import GenerateReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRequestLengthValidation(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.context_len = 100
        self.manager.num_reserved_tokens = 0
        self.manager.max_req_input_len = None
        self.manager.allow_auto_truncate = False
        self.manager.validate_total_tokens = True
        self.manager.is_generation = True
        self.manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )

    def _validate(
        self,
        input_len,
        max_new_tokens,
        *,
        context_len=100,
        num_reserved_tokens=0,
        allow_auto_truncate=False,
    ):
        self.manager.context_len = context_len
        self.manager.num_reserved_tokens = num_reserved_tokens
        self.manager.allow_auto_truncate = allow_auto_truncate
        sampling_params = (
            {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
        )
        req = GenerateReqInput(
            input_ids=[1] * input_len, sampling_params=sampling_params
        )
        req.normalize_batch_and_arguments()
        obj = req[0]
        self.manager._validate_one_request(obj, obj.input_ids)
        return obj

    def test_cap_beyond_remaining_room_is_left_to_the_scheduler(self):
        """A cap over the prompt's remaining room is not a reservation."""
        obj = self._validate(input_len=90, max_new_tokens=20)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 20, "not clamped")
        self.assertEqual(len(obj.input_ids), 90, "prompt untouched")

    def test_cap_equal_to_the_whole_context_is_allowed(self):
        obj = self._validate(input_len=10, max_new_tokens=100)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 100)

    def test_cap_above_the_whole_context_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"max_new_tokens is too large: 101"):
            self._validate(input_len=10, max_new_tokens=101)

    def test_cap_above_the_whole_context_is_rejected_with_auto_truncate(self):
        """The cap guard must not be bypassed by --allow-auto-truncate."""
        with self.assertRaisesRegex(ValueError, r"max_new_tokens is too large: 101"):
            self._validate(input_len=10, max_new_tokens=101, allow_auto_truncate=True)

    def test_prompt_without_room_for_an_output_token_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, r"100 tokens.*longer than the model's context"
        ):
            self._validate(input_len=100, max_new_tokens=1)

    def test_prompt_without_room_is_rejected_without_a_cap(self):
        with self.assertRaisesRegex(
            ValueError, r"100 tokens.*longer than the model's context"
        ):
            self._validate(input_len=100, max_new_tokens=None)

    def test_prompt_one_token_short_of_the_window_is_allowed(self):
        obj = self._validate(input_len=99, max_new_tokens=50)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 50)

    def test_reserved_tokens_count_toward_the_prompt(self):
        # 90 prompt + 5 reserved = 95, leaving five slots; 20 is still only a cap.
        obj = self._validate(input_len=90, max_new_tokens=20, num_reserved_tokens=5)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 20)

        with self.assertRaisesRegex(ValueError, r"longer than the model's context"):
            self._validate(input_len=95, max_new_tokens=1, num_reserved_tokens=5)

    def test_auto_truncate_still_clamps_the_cap(self):
        """--allow-auto-truncate keeps its explicit frontend cap clamp."""
        obj = self._validate(input_len=90, max_new_tokens=20, allow_auto_truncate=True)
        self.assertEqual(obj.sampling_params["max_new_tokens"], 10)

    def test_auto_truncate_still_truncates_the_prompt(self):
        obj = self._validate(input_len=110, max_new_tokens=20, allow_auto_truncate=True)
        self.assertEqual(len(obj.input_ids), 100)
        # The truncated prompt fills the window, so no completion budget remains.
        self.assertEqual(obj.sampling_params["max_new_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
