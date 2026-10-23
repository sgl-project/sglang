import unittest
from types import SimpleNamespace

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CONTEXT_LEN = 128
_RESERVED_TOKENS = 2


class TestMaxNewTokensContextClamp(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)

    def _manager(self, *, clamp: bool) -> TokenizerManager:
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = _CONTEXT_LEN
        manager.num_reserved_tokens = _RESERVED_TOKENS
        manager.allow_auto_truncate = False
        manager.clamp_max_new_tokens_to_context = clamp
        manager.validate_total_tokens = True
        manager.is_generation = True
        manager.model_config = SimpleNamespace(
            joint_head_config=None, pointer_head=None
        )
        return manager

    def _request(self, num_input_tokens: int, max_new_tokens: int):
        return GenerateReqInput(
            input_ids=list(range(num_input_tokens)),
            sampling_params={"max_new_tokens": max_new_tokens},
        )

    def test_clamp_caps_max_new_tokens_to_the_remaining_context(self):
        """With the flag, a prompt that fits gets the rest of the window (reserved
        tokens included) instead of a 400; a cap that already fits is untouched."""
        manager = self._manager(clamp=True)
        req = self._request(num_input_tokens=100, max_new_tokens=1000)
        manager._validate_one_request(req, req.input_ids)
        self.assertEqual(
            req.sampling_params["max_new_tokens"],
            _CONTEXT_LEN - 100 - _RESERVED_TOKENS,
        )

        req = self._request(num_input_tokens=100, max_new_tokens=10)
        manager._validate_one_request(req, req.input_ids)
        self.assertEqual(req.sampling_params["max_new_tokens"], 10)

    def test_default_still_rejects_prompt_plus_max_new_tokens_over_context(self):
        """Without the flag the OpenAI-compatible 400 is unchanged."""
        req = self._request(num_input_tokens=100, max_new_tokens=1000)
        with self.assertRaisesRegex(ValueError, "maximum context length"):
            self._manager(clamp=False)._validate_one_request(req, req.input_ids)

    def test_clamp_does_not_accept_a_prompt_longer_than_the_context(self):
        """The flag only relaxes the output cap; an overlong prompt is still
        rejected rather than truncated."""
        req = self._request(num_input_tokens=_CONTEXT_LEN, max_new_tokens=10)
        with self.assertRaisesRegex(ValueError, "longer than the model's context"):
            self._manager(clamp=True)._validate_one_request(req, req.input_ids)
        self.assertEqual(len(req.input_ids), _CONTEXT_LEN)


if __name__ == "__main__":
    unittest.main()
