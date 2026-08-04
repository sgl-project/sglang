import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiddenStateServerMode(CustomTestCase):
    @staticmethod
    def _make_tokenizer_manager(mode):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = 128
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = True
        manager.server_args = SimpleNamespace(
            enable_return_hidden_states=mode is not None,
            return_hidden_states_mode=mode,
            enable_custom_logit_processor=False,
        )
        manager._validate_token_ids_logprob = Mock()
        return manager

    @staticmethod
    def _make_request(return_hidden_states):
        return GenerateReqInput(
            input_ids=[1, 2, 3],
            sampling_params={},
            return_hidden_states=return_hidden_states,
        )

    def test_last_server_accepts_false_and_last(self):
        manager = self._make_tokenizer_manager("last")

        for mode in (False, "last"):
            with self.subTest(mode=mode):
                manager._validate_one_request(
                    self._make_request(mode),
                    [1, 2, 3],
                )

    def test_last_server_rejects_full(self):
        manager = self._make_tokenizer_manager("last")

        with self.assertRaisesRegex(
            ValueError,
            "server maximum `last`",
        ):
            manager._validate_one_request(
                self._make_request(True),
                [1, 2, 3],
            )

    def test_full_server_accepts_all_request_modes(self):
        manager = self._make_tokenizer_manager("full")

        for mode in (False, "last", True):
            with self.subTest(mode=mode):
                manager._validate_one_request(
                    self._make_request(mode),
                    [1, 2, 3],
                )


if __name__ == "__main__":
    unittest.main()
