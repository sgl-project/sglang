import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestHiddenStateServerMode(CustomTestCase):
    def test_native_request_fixture_matches_python_modes_and_validation(self):
        fixtures = json.loads(
            (
                Path(__file__).resolve().parents[4]
                / "rust/sglang-server/testdata/hidden_states_python.json"
            ).read_text()
        )
        for maximum in (None, "last", "full"):
            manager = self._make_tokenizer_manager(maximum)
            for fixture in fixtures["requests"]:
                request = GenerateReqInput(**fixture["body"])
                request.normalize_batch_and_arguments()
                prompts = (
                    [request]
                    if request.is_single
                    else [request[i] for i in range(request.batch_size)]
                )
                self.assertEqual(
                    [
                        prompt.return_hidden_states
                        for prompt in prompts
                        for _ in range(request.parallel_sample_num)
                    ],
                    fixture["modes"],
                )
                for prompt in prompts:
                    supported = (
                        prompt.return_hidden_states is False
                        or maximum == "full"
                        or (maximum == "last" and prompt.return_hidden_states == "last")
                    )
                    if supported:
                        manager._validate_one_request(prompt, prompt.input_ids)
                    else:
                        with self.assertRaisesRegex(ValueError, "return-hidden-states"):
                            manager._validate_one_request(prompt, prompt.input_ids)
        for invalid in fixtures["invalid_modes"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                request = GenerateReqInput(input_ids=[1], return_hidden_states=invalid)
                request.normalize_batch_and_arguments()
                manager._validate_one_request(request, request.input_ids)

    def _make_tokenizer_manager(self, mode):
        # The server-side hidden-state mode is a bag leaf.
        override = get_context().override_server_args(
            enable_return_hidden_states=mode is not None,
            return_hidden_states_mode=mode,
        )
        override.install()
        self.addCleanup(override.restore)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.context_len = 128
        manager.num_reserved_tokens = 0
        manager.allow_auto_truncate = False
        manager.validate_total_tokens = False
        manager.is_generation = True
        manager.server_args = SimpleNamespace(enable_custom_logit_processor=False)
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
