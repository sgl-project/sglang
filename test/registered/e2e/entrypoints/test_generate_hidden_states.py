"""HTTP `/generate` returns hidden states when the server allows the mode:
one vector for `return_hidden_states="last"`."""

import json
import unittest

import requests
from transformers import AutoConfig

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")

PROMPT = "The capital of France is"


class HiddenStatesServerCase(CustomTestCase):
    """One server per class, launched with `server_mode` and `env`."""

    server_mode = "last"
    env = {"SGLANG_RUST_SERVER": "0"}

    @classmethod
    def setUpClass(cls):
        cls.hidden_size = AutoConfig.from_pretrained(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        ).hidden_size
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--return-hidden-states-mode", cls.server_mode],
            env=cls.env,
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)

    def _post(self, text, return_hidden_states, stream=False, status_code=200):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                "return_hidden_states": return_hidden_states,
                "stream": stream,
            },
            stream=stream,
            timeout=60,
        )
        self.assertEqual(response.status_code, status_code, response.text)
        return response

    def _stream_chunks(self, text, return_hidden_states):
        response = self._post(text, return_hidden_states, stream=True)
        chunks = [
            json.loads(line[len(b"data: ") :])
            for line in response.iter_lines()
            if line.startswith(b"data: ") and line != b"data: [DONE]"
        ]
        self.assertGreater(len(chunks), 0)
        return chunks

    def assert_vector(self, vector):
        self.assertEqual(len(vector), self.hidden_size)
        self.assertTrue(all(isinstance(value, float) for value in vector))


class TestGenerateHiddenStatesLast(HiddenStatesServerCase):
    def assert_one_vector(self, output):
        self.assert_vector(output["meta_info"]["hidden_states"])

    def test_last_returns_one_vector(self):
        self.assert_one_vector(self._post(PROMPT, "last").json())

    def test_last_is_selected_per_batch_item(self):
        with_last, without = self._post(
            [PROMPT, "Hello, my name is"], ["last", False]
        ).json()
        self.assert_one_vector(with_last)
        self.assertNotIn("hidden_states", without["meta_info"])

    def test_one_item_list_for_one_prompt(self):
        self.assert_one_vector(self._post(PROMPT, ["last"]).json())
        without = self._post(PROMPT, [False]).json()
        self.assertNotIn("hidden_states", without["meta_info"])

    def test_list_of_wrong_length_is_rejected(self):
        self._post(PROMPT, ["last", "last"], status_code=400)
        self._post(PROMPT, [], status_code=400)
        # The server still serves after the rejected requests.
        self.assert_one_vector(self._post(PROMPT, "last").json())

    def test_unknown_string_is_rejected_before_a_stream_starts(self):
        self._post(PROMPT, "full", status_code=400)
        self._post(PROMPT, "full", stream=True, status_code=400)
        # Not read as a bool.
        self._post(PROMPT, "true", status_code=400)

    def test_last_streams_one_vector_per_chunk(self):
        for chunk in self._stream_chunks(PROMPT, "last"):
            self.assert_one_vector(chunk)


class TestGenerateHiddenStatesLastOnFullServer(TestGenerateHiddenStatesLast):
    server_mode = "full"


if __name__ == "__main__":
    unittest.main()
