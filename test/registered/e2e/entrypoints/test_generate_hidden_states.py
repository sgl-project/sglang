"""HTTP `/generate` returns hidden states when the server allows the mode:
one vector for `return_hidden_states="last"`, and for `True` a matrix for the
prefill step followed by one vector per decode step. A mode the server does not
allow is a 400. The Python frontend and the Rust server answer the same, on a
server launched with no mode, with `last` and with `full`."""

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
    is_rust_server_built,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")

PROMPT = "The capital of France is"


class HiddenStatesServerCase(CustomTestCase):
    """One server per class, launched with `server_mode` and `env`."""

    # None launches without `--return-hidden-states-mode`.
    server_mode = None
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
            other_args=(
                ["--return-hidden-states-mode", cls.server_mode]
                if cls.server_mode is not None
                else []
            ),
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


class LastCases:
    """`return_hidden_states="last"` on any server that allows it."""

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


class TestGenerateHiddenStatesUnset(HiddenStatesServerCase):
    def test_every_mode_is_refused(self):
        for mode in ("last", True):
            error = self._post(PROMPT, mode, status_code=400).json()["error"]
            self.assertIn("not configured to return hidden states", error["message"])
        # The server still serves after the refused requests.
        self.assertNotIn("hidden_states", self._post(PROMPT, False).json()["meta_info"])


class TestGenerateHiddenStatesLast(LastCases, HiddenStatesServerCase):
    server_mode = "last"


class TestGenerateHiddenStatesFull(LastCases, HiddenStatesServerCase):
    """`return_hidden_states=True`: a matrix with one row per prefill token,
    then one vector per decode step. `"last"` is served too."""

    server_mode = "full"
    long_prompt = "The quick brown fox jumps over the lazy dog near the quiet river"

    def assert_prefill_matrix_then_vectors(self, hidden_states):
        prefill, decode = hidden_states[0], hidden_states[1:]
        self.assertGreater(len(prefill), 0)
        for row in prefill:
            self.assertIsInstance(row, list)
            self.assert_vector(row)
        for vector in decode:
            self.assert_vector(vector)

    def test_prefill_item_has_one_row_per_token(self):
        # A prompt no other test sends, so most of it is not in the prefix cache.
        meta = self._post(f"Shape test one. {self.long_prompt}", True).json()[
            "meta_info"
        ]
        hidden_states = meta["hidden_states"]
        self.assert_prefill_matrix_then_vectors(hidden_states)
        self.assertGreater(len(hidden_states[0]), 1)
        self.assertEqual(
            len(hidden_states[0]), meta["prompt_tokens"] - meta["cached_tokens"]
        )
        self.assertEqual(len(hidden_states), meta["completion_tokens"])

    def test_cached_prompt_leaves_a_one_row_matrix(self):
        text = f"Shape test two. {self.long_prompt}"
        self._post(text, True)
        meta = self._post(text, True).json()["meta_info"]
        hidden_states = meta["hidden_states"]
        self.assertEqual(meta["prompt_tokens"] - meta["cached_tokens"], 1)
        self.assert_prefill_matrix_then_vectors(hidden_states)
        self.assertEqual(len(hidden_states[0]), 1)

    def test_stream_keeps_the_shape_in_every_chunk(self):
        for chunk in self._stream_chunks(f"Shape test three. {self.long_prompt}", True):
            hidden_states = chunk["meta_info"]["hidden_states"]
            self.assert_prefill_matrix_then_vectors(hidden_states)
            self.assertGreater(len(hidden_states[0]), 1)


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestGenerateHiddenStatesUnsetRustServer(TestGenerateHiddenStatesUnset):
    env = {"SGLANG_RUST_SERVER": "1"}


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestGenerateHiddenStatesLastRustServer(TestGenerateHiddenStatesLast):
    env = {"SGLANG_RUST_SERVER": "1"}


@unittest.skipUnless(is_rust_server_built(), "embedded rust server extension not built")
class TestGenerateHiddenStatesFullRustServer(TestGenerateHiddenStatesFull):
    env = {"SGLANG_RUST_SERVER": "1"}


if __name__ == "__main__":
    unittest.main()
