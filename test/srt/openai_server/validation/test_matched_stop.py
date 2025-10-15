import json
import unittest

import requests

from sglang.srt.sampling.sampling_params import MAX_LEN, get_max_seq_length
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MANY_NEW_TOKENS_PROMPT = """
Please write an extremely detailed and vivid fantasy story, set in a world full of intricate magic systems, political intrigue, and complex characters.
Ensure that you thoroughly describe every scene, character's motivations, and the environment. Include long, engaging dialogues and elaborate on the inner thoughts of the characters.
Each section should be as comprehensive as possible to create a rich and immersive experience for the reader.
The story should span multiple events, challenges, and character developments over time. Aim to make the story at least 3,000 words long.
"""


class TestMatchedStop(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=300,
            other_args=["--max-running-requests", "10"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_completions_generation(
        self,
        prompt=MANY_NEW_TOKENS_PROMPT,
        max_tokens=1,
        stop=None,
        stop_regex=None,
        finish_reason=None,
        matched_stop=None,
    ):
        payload = {
            "prompt": prompt,
            "model": self.model,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }

        if stop is not None:
            payload["stop"] = stop

        if stop_regex is not None:
            payload["stop_regex"] = stop_regex

        response_completions = requests.post(
            self.base_url + "/v1/completions",
            json=payload,
        )
        print(json.dumps(response_completions.json()))
        print("=" * 100)

        assert (
            response_completions.json()["choices"][0]["finish_reason"] == finish_reason
        )
        assert response_completions.json()["choices"][0]["matched_stop"] == matched_stop

    def run_chat_completions_generation(
        self,
        prompt=MANY_NEW_TOKENS_PROMPT,
        max_tokens=1,
        stop=None,
        stop_regex=None,
        finish_reason=None,
        matched_stop=None,
    ):
        chat_payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }

        if stop is not None:
            chat_payload["stop"] = stop

        if stop_regex is not None:
            chat_payload["stop_regex"] = stop_regex

        response_chat = requests.post(
            self.base_url + "/v1/chat/completions",
            json=chat_payload,
        )
        print(json.dumps(response_chat.json()))
        print("=" * 100)

        assert response_chat.json()["choices"][0]["finish_reason"] == finish_reason
        assert response_chat.json()["choices"][0]["matched_stop"] == matched_stop

    def test_finish_stop_str(self):
        self.run_completions_generation(
            max_tokens=1000, stop="\n", finish_reason="stop", matched_stop="\n"
        )
        self.run_chat_completions_generation(
            max_tokens=1000, stop="\n", finish_reason="stop", matched_stop="\n"
        )

    def test_finish_stop_regex_str(self):
        STOP_REGEX_STR = r"and|or"
        self.run_completions_generation(
            max_tokens=1000,
            stop_regex=STOP_REGEX_STR,
            finish_reason="stop",
            matched_stop=STOP_REGEX_STR,
        )
        self.run_chat_completions_generation(
            max_tokens=1000,
            stop_regex=STOP_REGEX_STR,
            finish_reason="stop",
            matched_stop=STOP_REGEX_STR,
        )

        # Match a complete sentence
        STOP_REGEX_STR_SENTENCE = r"[.!?]\s*$"
        self.run_chat_completions_generation(
            max_tokens=1000,
            stop_regex=STOP_REGEX_STR_SENTENCE,
            finish_reason="stop",
            matched_stop=STOP_REGEX_STR_SENTENCE,
        )

    def test_finish_stop_eos(self):
        llama_format_prompt = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

        What is 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        eos_token_id = 128009
        self.run_completions_generation(
            prompt=llama_format_prompt,
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_id,
        )
        self.run_chat_completions_generation(
            prompt="What is 2 + 2?",
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_id,
        )

    def test_finish_length(self):
        self.run_completions_generation(
            max_tokens=5, finish_reason="length", matched_stop=None
        )
        self.run_chat_completions_generation(
            max_tokens=5, finish_reason="length", matched_stop=None
        )


class TestRegexPatternMaxLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.regex_str_to_max_len = {
            "((ab|cd(e|f){2}){3,5}g|hij)*k": MAX_LEN,
            # - '*' → infinite tokens need to be stored
            "abc*?k": MAX_LEN,
            # - '*?' → infinite tokens still need to be stored even if lazy matching used
            "^spec(foo|at)$": 7,
            # - '^' and '$' don't add any characters to the max length
            # "spec" → 4
            # "(foo|at)" → max(3, 2) = 3
            # Whole regex = 7
            "(a(bca|de(fg|hi){2,3})j){2}kl": 22,
            # - Innermost alt: "fg" vs "hi" → 2
            # - Repeat {2,3}: max = 3 * 2 = 6
            # - Inner group "de(...)": 2 (for "de") + 6 = 8.
            # - "bca" or "de(...)" → max(3, 8) = 8
            # - Whole group: "a" (1) + group (8) + "j"(1) = 10
            # - Repeat {2} → 20
            # - Add "kl"(2) → 22
            "(foo(bar|baz(qux){1,2}))|(x(yz){5,10})": 21,
            # Branch 1:
            #   "foo"(3) + max("bar"(3), "baz"(3)+"qux"{2} = 3 + 6 = 9) = 3 + 9 = 12
            # Branch 2:
            #   "x"(1) + "yz"{10} = 1 + 20 =21
            # Whole regex = max(12, 21) = 21
            "(((a|bc){1,3}(d(e|f){2}|gh){2,4})|(ijk|lmp(no|p){3})){5}": 90,
            # Branch A:
            #   (a|bc){1,3} → max = 3 * 2 = 6
            #   Inside: d(e|f){2} = 1 + 2 * 1 = 3 vs gh = 2 → max = 3
            #   Repeat {2,4} → 4 * 3 = 12
            #   Branch A total = 18
            # Branch B:
            #   "ijk"(3) vs "lmp(no|p){3}" = 3 + 3 * max(2, 1) = 3 + 6 = 9 → max = 9
            #   Branch B total = 9
            # Whole outer alt = max(18, 9) = 18
            # Repeat {5} → 90
        }

    def test_get_max_length(self):
        for regex_str, max_len in self.regex_str_to_max_len.items():
            if max_len == MAX_LEN:
                self.assertGreaterEqual(get_max_seq_length(regex_str), MAX_LEN)
            else:
                self.assertEqual(get_max_seq_length(regex_str), max_len)


if __name__ == "__main__":
    unittest.main()
