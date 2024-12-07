import json
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


def setup_class(cls, backend: str, disable_overlap: bool):
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST
    cls.ebnf_grammar = r"""
    root ::= "Hello" " " name ("," " my name is " name)? "." | "Goodbye" "."
    name ::= [A-Za-z]+
    """

    other_args = [
        "--max-running-requests",
        "10",
        "--grammar-backend",
        backend,
    ]

    if disable_overlap:
        other_args += ["--disable-overlap-schedule"]

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )


class TestEBNFConstrainedOutlinesBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="outlines", disable_overlap=False)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(
        self, ebnf, expected_patterns, prompt, temperature=0.5, max_new_tokens=50, n=1
    ):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "n": n,
                    "ebnf": ebnf,
                },
                "stream": False,
                "return_logprob": False,
                "top_logprobs_num": 0,
                "logprob_start_len": 0,
            },
        )

        ret = response.json()
        print(json.dumps(ret, indent=2))

        if not isinstance(ret, list):
            self.fail(f"Expected response to be a list, but got {type(ret)}")

        for item in ret:
            text = item.get("text", "").strip()
            if not text:
                self.fail("Generated text is empty.")

            match = False
            for pattern in expected_patterns:
                if self.regex_match(text, pattern):
                    match = True
                    break
            if not match:
                self.fail(f"Text '{text}' does not match any of the allowed patterns.")

    def regex_match(self, text, pattern):
        import re

        return re.match(pattern, text) is not None

    def test_ebnf_generate(self):
        allowed_patterns = [
            r"^Hello [A-Za-z]+\.$",
            r"^Hello [A-Za-z]+, my name is [A-Za-z]+\.$",
            r"^Goodbye\.$",
        ]
        prompt = "Generate a greeting or farewell:"

        self.run_decode(
            ebnf=self.__class__.ebnf_grammar,
            expected_patterns=allowed_patterns,
            prompt=prompt,
            temperature=0.5,
            max_new_tokens=20,
            n=3,
        )


class TestEBNFConstrainedXGrammarBackend(TestEBNFConstrainedOutlinesBackend):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar", disable_overlap=False)


if __name__ == "__main__":
    unittest.main()
