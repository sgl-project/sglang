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
    # A simple EBNF grammar example:
    # This grammar expects the model to produce something like: "Hello Alice"
    # where the name is one or more alphabetic characters.
    cls.ebnf_grammar = r"""
    root ::= "Hello" " " name
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


class TestEBNFConstrainedXGrammarBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="xgrammar", disable_overlap=False)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(self, ebnf, return_logprob=False, top_logprobs_num=0, n=1):
        # Send a request that requires EBNF guided output
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Generate a greeting:",
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 128,
                    "n": n,
                    "ebnf": ebnf,
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )
        ret = response.json()
        print(json.dumps(ret, indent=2))

        # Validate that the returned text matches the EBNF pattern "Hello <name>"
        # We'll perform a basic check to ensure the output starts with "Hello "
        # and that the following characters are alphabetic.
        text = ret["text"].strip()
        self.assertTrue(
            text.startswith("Hello "), f"Text does not start with 'Hello ': {text}"
        )
        name_part = text[len("Hello ") :].strip()
        self.assertRegex(
            name_part, r"^[A-Za-z]+$", f"Name part is not alphabetic: {name_part}"
        )

    def test_ebnf_generate(self):
        self.run_decode(ebnf=self.__class__.ebnf_grammar)
