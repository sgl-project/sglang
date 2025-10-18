"""
gRPC Router E2E Test - Test Openai Server Ebnf

This test file is REUSED from test/srt/openai_server/features/test_openai_server_ebnf.py
with minimal changes:
    num_workers=2,
- Swap popen_launch_server() → popen_launch_grpc_router()
- Update teardown to cleanup router + workers
- All test logic and assertions remain identical

Run with:
    pytest py_test/e2e_grpc/e2e_grpc/features/test_openai_server_ebnf.py -v
"""

import re

# CHANGE: Import router launcher instead of server launcher
import sys
from pathlib import Path
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR.parent))
from fixtures import popen_launch_grpc_router

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,

)

# -------------------------------------------------------------------------
#    EBNF Test Class: TestOpenAIServerEBNF
#    Launches the server with xgrammar, has only EBNF tests
# -------------------------------------------------------------------------
class TestOpenAIServerEBNF(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # CHANGE: Launch gRPC router with integrated workers (single command)
        cls.model = "/home/ubuntu/models/llama-3.1-8b-instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # passing xgrammar specifically
        other_args = ["--grammar-backend", "xgrammar"]
        cls.cluster = popen_launch_grpc_router(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
            num_workers=2,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer("/home/ubuntu/models/llama-3.1-8b-instruct")

    @classmethod
    def tearDownClass(cls):
        # CHANGE: Cleanup single process (router + workers integrated)
        kill_process_tree(cls.cluster["process"].pid)

    def test_ebnf(self):
        """
        Ensure we can pass `ebnf` to the local openai server
        and that it enforces the grammar.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        ebnf_grammar = r"""
        root ::= "Hello" | "Hi" | "Hey"
        """
        pattern = re.compile(r"^(Hello|Hi|Hey)[.!?]*\s*$")

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful EBNF test bot."},
                {"role": "user", "content": "Say a greeting (Hello, Hi, or Hey)."},
            ],
            temperature=0,
            max_tokens=32,
            extra_body={"ebnf": ebnf_grammar},
        )
        text = response.choices[0].message.content.strip()
        self.assertTrue(len(text) > 0, "Got empty text from EBNF generation")
        self.assertRegex(text, pattern, f"Text '{text}' doesn't match EBNF choices")

    def test_ebnf_strict_json(self):
        """
        A stricter EBNF that produces exactly {"name":"Alice"} format
        with no trailing punctuation or extra fields.
        """
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        ebnf_grammar = r"""
        root    ::= "{" pair "}"
        pair    ::= "\"name\"" ":" string
        string  ::= "\"" [A-Za-z]+ "\""
        """
        pattern = re.compile(r'^\{"name":"[A-Za-z]+"\}$')

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "EBNF mini-JSON generator."},
                {
                    "role": "user",
                    "content": "Generate single key JSON with only letters.",
                },
            ],
            temperature=0,
            max_tokens=64,
            extra_body={"ebnf": ebnf_grammar},
        )
        text = response.choices[0].message.content.strip()
        self.assertTrue(len(text) > 0, "Got empty text from EBNF strict JSON test")
        self.assertRegex(
            text, pattern, f"Text '{text}' not matching the EBNF strict JSON shape"
        )
