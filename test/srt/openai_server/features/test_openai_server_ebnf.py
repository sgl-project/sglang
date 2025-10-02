import re

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


# -------------------------------------------------------------------------
#    EBNF Test Class: TestOpenAIServerEBNF
#    Launches the server with xgrammar, has only EBNF tests
# -------------------------------------------------------------------------
class TestOpenAIServerEBNF(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # passing xgrammar specifically
        other_args = ["--grammar-backend", "xgrammar"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=other_args,
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

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
