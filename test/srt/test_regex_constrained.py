"""
python3 -m unittest test_regex_constrained.TestRegexConstrained.test_regex_generate_email
python3 -m unittest test_regex_constrained.TestRegexConstrained.test_regex_generate_greeting
python3 -m unittest test_regex_constrained.TestRegexConstrainedLLGuidance.test_regex_generate_email
python3 -m unittest test_regex_constrained.TestRegexConstrainedLLGuidance.test_regex_generate_greeting
"""

import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def setup_class(cls, backend: str, disable_overlap: bool):
    cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST

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


class TestRegexConstrained(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, "xgrammar", disable_overlap=False)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def run_decode(
        self,
        regex,
        prompt,
        return_logprob=False,
        top_logprobs_num=0,
        n=1,
    ):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0 if n == 1 else 0.5,
                    "max_new_tokens": 128,
                    "n": n,
                    "regex": regex,
                },
                "stream": False,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
                "logprob_start_len": 0,
            },
        )

        ret = response.json()
        print(json.dumps(ret, indent=2))
        print("=" * 100)

        if not isinstance(ret, list):
            self.fail(f"Expected response to be a list, but got {type(ret)}")

        for item in ret:
            text = item.get("text", "").strip()
            if not text:
                self.fail("Generated text is empty.")

            if not self.regex_match(text, regex):
                self.fail(f"Text '{text}' does not match regex pattern.")

    def regex_match(self, text, pattern):
        import re

        return re.match(pattern, text) is not None

    def test_regex_generate_email(self):
        pattern = r"^user@example\.com$"
        prompt = "Generate an email address:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )

    def test_regex_generate_greeting(self):
        pattern = r"^(Hello|Hi|Hey)$"
        prompt = "Generate a greeting:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )

    def test_regex_generate_number(self):
        pattern = r"^\d{3}$"
        prompt = "Generate a three-digit number:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )

    def test_regex_generate_phone(self):
        pattern = r"^\(\d{3}\) \d{3}-\d{4}$"
        prompt = "Generate a phone number:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )

    def test_regex_generate_date(self):
        pattern = r"^2024-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"
        prompt = "Generate a date in YYYY-MM-DD format:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )

    def test_regex_generate_hex_color(self):
        pattern = r"^#[0-9A-F]{6}$"
        prompt = "Generate a hex color code:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )

    def test_regex_generate_complex_json(self):
        pattern = r'^\{\s*"name"\s*:\s*"[a-zA-Z0-9 ]+"\s*,\s*"age"\s*:\s*[1-9][0-9]*\s*,\s*"city"\s*:\s*"[a-zA-Z0-9 ]+"\s*\}$'
        prompt = "Generate a simple JSON with name, age, and city:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )

    def test_regex_generate_custom_log_format(self):
        pattern = r"^\[2024-01-01T12:00:00Z\] INFO: System\.process - Operation [a-z]+ successfully$"
        prompt = "Generate a log entry:"

        self.run_decode(
            regex=pattern,
            prompt=prompt,
            n=3,
        )


class TestRegexConstrainedLLGuidance(TestRegexConstrained):
    @classmethod
    def setUpClass(cls):
        setup_class(cls, "llguidance", disable_overlap=True)


if __name__ == "__main__":
    unittest.main()
