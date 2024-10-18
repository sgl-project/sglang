import json
import unittest

import requests

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

MANY_NEW_TOKENS_PROMPT = """
Please write an extremely detailed and vivid fantasy story, set in a world full of intricate magic systems, political intrigue, and complex characters. 
Ensure that you thoroughly describe every scene, character's motivations, and the environment. Include long, engaging dialogues and elaborate on the inner thoughts of the characters. 
Each section should be as comprehensive as possible to create a rich and immersive experience for the reader. 
The story should span multiple events, challenges, and character developments over time. Aim to make the story at least 3,000 words long.
"""


class TestMatchedStop(unittest.TestCase):
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
        kill_child_process(cls.process.pid)

    def run_completions_generation(
        self,
        prompt=MANY_NEW_TOKENS_PROMPT,
        max_tokens=1,
        stop=None,
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


if __name__ == "__main__":
    unittest.main()
