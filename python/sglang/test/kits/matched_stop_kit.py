import json

import requests

MANY_NEW_TOKENS_PROMPT = """
Please write an extremely detailed and vivid fantasy story, set in a world full of intricate magic systems, political intrigue, and complex characters.
Ensure that you thoroughly describe every scene, character's motivations, and the environment. Include long, engaging dialogues and elaborate on the inner thoughts of the characters.
Each section should be as comprehensive as possible to create a rich and immersive experience for the reader.
The story should span multiple events, challenges, and character developments over time. Aim to make the story at least 3,000 words long.
"""


class MatchedStopMixin:
    def _run_completions_generation(
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
        res = response_completions.json()
        print(json.dumps(res))
        print("=" * 100)

        if not isinstance(matched_stop, list):
            matched_stop = [matched_stop]

        assert (
            res["choices"][0]["finish_reason"] == finish_reason
        ), f"Expected finish_reason: {finish_reason}, but got: {res['choices'][0]['finish_reason']}"
        assert (
            res["choices"][0]["matched_stop"] in matched_stop
        ), f"Expected matched_stop: {matched_stop}, but got: {res['choices'][0]['matched_stop']}"

    def _run_chat_completions_generation(
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
        res = response_chat.json()
        print(json.dumps(res))
        print("=" * 100)

        if not isinstance(matched_stop, list):
            matched_stop = [matched_stop]

        assert (
            res["choices"][0]["finish_reason"] == finish_reason
        ), f"Expected finish_reason: {finish_reason}, but got: {res['choices'][0]['finish_reason']}"
        assert (
            res["choices"][0]["matched_stop"] in matched_stop
        ), f"Expected matched_stop: {matched_stop}, but got: {res['choices'][0]['matched_stop']}"

    def test_finish_stop_str(self):
        self._run_completions_generation(
            max_tokens=1000, stop="\n", finish_reason="stop", matched_stop="\n"
        )
        self._run_chat_completions_generation(
            max_tokens=1000, stop="\n", finish_reason="stop", matched_stop="\n"
        )

    def test_finish_stop_regex_str(self):
        STOP_REGEX_STR = r"and|or"
        self._run_completions_generation(
            max_tokens=1000,
            stop_regex=STOP_REGEX_STR,
            finish_reason="stop",
            matched_stop=STOP_REGEX_STR,
        )
        self._run_chat_completions_generation(
            max_tokens=1000,
            stop_regex=STOP_REGEX_STR,
            finish_reason="stop",
            matched_stop=STOP_REGEX_STR,
        )

        # Match a complete sentence
        STOP_REGEX_STR_SENTENCE = r"[.!?]\s*$"
        self._run_chat_completions_generation(
            max_tokens=1000,
            stop_regex=STOP_REGEX_STR_SENTENCE,
            finish_reason="stop",
            matched_stop=STOP_REGEX_STR_SENTENCE,
        )

    def test_finish_stop_eos(self):
        llama_format_prompt = """\
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
What is 2 + 2?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        eos_token_ids = [128000, 128009, 2]
        self._run_completions_generation(
            prompt=llama_format_prompt,
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_ids,
        )
        self._run_chat_completions_generation(
            prompt="What is 2 + 2?",
            max_tokens=1000,
            finish_reason="stop",
            matched_stop=eos_token_ids,
        )

    def test_finish_length(self):
        self._run_completions_generation(
            max_tokens=5, finish_reason="length", matched_stop=None
        )
        self._run_chat_completions_generation(
            max_tokens=5, finish_reason="length", matched_stop=None
        )
