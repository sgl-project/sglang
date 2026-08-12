"""JSON mode (response_format json_object) test mixin.

Host class must provide ``self.client`` (openai.Client) and ``self.model``.
"""

import json


class JSONModeMixin:
    def test_json_mode_response(self):
        """json_object without a JSON-mentioning system prompt must still
        produce valid JSON."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # No JSON hint in the prompt on purpose -- the format must be
                # enforced by response_format, not by the instruction.
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
        )
        text = response.choices[0].message.content

        print(f"Response ({len(text)} characters): {text}")

        try:
            js_obj = json.loads(text)
        except json.JSONDecodeError as e:
            self.fail(f"Response is not valid JSON. Error: {e}. Response: {text}")

        self.assertIsInstance(js_obj, dict, f"Response is not a JSON object: {text}")

    def test_json_mode_with_streaming(self):
        """Same contract over a stream: the concatenated chunks must parse."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[
                # No JSON hint in the prompt on purpose -- the format must be
                # enforced by response_format, not by the instruction.
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that gives a short answer.",
                },
                {"role": "user", "content": "What is the capital of Bulgaria?"},
            ],
            temperature=0,
            max_tokens=128,
            response_format={"type": "json_object"},
            stream=True,
        )

        chunks = []
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                chunks.append(chunk.choices[0].delta.content)
        full_response = "".join(chunks)

        print(
            f"Concatenated Response ({len(full_response)} characters): {full_response}"
        )

        try:
            js_obj = json.loads(full_response)
        except json.JSONDecodeError as e:
            self.fail(
                f"Streamed response is not valid JSON. Error: {e}. Response: {full_response}"
            )

        self.assertIsInstance(js_obj, dict)
