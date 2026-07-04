"""Basic HTTP / SSE API contract sanity kit.

Probes that catch the server failing at the protocol layer: endpoints
missing, 5xx returned, response schema broken, or OpenAI-compatible
routes drifting from the spec.

Mix into any ``CustomTestCase`` subclass that exposes ``self.base_url``
and ``self.process``. Override ``served_model_name`` if the OpenAI
probes should pin a specific model id."""

import requests

_REQUEST_TIMEOUT = 60


class BasicAPIContractMixin:
    """Health endpoints + OpenAI /v1 surface probes."""

    served_model_name: str = "default"

    def test_health(self):
        # Cheapest possible alive check; FastAPI route alone.
        resp = requests.get(self.base_url + "/health", timeout=10)
        self.assertEqual(resp.status_code, 200)

    def test_health_generate(self):
        # sglang's built-in minimal-forward sanity. 200 only if the
        # scheduler can complete one prefill+decode end to end.
        resp = requests.get(self.base_url + "/health_generate", timeout=60)
        self.assertEqual(resp.status_code, 200)

    def test_get_server_info(self):
        resp = requests.get(self.base_url + "/get_server_info", timeout=10)
        self.assertEqual(resp.status_code, 200)
        info = resp.json()
        # Must expose at least some scheduler/server-args bundle.
        self.assertIsInstance(info, dict)
        self.assertGreater(len(info), 0)

    def test_get_model_info(self):
        resp = requests.get(self.base_url + "/get_model_info", timeout=10)
        self.assertEqual(resp.status_code, 200)
        info = resp.json()
        self.assertIn("model_path", info)
        self.assertTrue(info["model_path"])

    def test_openai_chat_completion(self):
        resp = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.served_model_name,
                "messages": [
                    {"role": "user", "content": "Say hi in one word."},
                ],
                "temperature": 0.0,
                "max_tokens": 16,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("choices", body)
        self.assertGreater(len(body["choices"]), 0)
        content = body["choices"][0]["message"]["content"]
        self.assertIsInstance(content, str)
        self.assertGreater(len(content), 0)
        self.assertIn("usage", body)

    def test_openai_completion(self):
        resp = requests.post(
            self.base_url + "/v1/completions",
            json={
                "model": self.served_model_name,
                "prompt": "The capital of France is",
                "temperature": 0.0,
                "max_tokens": 16,
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertIn("choices", body)
        self.assertGreater(len(body["choices"]), 0)
        text = body["choices"][0]["text"]
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        self.assertIn("usage", body)
