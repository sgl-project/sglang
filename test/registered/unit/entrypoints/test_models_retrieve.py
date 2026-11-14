"""`GET /v1/models/{model}` must answer for the adapters `GET /v1/models` publishes.

`available_models` appends one card per loaded adapter; before this change the retrieve
handler compared the id against the base served name only and answered 404 for every
adapter the same server had just listed.
"""

import asyncio
from types import SimpleNamespace
from unittest import mock

import orjson

from sglang.srt.entrypoints import http_server
from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _retrieve(model, *, adapters=(), enable_lora=True):
    registry = LoRARegistry(
        lora_paths=[
            LoRARef(lora_name=lora_name, lora_path=lora_path)
            for lora_name, lora_path in adapters
        ]
    )
    tokenizer_manager = SimpleNamespace(
        served_model_name="base-model",
        lora_registry=registry,
        model_config=SimpleNamespace(context_len=32768),
    )
    prior_state = http_server.get_global_state()
    http_server.set_global_state(SimpleNamespace(tokenizer_manager=tokenizer_manager))
    try:
        with mock.patch.object(
            http_server,
            "get_lora",
            return_value=SimpleNamespace(enable_lora=enable_lora),
        ):
            return asyncio.run(http_server.retrieve_model(model))
    finally:
        http_server._global_state = prior_state


class TestRetrieveModel(CustomTestCase):
    def test_base_model_is_unchanged(self):
        card = _retrieve("base-model")

        self.assertEqual(card.id, "base-model")
        self.assertEqual(card.root, "base-model")
        self.assertIsNone(card.parent)
        self.assertEqual(card.max_model_len, 32768)

    def test_listed_adapter_is_retrievable(self):
        card = _retrieve("billing", adapters=[("billing", "/models/billing")])

        # The same card `available_models` appends for a loaded adapter.
        self.assertEqual(card.id, "billing")
        self.assertEqual(card.root, "/models/billing")
        self.assertEqual(card.parent, "base-model")
        self.assertIsNone(card.max_model_len)

    def test_only_the_requested_adapter_matches(self):
        response = _retrieve(
            "shipping",
            adapters=[("billing", "/models/billing"), ("shipping", "/models/shipping")],
        )

        self.assertEqual(response.id, "shipping")
        self.assertEqual(response.root, "/models/shipping")

    def test_unknown_model_is_still_a_404(self):
        response = _retrieve("missing", adapters=[("billing", "/models/billing")])

        self.assertEqual(response.status_code, 404)
        payload = orjson.loads(response.body)
        self.assertEqual(payload["error"]["code"], "model_not_found")
        self.assertEqual(payload["error"]["param"], "model")

    def test_adapter_is_not_retrievable_without_lora(self):
        response = _retrieve("billing", enable_lora=False)

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    import unittest

    unittest.main()
