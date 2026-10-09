"""The `/generate` schema gate (`entrypoints/api_contract.py`): the typed media
contract that replaced the raw_json passthrough, checked without a server, and
where the route applies it."""

import unittest
from types import SimpleNamespace

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.api_contract import generate_contract_error
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestGenerateContract(CustomTestCase):
    def test_every_media_container_form_is_accepted(self):
        for image_data in (
            "u",
            {"url": "u", "detail": "high"},
            ["u", None, {"url": "v"}],
            [],
            [["a", "b"], None, []],
        ):
            with self.subTest(image_data=image_data):
                self.assertIsNone(
                    generate_contract_error({"text": "hi", "image_data": image_data})
                )
        self.assertIsNone(
            generate_contract_error({"text": "hi", "mm_hashes": ["a1b2", "0xff"]})
        )
        self.assertIsNone(
            generate_contract_error(
                {
                    "text": "hi",
                    "video_data": {"url": "v", "fps": 2.0, "use_audio": True},
                }
            )
        )

    def test_unknown_media_hint_is_rejected(self):
        """A hint the schema does not name used to be silently dropped."""
        error = generate_contract_error(
            {
                "text": "hi",
                "image_data": {"url": "u", "max_dynamic_patc": 4},
            }  # codespell:ignore patc
        )
        self.assertIsNotNone(error)
        self.assertIn("unknown field", error)

    def test_preprocessed_inputs_are_not_an_http_shape(self):
        """processor_output / precomputed_embedding carry tensors: Engine-only."""
        error = generate_contract_error(
            {"input_ids": [1, 2], "image_data": [{"format": "processor_output"}]}
        )
        self.assertIsNotNone(error)

    def test_mixed_list_shapes_are_rejected(self):
        error = generate_contract_error({"text": "hi", "image_data": ["a", ["b"]]})
        self.assertIsNotNone(error)

    def test_sampling_params_follow_the_schema(self):
        self.assertIsNone(
            generate_contract_error(
                {"text": "hi", "sampling_params": {"beam_width": 2, "n": 2}}
            )
        )
        bad = {"temperatur": 1}  # codespell:ignore temperatur
        error = generate_contract_error({"text": "hi", "sampling_params": bad})
        self.assertIsNotNone(error)
        self.assertIn("unknown field", error)


class TestGenerateRouteGate(CustomTestCase):
    def test_route_checks_the_contract_and_serve_does_not(self):
        """`/generate` refuses a contract violation before serving it;
        `serve_generate_request`, which routes with their own parser call,
        serves it without decoding and checking the body again."""
        served = []

        class FakeTokenizerManager:
            async def generate_request(self, obj, request):
                served.append(obj)
                yield {"text": "ok"}

        bad = {"temperatur": 1}  # codespell:ignore temperatur
        body = {"text": "hi", "sampling_params": bad}

        async def admitted(request: Request):
            obj = GenerateReqInput(**(await request.json()))
            return await http_server.serve_generate_request(obj, request)

        app = FastAPI()
        app.add_api_route("/generate", http_server.generate_request, methods=["POST"])
        app.add_api_route("/admitted", admitted, methods=["POST"])
        prior_state = http_server.get_global_state()
        http_server.set_global_state(
            SimpleNamespace(tokenizer_manager=FakeTokenizerManager())
        )
        try:
            client = TestClient(app)
            response = client.post("/generate", json=body)
            self.assertEqual(response.status_code, 400)
            self.assertIn("unknown field", response.json()["error"])
            self.assertEqual(served, [])

            response = client.post("/admitted", json=body)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"text": "ok"})
            self.assertEqual(len(served), 1)
        finally:
            http_server._global_state = prior_state


if __name__ == "__main__":
    unittest.main()
