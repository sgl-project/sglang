"""Invalid chat sampling parameters must return HTTP 400 before generation."""

import unittest

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.sampling.sampling_params import MAX_N
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestChatSamplingValidation(CustomTestCase):
    client = None

    @classmethod
    def setUpClass(cls):
        class SamplingEcho:
            async def handle_request(self, request, raw_request):
                return {
                    "n": request.n,
                    "top_logprobs": request.top_logprobs,
                    "temperature": request.temperature,
                }

        validation_handler = http_server.app.exception_handlers[RequestValidationError]
        cls.validation_errors = []

        async def capture_validation_error(request, exc):
            cls.validation_errors[:] = exc.errors()
            return await validation_handler(request, exc)

        app = FastAPI()
        app.add_exception_handler(RequestValidationError, capture_validation_error)
        app.add_api_route(
            "/v1/chat/completions",
            http_server.openai_v1_chat_completions,
            methods=["POST"],
        )
        app.state.openai_serving_chat = SamplingEcho()
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        if cls.client is not None:
            cls.client.close()
            cls.client = None

    def _post(self, **params):
        self.validation_errors.clear()
        return self.client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1,
                **params,
            },
        )

    def _assert_validation_error(self, response, *loc):
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            [tuple(error["loc"]) for error in self.validation_errors],
            [("body", *loc)],
        )

    def test_invalid_sampling_bounds_return_400(self):
        """Out-of-range requests previously passed validation and reached serving."""
        for field, values in (
            ("n", (-1, 0, MAX_N + 1)),
            ("top_logprobs", (-1,)),
            ("temperature", (-0.1,)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    response = self._post(**{field: value}, logprobs=True)
                    self._assert_validation_error(response, field)

    def test_sampling_boundaries_are_accepted(self):
        for n, top_logprobs in ((1, 0), (MAX_N, 20)):
            with self.subTest(n=n, top_logprobs=top_logprobs):
                response = self._post(n=n, top_logprobs=top_logprobs, logprobs=True)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {"n": n, "top_logprobs": top_logprobs, "temperature": None},
                )

    def test_temperature_boundaries_and_defaults_are_accepted(self):
        for params in (
            {"temperature": None},
            {"temperature": 0.0},
        ):
            with self.subTest(params=params):
                response = self._post(**params)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json()["temperature"], params.get("temperature")
                )

    def test_temperature_infinity_returns_400(self):
        """An infinite temperature must be rejected now that temperature has no cap."""
        response = self.client.post(
            "/v1/chat/completions",
            content=(
                '{"model":"test-model",'
                '"messages":[{"role":"user","content":"Hello"}],'
                '"temperature":Infinity}'
            ),
            headers={"Content-Type": "application/json"},
        )
        self._assert_validation_error(response, "temperature")

    def test_negative_output_limits_return_400(self):
        """A preferred completion limit previously hid a negative legacy limit."""
        for params, field in (
            ({"max_tokens": -1}, "max_tokens"),
            ({"max_completion_tokens": -1}, "max_completion_tokens"),
            ({"max_tokens": -1, "max_completion_tokens": 1}, "max_tokens"),
        ):
            with self.subTest(params=params):
                response = self._post(**params)
                self._assert_validation_error(response, field)

    def test_openai_limits_are_deliberately_not_enforced(self):
        """The engine applies these values as expected, so they must not 400."""
        for params in (
            {"temperature": 3.0},
            {"logprobs": True, "top_logprobs": 50},
            {"top_logprobs": 3},
            {"logit_bias": {"1": -1000, "2": 1e4}},
        ):
            with self.subTest(params=params):
                self.assertEqual(self._post(**params).status_code, 200)


if __name__ == "__main__":
    unittest.main()
