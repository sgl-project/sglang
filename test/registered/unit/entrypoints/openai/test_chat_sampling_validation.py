"""Invalid chat sampling parameters must return HTTP 400 before generation."""

import unittest

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.srt.sampling.sampling_params import MAX_N
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


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
            ("top_logprobs", (-1, 21)),
            ("temperature", (-0.1, 2.1)),
        ):
            for value in values:
                with self.subTest(field=field, value=value):
                    response = self._post(**{field: value}, logprobs=True)
                    self._assert_validation_error(response, field)

    def test_invalid_streaming_request_returns_400(self):
        """A streaming request must fail validation before generation starts."""
        response = self._post(n=0, stream=True, logprobs=True)
        self._assert_validation_error(response, "n")

    def test_sampling_boundaries_are_accepted(self):
        for n, top_logprobs in ((1, 0), (MAX_N, 20)):
            with self.subTest(n=n, top_logprobs=top_logprobs):
                response = self._post(n=n, top_logprobs=top_logprobs, logprobs=True)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {"n": n, "top_logprobs": top_logprobs, "temperature": None},
                )

    def test_default_and_null_top_logprobs_are_accepted(self):
        for params in ({}, {"top_logprobs": None}):
            with self.subTest(params=params):
                response = self._post(**params)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json(),
                    {"n": 1, "top_logprobs": None, "temperature": None},
                )

    def test_positive_top_logprobs_requires_logprobs(self):
        """A candidate count previously passed validation with logprobs disabled."""
        for top_logprobs in (1, 5, 20):
            for params in ({}, {"logprobs": False}):
                with self.subTest(top_logprobs=top_logprobs, params=params):
                    response = self._post(top_logprobs=top_logprobs, **params)
                    self._assert_validation_error(response)
                    self.assertIn("logprobs", response.json()["message"])
                    self.assertIn("top_logprobs", response.json()["message"])

    def test_valid_logprobs_combinations_are_accepted(self):
        for logprobs, top_logprobs in (
            (False, None),
            (False, 0),
            (True, None),
            (True, 0),
            (True, 1),
            (True, 5),
            (True, 20),
        ):
            with self.subTest(logprobs=logprobs, top_logprobs=top_logprobs):
                response = self._post(logprobs=logprobs, top_logprobs=top_logprobs)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["top_logprobs"], top_logprobs)

    def test_temperature_boundaries_and_defaults_are_accepted(self):
        for params in (
            {},
            {"temperature": None},
            {"temperature": 0.0},
            {"temperature": 2.0},
        ):
            with self.subTest(params=params):
                response = self._post(**params)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(
                    response.json()["temperature"], params.get("temperature")
                )

    def test_logit_bias_bounds_return_400(self):
        """Bias values outside the documented range previously reached serving."""
        for value in (-101.0, 101.0):
            with self.subTest(value=value):
                response = self._post(logit_bias={"1": value})
                self._assert_validation_error(response, "logit_bias", "1")

    def test_logit_bias_nonfinite_returns_400(self):
        """Valid JSON numbers that overflow float64 must not enter sampling."""
        for value in ("1e309", "-1e309"):
            with self.subTest(value=value):
                self.validation_errors.clear()
                response = self.client.post(
                    "/v1/chat/completions",
                    content=(
                        '{"model":"test-model",'
                        '"messages":[{"role":"user","content":"Hello"}],'
                        f'"logit_bias":{{"1":{value}}}'
                        "}"
                    ),
                    headers={"Content-Type": "application/json"},
                )
                self._assert_validation_error(response, "logit_bias", "1")

    def test_logit_bias_boundaries_and_defaults_are_accepted(self):
        for value in (None, {}, {"1": -100.0}, {"1": 100.0}, {"1": 0.5}):
            with self.subTest(value=value):
                response = self._post(logit_bias=value)
                self.assertEqual(response.status_code, 200)

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


if __name__ == "__main__":
    unittest.main()
