"""API-key authorization on the embedded Rust HTTP frontend.

The policy matrix (4 key configurations x 3 auth levels, Bearer parsing, empty-key
normalization) is pinned by `api_server::auth`, and the router-assembly boundary
by `api_server::disaggregation::bootstrap`; both already run in CPU CI through
`test/registered/rust/test_run_rust_tests.py`, so this file does not re-test them
on a GPU.

What only a live server can prove is the wiring: the resolved keys survive the
Python -> PyO3 handoff, and the observable HTTP surface matches the Python
server. Everything below runs against ONE launch.
"""

import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    is_rust_server_built,
    popen_launch_server,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")

UNAUTHORIZED_BODY = {"error": "Unauthorized"}
# Presented by the rejection tests and never configured anywhere, so it can be
# asserted absent from the whole log. The *configured* key cannot: the Python
# scheduler prints its full `server_args=ServerArgs(...)` record at startup,
# which is out of this layer's control (and identical under the Python server).
REJECTED_TOKEN = "sk-rust-authz-never-logged"


@unittest.skipUnless(
    is_rust_server_built(),
    "embedded rust server extension not built",
)
class TestRustServerAuthZ(CustomTestCase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    base_url = DEFAULT_URL_FOR_TEST
    api_key = "sk-rust-authz-user"
    # Configured so the "admin key is not a super-key for NORMAL endpoints"
    # branch is reachable: with both keys set, only `api_key` opens /v1/*.
    admin_api_key = "sk-rust-authz-admin"

    @classmethod
    def setUpClass(cls):
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            env={"SGLANG_RUST_SERVER": "1"},
            other_args=["--admin-api-key", cls.admin_api_key],
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        for stream, path in (
            (getattr(cls, "stdout", None), STDOUT_FILENAME),
            (getattr(cls, "stderr", None), STDERR_FILENAME),
        ):
            if stream:
                stream.close()
            if os.path.exists(path):
                os.remove(path)

    def _request(self, method, path, token=None, **kwargs):
        headers = kwargs.pop("headers", {})
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        return requests.request(
            method, self.base_url + path, headers=headers, timeout=60, **kwargs
        )

    def assert_unauthorized(self, response):
        """The 401 contract Python emits via ORJSONResponse, byte for byte."""
        self.assertEqual(response.status_code, 401)
        self.assertEqual(
            response.headers.get("content-type", "").split(";")[0].strip(),
            "application/json",
        )
        self.assertEqual(response.json(), UNAUTHORIZED_BODY)

    def test_customer_routes_require_the_api_key(self):
        """The key reaches the frontend: same route, three credentials."""
        for token in (None, "wrong-key"):
            with self.subTest(token=token):
                self.assert_unauthorized(self._request("GET", "/v1/models", token))

        self.assertEqual(
            self._request("GET", "/v1/models", self.api_key).status_code, 200
        )

    def test_admin_key_is_not_a_super_key_for_normal_routes(self):
        """Both keys configured: only `api_key` opens NORMAL endpoints -- the one
        policy row that needs a real launch carrying both credentials."""
        for path in ("/v1/models", "/server_info"):
            with self.subTest(path=path):
                self.assert_unauthorized(self._request("GET", path, self.admin_api_key))
                self.assertEqual(
                    self._request("GET", path, self.api_key).status_code, 200
                )

    def test_native_and_common_routes_are_protected(self):
        """Protection is not OpenAI-only: it covers every customer router."""
        self.assert_unauthorized(
            self._request("POST", "/generate", json={"text": "hi"})
        )
        for path in ("/server_info", "/model_info", "/get_model_info"):
            with self.subTest(path=path):
                self.assert_unauthorized(self._request("GET", path))
                self.assertEqual(
                    self._request("GET", path, self.api_key).status_code, 200
                )

    def test_probe_endpoints_stay_public(self):
        """k8s liveness and Prometheus scraping must not need a credential.
        `/metrics` has no Rust handler yet, so its bypass is asserted as not-401."""
        for path in ("/health", "/health_generate", "/metrics"):
            with self.subTest(path=path):
                self.assertNotEqual(self._request("GET", path).status_code, 401)

    def test_unknown_paths_authenticate_before_they_404(self):
        """Regression guard for the layer-before-merge assembly: an unauthenticated
        caller must not be able to probe which paths exist."""
        self.assert_unauthorized(self._request("GET", "/definitely_unknown"))
        self.assertEqual(
            self._request("GET", "/definitely_unknown", self.api_key).status_code, 404
        )
        # ... while the public prefixes keep bypassing it, so a mistyped probe
        # path reports 404 rather than a misleading 401.
        self.assertEqual(self._request("GET", "/health_unknown").status_code, 404)

    def test_authz_precedes_the_method_router(self):
        """A wrong method on a protected path is a 401, not a 405."""
        self.assert_unauthorized(self._request("DELETE", "/v1/models"))
        self.assertEqual(
            self._request("DELETE", "/v1/models", self.api_key).status_code, 405
        )
        # OPTIONS bypasses AuthZ entirely and gets the downstream answer.
        self.assertEqual(self._request("OPTIONS", "/v1/models").status_code, 405)

    def test_authz_precedes_body_extraction(self):
        """Malformed body plus no credential is a 401, never a 400/422: rejecting
        first keeps unparsed bodies out of the tokenizer and scheduler."""
        malformed = {
            "data": "not-json-at-all",
            "headers": {"Content-Type": "application/json"},
        }
        self.assert_unauthorized(self._request("POST", "/generate", **malformed))
        self.assertEqual(
            self._request("POST", "/generate", self.api_key, **malformed).status_code,
            400,
        )

    def test_streaming_rejection_is_a_plain_json_error(self):
        """Failing before the handler means no SSE stream is ever opened."""
        response = self._request(
            "POST",
            "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
                "stream": True,
            },
            stream=True,
        )
        self.assert_unauthorized(response)
        self.assertNotIn("text/event-stream", response.headers.get("content-type", ""))

    def test_authorized_requests_still_generate(self):
        """The layer must be transparent once the credential checks out."""
        response = self._request(
            "POST",
            "/v1/completions",
            self.api_key,
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["choices"][0]["text"])

    def test_no_endpoint_discloses_a_configured_key(self):
        """`/server_info` is NORMAL, so the customer key opens it: were it to echo
        the raw server-args dump, that key would read out `admin_api_key`."""
        for path in ("/server_info", "/model_info", "/get_model_info"):
            with self.subTest(path=path):
                body = self._request("GET", path, self.api_key).text
                self.assertNotIn(self.api_key, body)
                self.assertNotIn(self.admin_api_key, body)

    def _read_server_log(self):
        text = ""
        for path in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(path):
                with open(path, errors="replace") as handle:
                    text += handle.read()
        return text

    def test_access_log_records_rejections_without_the_token(self):
        """401s belong in the access log; the credential that produced them does
        not -- the two failure modes of logging an auth denial."""
        self.assert_unauthorized(self._request("GET", "/v1/models", REJECTED_TOKEN))

        # Rust's stdout is line-buffered, but give the writer a moment to land.
        deadline = time.time() + 10
        log = ""
        while time.time() < deadline:
            log = self._read_server_log()
            if any(
                "/v1/models" in line and "401" in line
                for line in log.splitlines()
                if 'HTTP/1.1"' in line
            ):
                break
            time.sleep(0.5)
        else:
            self.fail("no 401 access-log line for /v1/models found in the server log")

        self.assertNotIn(
            REJECTED_TOKEN, log, "the presented credential was written to the log"
        )


if __name__ == "__main__":
    unittest.main(verbosity=3)
