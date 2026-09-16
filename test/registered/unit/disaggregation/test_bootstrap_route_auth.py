"""Unit tests for PD bootstrap PUT /route write-auth (issue #39400)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

import socket
import time
import unittest

import requests

from sglang.srt.disaggregation.common.bootstrap_auth import (
    bootstrap_auth_headers,
    ensure_bootstrap_auth_token,
    get_bootstrap_auth_token,
    is_bootstrap_write_authorized,
)
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase

_ROUTE_PUT_PAYLOAD = {
    "attn_tp_size": 1,
    "attn_tp_rank": 0,
    "attn_cp_size": 1,
    "attn_cp_rank": 0,
    "attn_dp_size": 1,
    "attn_dp_rank": 0,
    "pp_size": 1,
    "pp_rank": 0,
    "system_dp_size": 1,
    "system_dp_rank": 0,
    "rank_ip": "198.51.100.77",
    "rank_port": 65001,
    "page_size": 1,
    "kv_cache_dtype": "auto",
    "load_balance_method": "round_robin",
    "prefill_http_port": 30000,
}

_SENTINEL_ROUTE = (
    "/route?prefill_dp_rank=-1&prefill_cp_rank=-1&target_tp_rank=-1&target_pp_rank=-1"
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_healthy(port: int, timeout_s: float = 5.0) -> None:
    deadline = time.time() + timeout_s
    last_err = None
    while time.time() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/health", timeout=0.2)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(0.02)
    raise TimeoutError(
        f"bootstrap server on port {port} never became healthy: {last_err}"
    )


class TestBootstrapWriteAuthHelper(CustomTestCase):
    def test_missing_expected_token_is_rejected(self):
        self.assertFalse(
            is_bootstrap_write_authorized("Bearer secret", expected_token=None)
        )
        self.assertFalse(
            is_bootstrap_write_authorized("Bearer secret", expected_token="")
        )

    def test_missing_header_is_rejected(self):
        self.assertFalse(is_bootstrap_write_authorized(None, "secret"))
        self.assertFalse(is_bootstrap_write_authorized("", "secret"))

    def test_wrong_scheme_or_token_is_rejected(self):
        self.assertFalse(is_bootstrap_write_authorized("Basic secret", "secret"))
        self.assertFalse(is_bootstrap_write_authorized("Bearer other", "secret"))

    def test_matching_bearer_token_is_accepted(self):
        self.assertTrue(is_bootstrap_write_authorized("Bearer secret", "secret"))
        self.assertTrue(is_bootstrap_write_authorized("bearer secret", "secret"))

    def test_ensure_generates_then_reuses_env_token(self):
        with envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.override(None):
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.clear()
            generated = ensure_bootstrap_auth_token()
            self.assertTrue(generated)
            self.assertEqual(get_bootstrap_auth_token(), generated)
            self.assertEqual(ensure_bootstrap_auth_token(api_key="ignored"), generated)

    def test_ensure_prefers_existing_env_over_api_key(self):
        with envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.override("env-token"):
            self.assertEqual(
                ensure_bootstrap_auth_token(api_key="api-key"), "env-token"
            )
            self.assertEqual(
                bootstrap_auth_headers(), {"Authorization": "Bearer env-token"}
            )


class TestBootstrapRoutePutHttpAuth(CustomTestCase):
    """Live aiohttp bootstrap server: unauthenticated PUT /route must not persist."""

    def setUp(self):
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        self.token = "bootstrap-test-secret"
        self.port = _free_port()
        self.server = CommonKVBootstrapServer(
            "127.0.0.1", self.port, auth_token=self.token
        )
        self.addCleanup(self.server.close)
        _wait_healthy(self.port)
        self.base = f"http://127.0.0.1:{self.port}"

    def test_unauthenticated_put_is_unauthorized_and_does_not_register(self):
        response = requests.put(
            f"{self.base}/route", json=_ROUTE_PUT_PAYLOAD, timeout=2
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.text, "Unauthorized")
        self.assertEqual(self.server._registered_count, 0)
        self.assertEqual(self.server.prefill_port_table, {})

        sentinel = requests.get(f"{self.base}{_SENTINEL_ROUTE}", timeout=2)
        self.assertEqual(sentinel.status_code, 503)

    def test_wrong_token_put_is_unauthorized_and_does_not_register(self):
        response = requests.put(
            f"{self.base}/route",
            json=_ROUTE_PUT_PAYLOAD,
            headers={"Authorization": "Bearer wrong-token"},
            timeout=2,
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.server._registered_count, 0)

    def test_authenticated_put_registers_and_get_stays_unauthenticated(self):
        response = requests.put(
            f"{self.base}/route",
            json=_ROUTE_PUT_PAYLOAD,
            headers=bootstrap_auth_headers(self.token),
            timeout=2,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.server._registered_count, 1)
        info = self.server.prefill_port_table[0][0][0][0]
        self.assertEqual(info.rank_ip, "198.51.100.77")
        self.assertEqual(info.rank_port, 65001)

        sentinel = requests.get(f"{self.base}{_SENTINEL_ROUTE}", timeout=2)
        self.assertEqual(sentinel.status_code, 200)

        rank = requests.get(
            f"{self.base}/route?prefill_dp_rank=0&prefill_cp_rank=0"
            "&target_tp_rank=0&target_pp_rank=0",
            timeout=2,
        )
        self.assertEqual(rank.status_code, 200)
        self.assertEqual(rank.json(), {"rank_ip": "198.51.100.77", "rank_port": 65001})

    def test_unauthenticated_register_dp_rank_is_rejected(self):
        response = requests.post(
            f"{self.base}/register_dp_rank",
            json={"bootstrap_room": 1, "dp_rank": 0},
            timeout=2,
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.server.room_to_dp_rank, {})

    def test_no_token_configured_still_rejects_put(self):
        from sglang.srt.disaggregation.common.conn import CommonKVBootstrapServer

        port = _free_port()
        with envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.override(None):
            envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.clear()
            server = CommonKVBootstrapServer("127.0.0.1", port, auth_token="")
            self.addCleanup(server.close)
            _wait_healthy(port)
            response = requests.put(
                f"http://127.0.0.1:{port}/route",
                json=_ROUTE_PUT_PAYLOAD,
                timeout=2,
            )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(server._registered_count, 0)


if __name__ == "__main__":
    unittest.main()
