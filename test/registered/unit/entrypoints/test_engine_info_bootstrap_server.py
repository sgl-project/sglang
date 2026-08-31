import socket
import time
import unittest

import requests

from sglang.srt.entrypoints.engine_info_bootstrap_server import (
    EngineInfoBootstrapServer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _get_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestEngineInfoBootstrapServer(unittest.TestCase):
    def setUp(self):
        port = _get_free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        self.server = EngineInfoBootstrapServer("127.0.0.1", port)
        self.addCleanup(self.server.close)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if requests.get(f"{self.base_url}/health", timeout=0.1).ok:
                    break
            except requests.RequestException:
                pass
        else:
            self.fail("EngineInfoBootstrapServer did not become ready")

    def _register(
        self,
        rank: int,
        session_id: str,
        weight_name: str,
        *,
        rank_field: str = "rank",
        extra_fields: dict | None = None,
    ) -> None:
        payload = {
            rank_field: rank,
            "transfer_engine_info": {
                "session_id": session_id,
                "weights_info_dict": {weight_name: [1, 2, 4]},
            },
        }
        if extra_fields is not None:
            payload.update(extra_fields)
        response = requests.put(
            f"{self.base_url}/register_transfer_engine_info",
            json=payload,
            timeout=1,
        )
        response.raise_for_status()

    def _get(self, rank: int) -> dict:
        response = requests.get(
            f"{self.base_url}/get_transfer_engine_info",
            params={"rank": rank},
            timeout=1,
        )
        response.raise_for_status()
        return response.json()

    def test_metadata_is_kept_for_each_pipeline_rank(self):
        self._register(0, "pp0-session", "model.layers.0.weight")
        self._register(1, "pp1-session", "model.layers.9.weight")

        self.assertEqual(
            self._get(0)["remote_instance_transfer_engine_info"][0],
            "pp0-session",
        )
        self.assertEqual(
            self._get(1)["remote_instance_transfer_engine_info"][0],
            "pp1-session",
        )

    def test_legacy_tp_rank_registration_is_still_accepted(self):
        self._register(
            0,
            "legacy-session",
            "model.weight",
            rank_field="tp_rank",
        )

        self.assertEqual(
            self._get(0)["remote_instance_transfer_engine_info"][0],
            "legacy-session",
        )

    def test_rank_takes_precedence_when_both_fields_are_present(self):
        self._register(
            0,
            "rank-zero-session",
            "model.weight",
            extra_fields={"tp_rank": 7},
        )

        self.assertEqual(
            self._get(0)["remote_instance_transfer_engine_info"][0],
            "rank-zero-session",
        )
        missing_legacy_rank = requests.get(
            f"{self.base_url}/get_transfer_engine_info",
            params={"rank": 7},
            timeout=1,
        )
        self.assertEqual(missing_legacy_rank.status_code, 404)


if __name__ == "__main__":
    unittest.main()
