"""Pause control errors reach clients without starting an inference server."""

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sglang.srt.entrypoints import http_server
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPauseGenerationRoute(CustomTestCase):
    def test_rejected_pause_is_bad_request_and_valid_pause_still_succeeds(self):
        """An unsupported pause mode must return its reason as HTTP 400,
        rather than an internal server error; a valid pause remains HTTP 200."""
        message = "PD decode PP speculative decoding does not support pause/retract"
        manager = SimpleNamespace(
            pause_generation=AsyncMock(side_effect=ValueError(message))
        )
        app = FastAPI()
        app.add_api_route(
            "/pause_generation", http_server.pause_generation, methods=["POST"]
        )
        with (
            patch.object(
                http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
            ),
            TestClient(app, raise_server_exceptions=False) as client,
        ):
            response = client.post("/pause_generation", json={"mode": "retract"})
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json(), {"error": {"message": message}})

            manager.pause_generation.side_effect = None
            response = client.post("/pause_generation", json={"mode": "in_place"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json(),
                {"message": "Generation paused successfully.", "status": "ok"},
            )


if __name__ == "__main__":
    unittest.main()
