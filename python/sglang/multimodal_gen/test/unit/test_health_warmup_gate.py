"""Unit tests for diffusion server liveness and readiness endpoints.

`/liveness` reports HTTP availability independently of model warmup.
`/health` and `/health_generate` report readiness for inference traffic.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import URLError

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.http_server import (
    health,
    health_generate,
    liveness,
)
from sglang.multimodal_gen.test.server.test_server_common import (
    _case_warmup_sampling_params,
)
from sglang.multimodal_gen.test.server.test_server_utils import ServerManager


def _make_request(warmup_done) -> SimpleNamespace:
    state = SimpleNamespace(server_warmup_done=warmup_done)
    return SimpleNamespace(app=SimpleNamespace(state=state))


class TestHealthWarmupGate(unittest.IsolatedAsyncioTestCase):
    async def test_liveness_returns_200_before_warmup(self):
        self.assertEqual(await liveness(), {"status": "ok"})

    async def test_health_returns_503_before_warmup(self):
        warmup_done = asyncio.Event()
        resp = await health(_make_request(warmup_done))
        self.assertEqual(resp.status_code, 503)

    async def test_health_returns_200_after_warmup(self):
        warmup_done = asyncio.Event()
        warmup_done.set()
        resp = await health(_make_request(warmup_done))
        self.assertEqual(resp, {"status": "ok"})

    async def test_health_generate_returns_503_before_warmup(self):
        warmup_done = asyncio.Event()
        resp = await health_generate(_make_request(warmup_done))
        self.assertEqual(resp.status_code, 503)

    async def test_health_generate_returns_200_after_warmup(self):
        warmup_done = asyncio.Event()
        warmup_done.set()
        resp = await health_generate(_make_request(warmup_done))
        self.assertEqual(resp, {"status": "ok"})


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeAsyncClient:
    def __init__(self, status_codes: list[int]):
        self._status_codes = iter(status_codes)
        self.get_calls = 0
        self.urls = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, timeout=None):
        self.get_calls += 1
        self.urls.append(url)
        return _FakeResponse(next(self._status_codes))


class TestWaitUntilHttpLive(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_liveness_200(self):
        fake_client = _FakeAsyncClient([503, 200])
        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:11000")
        with (
            mock.patch.object(http_server.httpx, "AsyncClient", fake_client),
            mock.patch.object(http_server.asyncio, "sleep", mock.AsyncMock()),
        ):
            await asyncio.wait_for(
                http_server._wait_until_http_live(server_args), timeout=5.0
            )
        self.assertEqual(fake_client.get_calls, 2)
        self.assertEqual(fake_client.urls, ["http://127.0.0.1:11000/liveness"] * 2)


class _ReadyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _RunningProcess:
    returncode = None

    def poll(self):
        return None


class TestServerManagerReadiness(unittest.TestCase):
    def test_image_case_warmup_uses_one_frame_for_omni_model(self):
        case = SimpleNamespace(
            server_args=SimpleNamespace(modality="image"),
            sampling_params=SimpleNamespace(
                output_size="832x480",
                num_frames=None,
                fps=None,
                seconds=1,
                num_outputs_per_prompt=1,
                image_path=None,
                extras={"num_inference_steps": 35},
            ),
        )

        with mock.patch(
            "sglang.multimodal_gen.test.server.test_server_common."
            "get_sampling_param_field_names_for_server_args",
            return_value=frozenset({"num_inference_steps"}),
        ):
            self.assertEqual(
                _case_warmup_sampling_params(case),
                {
                    "width": 832,
                    "height": 480,
                    "num_frames": 1,
                    "num_inference_steps": 35,
                },
            )

    def test_image_case_warmup_keeps_explicit_frames_and_inputs(self):
        case = SimpleNamespace(
            server_args=SimpleNamespace(modality="image"),
            sampling_params=SimpleNamespace(
                output_size="1024x1024",
                num_frames=4,
                fps=None,
                seconds=1,
                num_outputs_per_prompt=2,
                image_path=["first.png", "second.png"],
                extras={"enable_upscaling": True, "test_only": "ignored"},
            ),
        )

        with mock.patch(
            "sglang.multimodal_gen.test.server.test_server_common."
            "get_sampling_param_field_names_for_server_args",
            return_value=frozenset({"enable_upscaling"}),
        ):
            self.assertEqual(
                _case_warmup_sampling_params(case),
                {
                    "width": 1024,
                    "height": 1024,
                    "num_frames": 4,
                    "num_outputs_per_prompt": 2,
                    "image_path": ["warmup", "warmup"],
                    "enable_upscaling": True,
                },
            )

    def test_video_case_warmup_matches_endpoint_default_fps(self):
        case = SimpleNamespace(
            server_args=SimpleNamespace(modality="video"),
            sampling_params=SimpleNamespace(
                output_size="",
                num_frames=None,
                fps=None,
                seconds=2,
                num_outputs_per_prompt=1,
                image_path=None,
                extras={},
            ),
        )

        with mock.patch(
            "sglang.multimodal_gen.test.server.test_server_common."
            "get_sampling_param_field_names_for_server_args",
            return_value=frozenset(),
        ):
            self.assertEqual(
                _case_warmup_sampling_params(case),
                {"num_frames": 48},
            )

    def test_waits_for_health_after_http_startup(self):
        manager = ServerManager("test-model", port=11000, wait_deadline=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout_path = Path(temp_dir) / "server.log"
            stdout_path.write_text("Application startup complete.\n", encoding="utf-8")
            with (
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.urlopen",
                    side_effect=[URLError("warming up"), _ReadyResponse()],
                ) as health_request,
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.time.sleep"
                ),
            ):
                manager._wait_for_ready(_RunningProcess(), stdout_path)

        self.assertEqual(health_request.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in health_request.call_args_list],
            ["http://127.0.0.1:11000/health"] * 2,
        )

    def test_start_preserves_quoted_args_and_cleans_up_on_readiness_failure(self):
        warmup = '{"height":720,"width":1280}'
        manager = ServerManager(
            "test-model",
            port=11000,
            wait_deadline=1,
            extra_args=f"--warmup-sampling-params '{warmup}'",
        )
        process = SimpleNamespace(pid=123, stdout=None)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            with (
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.prepare_perf_log",
                    return_value=(log_dir, log_dir / "perf.jsonl"),
                ),
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.subprocess.Popen",
                    return_value=process,
                ) as subprocess_popen,
                mock.patch.object(
                    manager,
                    "_wait_for_ready",
                    side_effect=TimeoutError("startup timed out"),
                ),
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.kill_process_tree"
                ) as kill_process,
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.time.sleep"
                ),
            ):
                with self.assertRaisesRegex(TimeoutError, "startup timed out"):
                    manager.start()

        kill_process.assert_called_once_with(123)
        command = subprocess_popen.call_args.args[0]
        self.assertEqual(command[command.index("--warmup-sampling-params") + 1], warmup)


if __name__ == "__main__":
    unittest.main()
