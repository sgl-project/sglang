"""Unit tests for the /start_profile body construction in bench_serving.

These tests do not start a server. They poke the body builder directly and,
for the HTTP path, monkeypatch the aiohttp session used by
`async_request_profile` so we can capture the exact POST body for a few
representative flag combinations.
"""

import argparse
import asyncio
import unittest
from contextlib import asynccontextmanager
from unittest import mock

from sglang import bench_serving

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:  # pragma: no cover - test infra not installed
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        profile=True,
        profile_num_steps=None,
        profile_start_step=None,
        profile_activities=None,
        profile_by_stage=False,
        profile_stages=None,
        profile_merge_profiles=False,
        profile_output_dir=None,
        profile_prefix=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class _RecordingResponse:
    status = 200
    reason = "OK"

    async def text(self):
        return ""


class _RecordingSession:
    """Minimal aiohttp-shaped session that records POSTs."""

    def __init__(self, recorded):
        self._recorded = recorded

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, *, url, json):
        self._recorded.append({"url": url, "json": json})

        @asynccontextmanager
        async def _ctx():
            yield _RecordingResponse()

        return _ctx()


class TestBuildStartProfileBody(CustomTestCase):
    def setUp(self):
        # Make sure no leftover SGLANG_TORCH_PROFILER_DIR sneaks output_dir
        # into bodies we expect to be empty.
        self._env_patch = mock.patch.dict("os.environ", {}, clear=False)
        self._env_patch.start()
        import os

        os.environ.pop("SGLANG_TORCH_PROFILER_DIR", None)

    def tearDown(self):
        self._env_patch.stop()

    def test_profile_only_yields_empty_body(self):
        bench_serving.args = _make_args()
        body = bench_serving._build_start_profile_body()
        # Either {} or {"output_dir": ...} per the task spec. With no env var
        # and no --profile-output-dir set, it must be exactly {}.
        self.assertEqual(body, {})

    def test_num_steps_only(self):
        bench_serving.args = _make_args(profile_num_steps=5)
        body = bench_serving._build_start_profile_body()
        self.assertEqual(body, {"num_steps": 5})

    def test_num_steps_and_start_step(self):
        bench_serving.args = _make_args(profile_num_steps=5, profile_start_step=3)
        body = bench_serving._build_start_profile_body()
        self.assertEqual(body, {"num_steps": 5, "start_step": 3})

    def test_activities_only(self):
        bench_serving.args = _make_args(profile_activities=["CPU", "GPU"])
        body = bench_serving._build_start_profile_body()
        self.assertEqual(body, {"activities": ["CPU", "GPU"]})

    def test_merge_profiles_flag(self):
        bench_serving.args = _make_args(profile_merge_profiles=True)
        body = bench_serving._build_start_profile_body()
        self.assertEqual(body, {"merge_profiles": True})

    def test_profile_by_stage_defaults_num_steps(self):
        bench_serving.args = _make_args(profile_by_stage=True)
        body = bench_serving._build_start_profile_body()
        self.assertEqual(body["profile_by_stage"], True)
        self.assertEqual(body["num_steps"], 5)


class TestAsyncRequestProfile(CustomTestCase):
    """Capture the POST body that actually reaches the HTTP layer."""

    def _run(self, args_ns, api_url):
        recorded = []

        def fake_session_factory():
            return _RecordingSession(recorded)

        bench_serving.args = args_ns
        with mock.patch.object(
            bench_serving, "_create_bench_client_session", fake_session_factory
        ):
            asyncio.run(bench_serving.async_request_profile(api_url=api_url))
        return recorded

    def test_start_profile_only_profile(self):
        recorded = self._run(_make_args(), "http://x/start_profile")
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0]["url"], "http://x/start_profile")
        # output_dir may or may not be present depending on env; check the
        # other keys are absent.
        body = recorded[0]["json"]
        for forbidden in (
            "num_steps",
            "start_step",
            "activities",
            "merge_profiles",
            "profile_by_stage",
            "profile_stages",
            "profile_prefix",
        ):
            self.assertNotIn(forbidden, body)

    def test_start_profile_num_steps(self):
        recorded = self._run(_make_args(profile_num_steps=5), "http://x/start_profile")
        self.assertEqual(recorded[0]["json"]["num_steps"], 5)

    def test_stop_profile_body_is_empty(self):
        recorded = self._run(_make_args(profile_num_steps=5), "http://x/stop_profile")
        self.assertEqual(recorded[0]["json"], {})


if __name__ == "__main__":
    unittest.main()
