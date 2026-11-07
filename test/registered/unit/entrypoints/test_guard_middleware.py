# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Unit tests for the optional fastapi-guard middleware wiring."""

import asyncio
import itertools
import os
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# Rate limiting and ban state in fastapi-guard are process-wide, so every
# test drives its own TEST-NET client IP to stay hermetic.
_IP = itertools.count(1)

try:
    import guard  # noqa: F401

    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False


def _unique_ip() -> str:
    n = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


class TestGuardIntegration(CustomTestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.pop(k)
            for k in list(os.environ)
            if k.startswith("SGLANG_GUARD_")
        }

    def tearDown(self):
        for name in list(os.environ):
            if name.startswith("SGLANG_GUARD_"):
                os.environ.pop(name, None)
        os.environ.update(self._saved)

    def _build_app(self):
        from fastapi import FastAPI

        from sglang.srt.entrypoints.guard_integration import attach_guard

        app = FastAPI()

        @app.get("/ping")
        async def ping():
            return {"ok": True}

        attach_guard(app)
        return app

    async def _get(self, client_ip):
        from httpx import ASGITransport, AsyncClient

        transport = ASGITransport(app=self._build_app(), client=(client_ip, 50000))
        async with AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            return (await client.get("/ping")).status_code

    def test_disabled_by_default(self):
        from fastapi import FastAPI

        from sglang.srt.entrypoints.guard_integration import attach_guard

        app = FastAPI()
        attach_guard(app)
        self.assertEqual(len(app.user_middleware), 0)

    def test_fail_loud_without_package(self):
        from fastapi import FastAPI

        from sglang.srt.entrypoints.guard_integration import attach_guard

        if _GUARD_AVAILABLE:
            self.skipTest("fastapi-guard is installed")
        os.environ["SGLANG_GUARD_ENABLED"] = "true"
        with self.assertRaisesRegex(RuntimeError, r"sglang\[guard\]"):
            attach_guard(FastAPI())

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_blocked_ip_is_rejected(self):
        blocked = _unique_ip()
        os.environ["SGLANG_GUARD_ENABLED"] = "true"
        os.environ["SGLANG_GUARD_BLOCKED_IPS"] = blocked
        self.assertEqual(asyncio.run(self._get(blocked)), 403)

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_rate_limit_returns_429(self):
        client_ip = _unique_ip()
        os.environ["SGLANG_GUARD_ENABLED"] = "true"
        os.environ["SGLANG_GUARD_RATE_LIMIT"] = "2"
        os.environ["SGLANG_GUARD_RATE_LIMIT_WINDOW"] = "60"
        codes = [asyncio.run(self._get(client_ip)) for _ in range(3)]
        self.assertEqual(codes, [200, 200, 429])

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_passive_mode_never_blocks(self):
        blocked = _unique_ip()
        os.environ["SGLANG_GUARD_ENABLED"] = "true"
        os.environ["SGLANG_GUARD_PASSIVE_MODE"] = "true"
        os.environ["SGLANG_GUARD_BLOCKED_IPS"] = blocked
        self.assertEqual(asyncio.run(self._get(blocked)), 200)

    @unittest.skipUnless(_GUARD_AVAILABLE, "fastapi-guard not installed")
    def test_redis_stays_off_without_config(self):
        from sglang.srt.entrypoints.guard_integration import _build_guard_config

        os.environ["SGLANG_GUARD_ENABLED"] = "true"
        config = _build_guard_config()
        self.assertFalse(config.enable_redis)


if __name__ == "__main__":
    unittest.main()
