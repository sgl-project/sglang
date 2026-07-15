"""
Unit tests for _get_fastapi_request_path (Prometheus middleware path resolver).

Regression guard for the FastAPI >= 0.137.2 behaviour where include_router()
keeps a route tree, so app.routes may contain _IncludedRouter nodes that have
no .path. Reading route.path on those raised AttributeError and made every
router-included endpoint (e.g. /v1/loads) return HTTP 500 when --enable-metrics
was set.

Usage:
    python3 -m pytest test/registered/unit/utils/test_get_fastapi_request_path.py -v
"""

import unittest

from fastapi import APIRouter, FastAPI

from sglang.srt.utils.common import _get_fastapi_request_path
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeURL:
    def __init__(self, path: str):
        self.path = path


class _FakeRequest:
    """Minimal stand-in exposing only what _get_fastapi_request_path reads."""

    def __init__(self, app: FastAPI, path: str, method: str = "GET"):
        self.app = app
        self.url = _FakeURL(path)
        self.scope = {
            "type": "http",
            "method": method,
            "path": path,
            "root_path": "",
            "headers": [],
            "path_params": {},
        }


def _build_app() -> FastAPI:
    app = FastAPI()

    # Directly registered route -> a plain APIRoute (has .path).
    @app.get("/get_load")
    def _get_load():
        return {}

    # Router-included routes -> reach the app via an _IncludedRouter node on
    # FastAPI >= 0.137.2. This is the case that used to crash.
    router = APIRouter()

    @router.get("/v1/loads")
    def _v1_loads():
        return {}

    @router.get("/v1/items/{item_id}")
    def _v1_items(item_id: str):
        return {}

    app.include_router(router)

    # Prefixed include -> the leaf's own path lacks the prefix, so the fix must
    # match against the fully-qualified template path.
    prefixed = APIRouter()

    @prefixed.get("/things/{tid}")
    def _things(tid: str):
        return {}

    app.include_router(prefixed, prefix="/api")

    return app


class TestGetFastapiRequestPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = _build_app()

    def _path(self, request_path: str, method: str = "GET") -> str:
        resolved, _handled = _get_fastapi_request_path(
            _FakeRequest(self.app, request_path, method)
        )
        return resolved

    def test_directly_registered_route(self):
        self.assertEqual(self._path("/get_load"), "/get_load")

    def test_router_included_route_does_not_crash(self):
        # Regression: this raised AttributeError on _IncludedRouter before the fix.
        self.assertEqual(self._path("/v1/loads"), "/v1/loads")

    def test_router_included_parameterized_route_keeps_template(self):
        self.assertEqual(self._path("/v1/items/123"), "/v1/items/{item_id}")

    def test_prefixed_included_route_keeps_full_template(self):
        self.assertEqual(self._path("/api/things/9"), "/api/things/{tid}")

    def test_unmatched_path_falls_back_to_request_url(self):
        self.assertEqual(self._path("/no/such/route"), "/no/such/route")


if __name__ == "__main__":
    unittest.main()
