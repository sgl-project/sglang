"""HttpServerEngineAdapter must expose the RL control calls the HTTP server offers."""

import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server_engine
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.entrypoints.EngineBase import EngineBase
from sglang.srt.entrypoints.http_server_engine import HttpServerEngineAdapter


def _adapter():
    # skip __init__: it would launch a server
    adapter = object.__new__(HttpServerEngineAdapter)
    adapter.server_args = SimpleNamespace(url=lambda: "http://host:1")
    return adapter


class TestHttpServerEngineAdapterControlCalls(CustomTestCase):
    def _call(self, method, *args, **kwargs):
        response = mock.Mock()
        response.json.return_value = {}
        with (
            mock.patch.object(
                http_server_engine.requests, "post", return_value=response
            ) as post,
            mock.patch.object(http_server_engine, "ssl_verify_of", return_value=True),
        ):
            getattr(_adapter(), method)(*args, **kwargs)
        return post.call_args

    def test_release_resume_forward_tags(self):
        for method in ("release_memory_occupation", "resume_memory_occupation"):
            call = self._call(method, tags=["kv_cache"])
            self.assertEqual(call.args[0], f"http://host:1/{method}")
            self.assertEqual(call.kwargs["json"], {"tags": ["kv_cache"]})

    def test_release_resume_default_is_all(self):
        for method in ("release_memory_occupation", "resume_memory_occupation"):
            self.assertEqual(self._call(method).kwargs["json"], {"tags": None})

    def test_pause_continue(self):
        call = self._call("pause_generation", mode="in_place")
        self.assertEqual(call.args[0], "http://host:1/pause_generation")
        self.assertEqual(call.kwargs["json"], {"mode": "in_place"})
        call = self._call("continue_generation", torch_empty_cache=False)
        self.assertEqual(call.args[0], "http://host:1/continue_generation")
        self.assertEqual(call.kwargs["json"], {"torch_empty_cache": False})

    def test_signatures_match_engine(self):
        # the adapter and EngineBase must accept whatever Engine accepts
        for name in ("release_memory_occupation", "resume_memory_occupation"):
            expected = list(inspect.signature(getattr(Engine, name)).parameters)
            for cls in (EngineBase, HttpServerEngineAdapter):
                self.assertEqual(
                    list(inspect.signature(getattr(cls, name)).parameters), expected
                )


if __name__ == "__main__":
    unittest.main()
