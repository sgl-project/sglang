"""Tests for the SystemOne example server."""

import importlib.util
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except ModuleNotFoundError:
    CustomTestCase = unittest.TestCase

    def register_cpu_ci(*args, **kwargs):
        pass


register_cpu_ci(est_time=1, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
SERVER_PATH = REPO_ROOT / "examples" / "runtime" / "systemone" / "systemone_server.py"


def _load_server():
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    previous = sys.modules.get("transformers")
    sys.modules["transformers"] = transformers
    spec = importlib.util.spec_from_file_location("_systemone_server", SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("transformers", None)
        else:
            sys.modules["transformers"] = previous
    return module


class _Response:
    def __init__(self, scores):
        self.scores = scores

    def raise_for_status(self):
        pass

    def json(self):
        return {"scores": self.scores}


class TestSystemOneServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _load_server()

    def test_mismatched_score_shape_returns_502(self):
        body = {
            "state": "state",
            "questions": {
                "first": {"type": "noul", "instructions": "first"},
                "second": {"type": "noul", "instructions": "second"},
            },
        }
        data = json.dumps(body).encode()

        for scores in ([[0.5, 0.5]], [[1.0], [0.5, 0.5]]):
            with self.subTest(scores=scores):
                engine = self.server.SystemOne.__new__(self.server.SystemOne)
                engine.upstream = "http://upstream"
                engine.model = "model"
                engine.temperature = 1.0
                engine.session = Mock()
                engine.session.post.return_value = _Response(scores)
                engine.encode = Mock(return_value=[1])
                engine.label_token = lambda label: ord(label[0])

                handler_type = self.server.make_handler(engine)
                handler = handler_type.__new__(handler_type)
                handler.path = "/v1/systemone"
                handler.headers = {"Content-Length": str(len(data))}
                handler.rfile = io.BytesIO(data)
                handler._json = Mock()

                handler.do_POST()

                code, payload = handler._json.call_args.args
                self.assertEqual(code, 502)
                self.assertEqual(payload["error"]["type"], "upstream_error")


if __name__ == "__main__":
    unittest.main()
