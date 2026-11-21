"""Download caches must follow the configured temporary directory (#50351)."""

import json
import os
import tempfile
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from sglang import utils
from sglang.benchmark import utils as benchmark_utils
from sglang.benchmark.datasets.mooncake import MOONCAKE_DATASET_URL, MooncakeDataset
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_RECORD = {"timestamp": 0, "input_length": 8, "output_length": 4}
_PAYLOAD = (json.dumps(_RECORD) + "\n").encode()


class _DownloadHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(_PAYLOAD)))
        self.end_headers()
        self.wfile.write(_PAYLOAD)

    def log_message(self, *args):
        pass


class TestDownloadCache(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _DownloadHandler)
        cls.addClassCleanup(cls.server.server_close)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.addClassCleanup(cls.thread.join)
        cls.addClassCleanup(cls.server.shutdown)
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    def setUp(self):
        self.default_directory = Path(tempfile.gettempdir())
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.enterContext(patch.dict(os.environ, {"TMPDIR": str(self.root)}))
        # tempfile caches the selected directory; restore it after each test.
        self.enterContext(patch.object(tempfile, "tempdir", None))

    def test_download_helpers_use_configured_tempdir(self):
        for module in (utils, benchmark_utils):
            for variable in ("TMPDIR", "TEMP", "TMP"):
                with self.subTest(module=module.__name__, variable=variable):
                    name = f"sglang-{uuid.uuid4().hex}.json"
                    expected = self.root / name
                    with (
                        patch.dict(
                            os.environ,
                            {
                                "TMPDIR": "",
                                "TEMP": "",
                                "TMP": "",
                                variable: str(self.root),
                            },
                        ),
                        patch.object(tempfile, "tempdir", None),
                    ):
                        result = module.download_and_cache_file(
                            f"{self.base_url}/{name}"
                        )
                    self.addCleanup(Path(result).unlink, missing_ok=True)
                    self.assertEqual(Path(result), expected)
                    self.assertEqual(expected.read_bytes(), _PAYLOAD)

    def test_download_helpers_reuse_configured_cache(self):
        for module in (utils, benchmark_utils):
            with self.subTest(module=module.__name__):
                name = f"sglang-{uuid.uuid4().hex}.json"
                expected = self.root / name
                expected.write_bytes(_PAYLOAD)
                with patch("requests.get", side_effect=AssertionError("cache miss")):
                    result = module.download_and_cache_file(f"{self.base_url}/{name}")
                self.assertEqual(Path(result), expected)
                self.assertEqual(expected.read_bytes(), _PAYLOAD)

    def test_download_helpers_preserve_explicit_filename(self):
        for module in (utils, benchmark_utils):
            with self.subTest(module=module.__name__):
                expected = self.root / f"explicit-{uuid.uuid4().hex}.json"
                result = module.download_and_cache_file(
                    f"{self.base_url}/source.json", str(expected)
                )
                self.assertEqual(Path(result), expected)
                self.assertEqual(expected.read_bytes(), _PAYLOAD)

    def test_download_helpers_use_platform_default_without_environment(self):
        with (
            patch.dict(os.environ, {"TMPDIR": "", "TEMP": "", "TMP": ""}),
            patch.object(tempfile, "tempdir", None),
        ):
            default_directory = Path(tempfile.gettempdir())
            for module in (utils, benchmark_utils):
                with self.subTest(module=module.__name__):
                    name = f"sglang-{uuid.uuid4().hex}.json"
                    result = module.download_and_cache_file(f"{self.base_url}/{name}")
                    self.addCleanup(Path(result).unlink, missing_ok=True)
                    self.assertEqual(Path(result).parent, default_directory)
                    self.assertEqual(Path(result).read_bytes(), _PAYLOAD)

    def test_mooncake_download_uses_configured_tempdir(self):
        workload = f"sglang-{uuid.uuid4().hex}"
        expected = self.root / f"{workload}_trace.jsonl"
        self.addCleanup(
            (self.default_directory / expected.name).unlink, missing_ok=True
        )
        with patch.dict(
            MOONCAKE_DATASET_URL, {workload: f"{self.base_url}/trace.jsonl"}
        ):
            result = MooncakeDataset("", workload, 1).load()
        self.assertEqual(result, [_RECORD])
        self.assertEqual(expected.read_bytes(), _PAYLOAD)

    def test_mooncake_reuses_configured_cache(self):
        workload = f"sglang-{uuid.uuid4().hex}"
        expected = self.root / f"{workload}_trace.jsonl"
        expected.write_bytes(_PAYLOAD)
        with (
            patch.dict(
                MOONCAKE_DATASET_URL, {workload: f"{self.base_url}/trace.jsonl"}
            ),
            patch("requests.get", side_effect=AssertionError("cache miss")),
        ):
            result = MooncakeDataset("", workload, 1).load()
        self.assertEqual(result, [_RECORD])
        self.assertEqual(expected.read_bytes(), _PAYLOAD)

    def test_mooncake_preserves_explicit_dataset_path(self):
        expected = self.root / "explicit.jsonl"
        expected.write_bytes(_PAYLOAD)
        with patch("requests.get", side_effect=AssertionError("unexpected download")):
            result = MooncakeDataset(str(expected), "synthetic", 1).load()
        self.assertEqual(result, [_RECORD])


if __name__ == "__main__":
    unittest.main()
