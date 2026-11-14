"""Download cache location regressions; no network, models, or accelerator."""

import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

from sglang import utils
from sglang.benchmark import utils as benchmark_utils
from sglang.benchmark.datasets import mooncake
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RECORD = {"timestamp": 1, "hash_ids": [2], "input_length": 4, "output_length": 1}
PAYLOAD = (json.dumps(RECORD) + "\n").encode()


class _TempdirTestCase(CustomTestCase):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.cache = self.root / "cache with spaces"
        self.cache.mkdir()

        env = {
            k: v for k, v in os.environ.items() if k not in ("TMPDIR", "TEMP", "TMP")
        }
        env["TMPDIR"] = str(self.cache)
        environ_patch = mock.patch.dict(os.environ, env, clear=True)
        environ_patch.start()
        self.addCleanup(environ_patch.stop)

        # Exercise real environment selection, restoring tempfile's cached value.
        tempdir_patch = mock.patch.object(tempfile, "tempdir", None)
        tempdir_patch.start()
        self.addCleanup(tempdir_patch.stop)

        response = mock.Mock(headers={"content-length": str(len(PAYLOAD))})
        response.iter_content.return_value = [PAYLOAD]
        http_patch = mock.patch("requests.get", return_value=response)
        self.http_get = http_patch.start()
        self.addCleanup(http_patch.stop)

    def unique_name(self, suffix):
        name = f"sglang-cache-{uuid.uuid4().hex}{suffix}"
        # Also clean up the old hardcoded destination when testing unpatched code.
        self.addCleanup((Path("/tmp") / name).unlink, missing_ok=True)
        return name

    def use_platform_default(self):
        for variable in ("TMPDIR", "TEMP", "TMP"):
            os.environ.pop(variable, None)
        tempfile.tempdir = None
        return Path(tempfile.gettempdir())


class TestDownloadCacheTempdir(_TempdirTestCase):
    def check_downloads(self, directory, *, cached=False, explicit=False):
        for module in (utils, benchmark_utils):
            with self.subTest(module=module.__name__):
                name = self.unique_name(".json")
                expected = directory / name
                self.addCleanup(expected.unlink, missing_ok=True)
                content = b'{"cached": true}\n' if cached else PAYLOAD
                if cached:
                    expected.write_bytes(content)
                self.http_get.reset_mock()

                url = f"https://example.com/{name}"
                filename = str(expected) if explicit else None
                actual = module.download_and_cache_file(url, filename)

                self.assertEqual(Path(actual), expected)
                self.assertEqual(expected.read_bytes(), content)
                self.assertEqual(module.download_and_cache_file(url, filename), actual)
                self.assertEqual(self.http_get.call_count, 0 if cached else 1)

    def test_download_uses_tmpdir(self):
        self.check_downloads(self.cache)

    def test_reuses_existing_tmpdir_cache(self):
        self.check_downloads(self.cache, cached=True)

    def test_explicit_filename_overrides_tmpdir(self):
        self.check_downloads(self.root, explicit=True)

    def test_platform_default_without_temp_environment(self):
        self.check_downloads(self.use_platform_default())


class TestMooncakeCacheTempdir(_TempdirTestCase):
    def check_dataset(self, directory, *, cached=False, explicit=False):
        name = self.unique_name("_trace.jsonl")
        workload = name.removesuffix("_trace.jsonl")
        expected = directory / name
        self.addCleanup(expected.unlink, missing_ok=True)
        record = {**RECORD, "timestamp": 99} if cached else RECORD
        if cached:
            expected.write_text(json.dumps(record) + "\n")

        with mock.patch.dict(
            mooncake.MOONCAKE_DATASET_URL,
            {workload: "https://example.com/trace.jsonl"},
        ):
            dataset = mooncake.MooncakeDataset(
                dataset_path=str(expected) if explicit else "",
                mooncake_workload=workload,
                num_requests=1,
            )
            self.assertEqual(dataset.load(), [record])
            self.assertEqual(json.loads(expected.read_text()), record)
            self.assertEqual(dataset.load(), [record])
            self.assertEqual(self.http_get.call_count, 0 if cached else 1)

    def test_download_uses_tmpdir(self):
        self.check_dataset(self.cache)

    def test_reuses_existing_tmpdir_cache(self):
        self.check_dataset(self.cache, cached=True)

    def test_explicit_dataset_path_overrides_tmpdir(self):
        self.check_dataset(self.root, explicit=True)

    def test_platform_default_without_temp_environment(self):
        self.check_dataset(self.use_platform_default())


if __name__ == "__main__":
    unittest.main()
