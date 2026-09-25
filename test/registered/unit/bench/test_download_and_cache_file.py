"""Unit tests for download_and_cache_file in sglang/benchmark/utils.py"""

import os
import tempfile
import unittest
from unittest import mock

import requests

from sglang.benchmark.datasets.mooncake import MooncakeDataset
from sglang.benchmark.utils import download_and_cache_file
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_RECORD = b'{"timestamp": 1, "hash_ids": []}\n'


class _Response:
    """Streams `chunks`, then raises `error` mid-body if one is given."""

    def __init__(self, chunks, error=None):
        self._chunks = chunks
        self._error = error
        self.headers = {"content-length": str(sum(len(c) for c in chunks) + 100)}

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield from self._chunks
        if self._error is not None:
            raise self._error


class TestDownloadAndCacheFile(CustomTestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._dir.name, "trace.jsonl")

    def tearDown(self):
        self._dir.cleanup()

    def test_interrupted_download_leaves_no_file_at_the_destination(self):
        """A stream that fails after some chunks must leave neither the destination
        nor a temporary file behind."""
        interrupted = _Response(
            [_RECORD], requests.exceptions.ConnectionError("interrupted")
        )
        with mock.patch(
            "sglang.benchmark.utils.requests.get", return_value=interrupted
        ):
            with self.assertRaises(requests.exceptions.ConnectionError):
                download_and_cache_file("https://example.com/trace.jsonl", self.path)

        self.assertEqual(os.listdir(self._dir.name), [])

    def test_mooncake_load_retries_after_an_interrupted_download(self):
        """After a failed download, the next load must fetch the trace again
        instead of reading the partial file as the whole dataset."""
        dataset = MooncakeDataset(self.path, "conversation", 10)
        responses = [
            _Response([_RECORD], requests.exceptions.ConnectionError("interrupted")),
            _Response([_RECORD, _RECORD, _RECORD]),
        ]
        with mock.patch(
            "sglang.benchmark.utils.requests.get", side_effect=responses
        ) as get:
            with self.assertRaises(requests.exceptions.ConnectionError):
                dataset.load()
            rows = dataset.load()

        self.assertEqual(get.call_count, 2)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
