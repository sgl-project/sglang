"""
Manual tests for the Atlas Cloud backend.

Requires ATLASCLOUD_API_KEY to be set in the environment.

Run all tests:
    python3 -m unittest test/manual/test_atlascloud_backend.py

Run a single test:
    python3 -m unittest test_atlascloud_backend.TestAtlasCloudBackend.test_mt_bench
"""

import unittest

from sglang import AtlasCloud, set_default_backend
from sglang.test.test_programs import (
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_stream,
)
from sglang.test.test_utils import CustomTestCase

# Default model available on Atlas Cloud.
DEFAULT_ATLASCLOUD_MODEL = "deepseek-ai/DeepSeek-V3.1-Terminus"


class TestAtlasCloudBackend(CustomTestCase):
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.backend = AtlasCloud(DEFAULT_ATLASCLOUD_MODEL)

    def setUp(self):
        set_default_backend(self.backend)

    def test_mt_bench(self):
        test_mt_bench()

    def test_stream(self):
        test_stream()

    def test_parallel_decoding(self):
        test_parallel_decoding()

    def test_parallel_encoding(self):
        test_parallel_encoding()


class TestAtlasCloudBackendInit(CustomTestCase):
    """Unit tests for Atlas Cloud backend initialisation — no network required."""

    def test_raises_without_api_key(self):
        import os

        key = os.environ.pop("ATLASCLOUD_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                AtlasCloud(DEFAULT_ATLASCLOUD_MODEL, api_key=None)
        finally:
            if key is not None:
                os.environ["ATLASCLOUD_API_KEY"] = key

    def test_accepts_explicit_api_key(self):
        backend = AtlasCloud(DEFAULT_ATLASCLOUD_MODEL, api_key="test-key")
        self.assertIsNotNone(backend)

    def test_custom_base_url(self):
        backend = AtlasCloud(
            DEFAULT_ATLASCLOUD_MODEL,
            api_key="test-key",
            base_url="https://api.atlascloud.ai/v1",
        )
        self.assertIsNotNone(backend)


if __name__ == "__main__":
    unittest.main()
