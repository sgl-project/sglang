"""
Manual tests for the Crusoe managed inference backend.

Requires CRUSOE_API_KEY to be set in the environment.

Run all tests:
    python3 -m unittest test/manual/test_crusoe_backend.py

Run a single test:
    python3 -m unittest test_crusoe_backend.TestCrusoeBackend.test_mt_bench
"""

import unittest

from sglang import Crusoe, set_default_backend
from sglang.test.test_programs import (
    test_mt_bench,
    test_parallel_decoding,
    test_parallel_encoding,
    test_stream,
)
from sglang.test.test_utils import CustomTestCase

# Default model available on Crusoe managed inference.
DEFAULT_CRUSOE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"


class TestCrusoeBackend(CustomTestCase):
    backend = None

    @classmethod
    def setUpClass(cls):
        cls.backend = Crusoe(DEFAULT_CRUSOE_MODEL)

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


class TestCrusoeBackendInit(CustomTestCase):
    """Unit tests for Crusoe backend initialisation — no network required."""

    def test_raises_without_api_key(self):
        import os

        key = os.environ.pop("CRUSOE_API_KEY", None)
        try:
            with self.assertRaises(ValueError):
                Crusoe(DEFAULT_CRUSOE_MODEL, api_key=None)
        finally:
            if key is not None:
                os.environ["CRUSOE_API_KEY"] = key

    def test_accepts_explicit_api_key(self):
        backend = Crusoe(DEFAULT_CRUSOE_MODEL, api_key="test-key")
        self.assertIsNotNone(backend)

    def test_custom_base_url(self):
        backend = Crusoe(
            DEFAULT_CRUSOE_MODEL,
            api_key="test-key",
            base_url="https://managed-inference-api-proxy.crusoecloud.com/v1/",
        )
        self.assertIsNotNone(backend)


if __name__ == "__main__":
    unittest.main()
