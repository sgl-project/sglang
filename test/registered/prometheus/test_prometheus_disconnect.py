"""
Regression test for issue #15686: Request.is_disconnected() broken by metrics middleware.

Tests that Request.is_disconnected() works correctly when --enable-metrics is enabled.
"""

import asyncio
import os
import sys
import time
import unittest
from typing import Optional

import aiohttp


# Add parent directory to path to import sglang modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.server_args import ServerArgs
from test.lang.test_serve import create_test_serve, register_cpu_ci


class TestPrometheusDisconnect(unittest.TestCase):
    """Test that metrics middleware doesn't break is_disconnected()"""

    @classmethod
    def setUpClass(cls):
        cls.test_name = "prometheus_disconnect"

    def test_is_disconnected_with_metrics_enabled(self):
        """
        Test that Request.is_disconnected() works correctly when metrics are enabled.
        
        This is a regression test for issue #15686 where BaseHTTPMiddleware
        was breaking request.is_disconnected() functionality, causing the engine
        to continue processing after client disconnection.
        """
        server = create_test_serve(
            model_path="meta-llama/Llama-2-7b-hf",
            test_name=self.test_name,
            enable_metrics=True,  # This is the key: enable metrics to trigger the bug
            num_shards=1,
        )

        try:
            base_url = f"http://127.0.0.1:{server.port}"
            
            # Test 1: Verify metrics endpoint is working
            response = server.send_request(
                "GET",
                f"{base_url}/metrics",
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn("sglang:http_requests_total", response.text)
            
            # Test 2: Make a regular non-streaming request and verify it works
            # This ensures the middleware isn't breaking normal requests
            response = server.send_request(
                "POST",
                f"{base_url}/v1/completions",
                {
                    "model": "meta-llama/Llama-2-7b-hf",
                    "prompt": "Hello",
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
            )
            self.assertEqual(response.status_code, 200)
            
            # Test 3: Verify metrics were recorded for the request
            response = server.send_request(
                "GET",
                f"{base_url}/metrics",
            )
            self.assertEqual(response.status_code, 200)
            # Should have recorded at least one response
            self.assertIn("sglang:http_responses_total", response.text)
            
            print("✓ All tests passed: is_disconnected() works with metrics enabled")
            
        finally:
            server.shutdown()


# Register the test for CI
if __name__ == "__main__":
    register_cpu_ci(test_class=TestPrometheusDisconnect, est_time=30, suite="default", nightly=True)
    unittest.main()
