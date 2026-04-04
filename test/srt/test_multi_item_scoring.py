import os
import signal
import subprocess
import tempfile
import time
import unittest

import requests

from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


class TestMultiItemScoringServer(CustomTestCase):
    """Test multi-item scoring functionality through the server API."""

    def setUp(self):
        """Set up each test case."""
        self.model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        self.port = 30001  # Use different port to avoid conflicts
        self.host = "localhost"
        self.base_url = f"http://{self.host}:{self.port}"
        self.server_process = None
        self.server_log_file = None

    def tearDown(self):
        """Clean up after each test case."""
        self.stop_server()

    def start_server(self, multi_item_scoring_delimiter=None):
        """Start the SGLang server with multi-item scoring enabled."""
        if self.server_process is not None:
            self.stop_server()

        # Create a temporary log file
        self.server_log_file = tempfile.NamedTemporaryFile(mode="w+", delete=False)

        cmd = [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--port",
            str(self.port),
            "--host",
            self.host,
            "--chunked-prefill-size",
            "-1",
            "--dtype",
            "float16",
            "--max-prefill-tokens",
            "30000",
            "--mem-fraction-static",
            "0.3",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--attention-backend",
            "flashinfer",
        ]

        if multi_item_scoring_delimiter is not None:
            cmd.extend(
                ["--multi-item-scoring-delimiter", str(multi_item_scoring_delimiter)]
            )

        # Start server process
        self.server_process = subprocess.Popen(
            cmd,
            stdout=self.server_log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        # Wait for server to start
        max_wait_time = 60  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(f"{self.base_url}/get_model_info", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)

        # If we get here, server didn't start properly
        self.stop_server()
        raise RuntimeError("Failed to start SGLang server within timeout period")

    def stop_server(self):
        """Stop the SGLang server."""
        if self.server_process is not None:
            try:
                # Kill the process group
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                else:
                    self.server_process.terminate()

                # Wait for graceful shutdown
                try:
                    self.server_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    if hasattr(os, "killpg"):
                        os.killpg(os.getpgid(self.server_process.pid), signal.SIGKILL)
                    else:
                        self.server_process.kill()
                    self.server_process.wait()
            except (ProcessLookupError, OSError):
                # Process already terminated
                pass
            finally:
                self.server_process = None

        if self.server_log_file is not None:
            try:
                self.server_log_file.close()
                os.unlink(self.server_log_file.name)
            except (OSError, FileNotFoundError):
                pass
            finally:
                self.server_log_file = None

    def get_server_logs(self):
        """Get server logs for debugging."""
        if self.server_log_file is not None:
            try:
                with open(self.server_log_file.name, "r") as f:
                    return f.read()
            except (OSError, FileNotFoundError):
                pass
        return "No logs available"

    def test_multi_item_scoring_server_basic(self):
        """Test basic multi-item scoring through server API."""
        # Start server with multi-item scoring enabled
        delimiter_token_id = 151655  # Example delimiter token ID
        self.start_server(multi_item_scoring_delimiter=delimiter_token_id)

        # Test data
        query = "What is the capital of California? Answer Yes or No for each of the following options:"
        items = ["Sacramento", "San Jose", "San Francisco"]
        label_token_ids = [9454, 2753]  # "Yes" and "No" tokens

        # Make scoring request
        payload = {
            "query": query,
            "items": items,
            "label_token_ids": label_token_ids,
            "model": self.model_path,
        }

        response = requests.post(
            f"{self.base_url}/v1/score",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        # Check response
        self.assertEqual(
            response.status_code,
            200,
            f"Server returned {response.status_code}. Logs: {self.get_server_logs()}",
        )

        result = response.json()

        # Verify response structure
        self.assertIn("scores", result)
        self.assertIn("model", result)
        self.assertIn("object", result)

        self.assertEqual(result["object"], "scoring")
        self.assertEqual(result["model"], self.model_path)

        # Verify scores
        scores = result["scores"]
        self.assertEqual(len(scores), len(items), "Should get one score list per item")

        for i, score_list in enumerate(scores):
            self.assertEqual(
                len(score_list),
                len(label_token_ids),
                f"Item {i} should have {len(label_token_ids)} scores",
            )
            # Verify scores are probabilities (sum to 1)
            self.assertAlmostEqual(
                sum(score_list),
                1.0,
                places=6,
                msg=f"Scores for item {i} should sum to 1",
            )
            # Verify all scores are non-negative
            for j, score in enumerate(score_list):
                self.assertGreaterEqual(
                    score, 0, f"Score {j} for item {i} should be non-negative"
                )

    def test_multi_item_scoring_server_different_sizes(self):
        """Test multi-item scoring with different numbers of items through server."""
        delimiter_token_id = 151655
        self.start_server(multi_item_scoring_delimiter=delimiter_token_id)

        query = "Rate each option:"
        label_token_ids = [1, 2, 3, 4, 5]

        test_cases = [
            ["Single item"],
            ["Item 1", "Item 2"],
            ["A", "B", "C", "D"],
            ["X", "Y", "Z", "W", "V", "U"],
        ]

        for items in test_cases:
            with self.subTest(items=items):
                payload = {
                    "query": query,
                    "items": items,
                    "label_token_ids": label_token_ids,
                    "model": self.model_path,
                }

                response = requests.post(
                    f"{self.base_url}/v1/score",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30,
                )

                self.assertEqual(
                    response.status_code,
                    200,
                    f"Failed for items {items}. Logs: {self.get_server_logs()}",
                )

                result = response.json()
                scores = result["scores"]

                self.assertEqual(
                    len(scores), len(items), f"Should get {len(items)} score lists"
                )

                for i, score_list in enumerate(scores):
                    self.assertEqual(len(score_list), len(label_token_ids))
                    self.assertAlmostEqual(sum(score_list), 1.0, places=6)

    def test_multi_item_scoring_server_empty_items(self):
        """Test multi-item scoring with empty items list through server."""
        delimiter_token_id = 151655
        self.start_server(multi_item_scoring_delimiter=delimiter_token_id)

        payload = {
            "query": "Test query",
            "items": [],
            "label_token_ids": [1, 2],
            "model": self.model_path,
        }

        response = requests.post(
            f"{self.base_url}/v1/score",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(
            len(result["scores"]), 0, "Should return empty list for empty items"
        )

    def test_multi_item_scoring_server_without_delimiter(self):
        """Test that server works without multi-item scoring delimiter."""
        # Start server without multi-item scoring delimiter
        self.start_server(multi_item_scoring_delimiter=None)

        payload = {
            "query": "Test query",
            "items": ["Item 1", "Item 2"],
            "label_token_ids": [1, 2],
            "model": self.model_path,
        }

        response = requests.post(
            f"{self.base_url}/v1/score",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )

        # Should still work (falls back to regular scoring)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("scores", result)

    def test_multi_item_scoring_server_error_handling(self):
        """Test error handling in multi-item scoring server API."""
        delimiter_token_id = 151655
        self.start_server(multi_item_scoring_delimiter=delimiter_token_id)

        # Test with invalid payload
        invalid_payloads = [
            {
                "query": "Test",
                "items": "not a list",
                "label_token_ids": [1, 2],
                "model": self.model_path,
            },
            {
                "query": "Test",
                "items": ["Item 1"],
                "label_token_ids": "not a list",
                "model": self.model_path,
            },
            {
                "query": "Test",
                "items": ["Item 1"],
                "label_token_ids": [1, 2],
            },  # Missing model
            {
                "items": ["Item 1"],
                "label_token_ids": [1, 2],
                "model": self.model_path,
            },  # Missing query
        ]

        for i, payload in enumerate(invalid_payloads):
            with self.subTest(payload=i):
                response = requests.post(
                    f"{self.base_url}/v1/score",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=30,
                )

                # Should return error status
                self.assertGreaterEqual(
                    response.status_code,
                    400,
                    f"Should return error for invalid payload {i}",
                )

    def test_multi_item_scoring_server_consistency(self):
        """Test that multi-item scoring gives consistent results through server."""
        delimiter_token_id = 151655
        self.start_server(multi_item_scoring_delimiter=delimiter_token_id)

        query = "Choose the best option:"
        items = ["Option A", "Option B", "Option C"]
        label_token_ids = [1, 2, 3]

        payload = {
            "query": query,
            "items": items,
            "label_token_ids": label_token_ids,
            "model": self.model_path,
        }

        # Run the same test multiple times
        scores1 = None
        scores2 = None

        for attempt in range(2):
            response = requests.post(
                f"{self.base_url}/v1/score",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30,
            )

            self.assertEqual(
                response.status_code,
                200,
                f"Attempt {attempt + 1} failed. Logs: {self.get_server_logs()}",
            )

            result = response.json()
            scores = result["scores"]

            if attempt == 0:
                scores1 = scores
            else:
                scores2 = scores

        # Results should be identical (deterministic)
        self.assertEqual(len(scores1), len(scores2), "Should get same number of items")
        for i, (s1, s2) in enumerate(zip(scores1, scores2)):
            self.assertEqual(
                len(s1), len(s2), f"Item {i} should have same number of scores"
            )
            for j, (score1, score2) in enumerate(zip(s1, s2)):
                self.assertAlmostEqual(
                    score1,
                    score2,
                    places=6,
                    msg=f"Score {j} for item {i} should be identical",
                )

    def test_multi_item_scoring_server_large_batch(self):
        """Test multi-item scoring with large batch through server."""
        delimiter_token_id = 151655
        self.start_server(multi_item_scoring_delimiter=delimiter_token_id)

        query = "Classify each item:"
        items = [f"Item {i}" for i in range(10)]  # 10 items (smaller for test)
        label_token_ids = [1, 2, 3]

        payload = {
            "query": query,
            "items": items,
            "label_token_ids": label_token_ids,
            "model": self.model_path,
        }

        response = requests.post(
            f"{self.base_url}/v1/score",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,  # Longer timeout for large batch
        )

        self.assertEqual(
            response.status_code,
            200,
            f"Large batch failed. Logs: {self.get_server_logs()}",
        )

        result = response.json()
        scores = result["scores"]

        self.assertEqual(len(scores), len(items), "Should handle large batches")

        for i, score_list in enumerate(scores):
            self.assertEqual(len(score_list), len(label_token_ids))
            self.assertAlmostEqual(sum(score_list), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
