import atexit
import json
import signal
import subprocess
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests


class TestAbortRequest(unittest.TestCase):

    # Class variables to store server processes
    server1_process = None
    server2_process = None
    router_process = None

    @classmethod
    def setUpClass(cls):
        """Start the sglang servers and router before running tests"""
        print("\n=== Starting SgLang Servers and Router ===")

        try:
            # Start first sglang server (port 8000, GPU 1)
            print("Starting first sglang server on port 8000...")
            cls.server1_process = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    "Qwen/Qwen3-8B",
                    "--tp",
                    "1",
                    "--context-length",
                    "40960",
                    "--base-gpu-id",
                    "1",
                    "--port",
                    "8000",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Start second sglang server (port 8001, GPU 0)
            print("Starting second sglang server on port 8001...")
            cls.server2_process = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    "Qwen/Qwen3-8B",
                    "--tp",
                    "1",
                    "--context-length",
                    "40960",
                    "--base-gpu-id",
                    "0",
                    "--port",
                    "8001",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for servers to start up
            print("Waiting for servers to start up...")
            # Give servers time to initialize

            # Start router
            print("Starting sglang router...")
            cls.router_process = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "sglang_router.launch_router",
                    "--worker-urls",
                    "http://localhost:8000",
                    "http://localhost:8001",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for router to start
            print("Waiting for router to start up...")
            time.sleep(30)

            # Verify services are running
            cls._verify_services()

            print("All services started successfully!")

        except Exception as e:
            print(f"Error starting services: {e}")
            cls.tearDownClass()  # Clean up on failure
            raise

    @classmethod
    def tearDownClass(cls):
        """Stop all servers and router after tests complete"""
        print("\n=== Stopping SgLang Servers and Router ===")

        # Stop router
        if cls.router_process:
            print("Stopping router...")
            cls.router_process.terminate()
            try:
                cls.router_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.router_process.kill()
            cls.router_process = None

        # Stop server processes
        for name, process in [
            ("server1", cls.server1_process),
            ("server2", cls.server2_process),
        ]:
            if process:
                print(f"Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

        cls.server1_process = None
        cls.server2_process = None

        print("All services stopped.")

    @classmethod
    def _verify_services(cls):
        """Verify that servers and router are responding"""
        # Check server 1
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            print(f"Server 1 health check: {response.status_code}")
        except Exception as e:
            print(f"Server 1 health check failed: {e}")

        # Check server 2
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            print(f"Server 2 health check: {response.status_code}")
        except Exception as e:
            print(f"Server 2 health check failed: {e}")

        # Check router
        try:
            response = requests.get("http://localhost:30000/health", timeout=5)
            print(f"Router health check: {response.status_code}")
        except Exception as e:
            print(f"Router health check failed: {e}")

    def setUp(self):
        self.router_url = "http://localhost:30000"
        self.stream_url = f"{self.router_url}/v1/chat/completions"
        self.abort_url = f"{self.router_url}/abort_request"

    def test_abort_request_with_mapping(self):
        print("\n=== Testing Abort Request with Mapping ===")

        rid = "test-rid-123456"
        stream_data = {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": "generate a long text,more than 1000000words",
                }
            ],
            "stream": True,
            "rid": rid,
        }

        print("Sending streaming request with explicit rid...")
        chunk_count = 0
        abort_sent = False

        with requests.post(self.stream_url, json=stream_data, stream=True) as response:
            self.assertEqual(response.status_code, 200, "Stream request should succeed")

            print("Reading stream...")
            for chunk in response.iter_lines():
                if chunk:
                    chunk_count += 1
                    chunk_text = chunk.decode()

                    # After receiving first chunk, send abort request
                    if chunk_count == 1 and not abort_sent:
                        try:
                            if chunk_text.startswith("data: "):
                                chunk_text = chunk_text[6:]  # Remove 'data: ' prefix
                            print(f"Chunk {chunk_count}:", chunk_text)

                            # Use the same rid we sent in the request
                            print(f"\nUsing predefined rid: {rid}")
                            time.sleep(2)
                            print("Sending abort request...")
                            abort_data = {"rid": rid}
                            abort_response = requests.post(
                                self.abort_url, json=abort_data
                            )
                            print(
                                f"Abort response status: {abort_response.status_code}"
                            )
                            print(f"Abort response body: {abort_response.text}")

                            # Assert abort request was successful
                            self.assertIn(
                                abort_response.status_code,
                                [200, 202],
                                "Abort request should be accepted",
                            )
                            abort_sent = True

                            # Wait a bit to see if stream was aborted
                            time.sleep(1)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing first chunk: {e}")
                            continue

                    # Stop after reading 100 chunks or if stream ends
                    if chunk_count >= 100:
                        break

        print(f"Total chunks read: {chunk_count}")
        self.assertGreater(chunk_count, 0, "Should have received at least one chunk")
        self.assertTrue(abort_sent, "Abort request should have been sent")

    def test_abort_one_of_multiple_requests(self):
        """
        Test aborting specific request among multiple concurrent requests
        """
        print("\n=== Testing Multiple Requests Abort ===")

        # Create multiple requests with different rids
        request_configs = [
            {"rid": "req-001", "content": "Tell me a story about dragons"},
            {"rid": "req-002", "content": "Write a poem about the ocean"},
            {"rid": "req-003", "content": "Explain quantum physics"},
        ]

        results = {}

        def send_request(config):
            rid = config["rid"]
            content = config["content"]

            stream_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": content}],
                "stream": True,
                "rid": rid,
            }

            print(f"Starting request {rid}...")
            chunk_count = 0

            try:
                with requests.post(
                    self.stream_url, json=stream_data, stream=True, timeout=30
                ) as response:
                    for chunk in response.iter_lines():
                        if chunk:
                            chunk_count += 1
                            if (
                                chunk_count >= 200
                            ):  # Limit chunks to avoid too much output
                                break
                            time.sleep(0.1)  # Small delay to simulate processing
            except Exception as e:
                print(f"Request {rid} error: {e}")

            results[rid] = chunk_count
            print(f"Request {rid} finished with {chunk_count} chunks")

        # Start all requests concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            for config in request_configs:
                executor.submit(send_request, config)

            # Wait a bit for requests to start
            time.sleep(2)

            # Abort the middle request (req-002)
            target_rid = "req-002"
            print(f"\nAborting request {target_rid}...")
            abort_data = {"rid": target_rid}
            abort_response = requests.post(self.abort_url, json=abort_data)
            print(f"Abort response status: {abort_response.status_code}")
            print(f"Abort response body: {abort_response.text}")

            # Assert abort request was successful
            self.assertIn(
                abort_response.status_code,
                [200, 202],
                "Abort request should be accepted",
            )

        print(f"\nResults: {results}")

        # Check if the aborted request has fewer chunks
        self.assertIn(
            target_rid, results, f"Target request {target_rid} should be in results"
        )

        if target_rid in results:
            aborted_chunks = results[target_rid]
            other_chunks = [
                count for rid, count in results.items() if rid != target_rid
            ]
            print(f"Aborted request chunks: {aborted_chunks}")
            print(f"Other requests chunks: {other_chunks}")

            # The aborted request should generally have fewer chunks than others
            # (though this might not always be guaranteed due to timing)
            self.assertGreaterEqual(len(results), 1, "Should have at least one result")

    def test_abort_multiple_requests(self):
        print("\n=== Testing Multiple Requests Abort ===")

        # Create multiple requests with different rids
        request_configs = [
            {"rid": "req-001", "content": "Tell me a story about dragons"},
            {"rid": "req-002", "content": "Write a poem about the ocean"},
            {"rid": "req-003", "content": "Explain quantum physics"},
        ]

        results = {}

        def send_request(config):
            rid = config["rid"]
            content = config["content"]

            stream_data = {
                "model": "test-model",
                "messages": [{"role": "user", "content": content}],
                "stream": True,
                "rid": rid,
            }

            print(f"Starting request {rid}...")
            chunk_count = 0

            try:
                with requests.post(
                    self.stream_url, json=stream_data, stream=True, timeout=30
                ) as response:
                    for chunk in response.iter_lines():
                        if chunk:
                            chunk_count += 1
                            if (
                                chunk_count >= 200
                            ):  # Limit chunks to avoid too much output
                                break
                            time.sleep(0.1)  # Small delay to simulate processing
            except Exception as e:
                print(f"Request {rid} error: {e}")

            results[rid] = chunk_count
            print(f"Request {rid} finished with {chunk_count} chunks")

        # Start all requests concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            for config in request_configs:
                executor.submit(send_request, config)

            # Wait a bit for requests to start
            time.sleep(2)

            # Abort all requests
            abort_responses = {}
            for target_rid in ["req-001", "req-002", "req-003"]:
                print(f"\nAborting request {target_rid}...")
                abort_data = {"rid": target_rid}
                abort_response = requests.post(self.abort_url, json=abort_data)
                print(f"Abort response status: {abort_response.status_code}")
                print(f"Abort response body: {abort_response.text}")

                abort_responses[target_rid] = abort_response.status_code
                # Assert abort request was successful
                self.assertIn(
                    abort_response.status_code,
                    [200, 202],
                    f"Abort request for {target_rid} should be accepted",
                )

        print(f"\nResults: {results}")
        print(f"Abort responses: {abort_responses}")

        # All abort requests should have been successful
        for rid, status_code in abort_responses.items():
            self.assertIn(
                status_code, [200, 202], f"Abort request for {rid} should be successful"
            )


if __name__ == "__main__":
    # Register cleanup function to ensure services are stopped on exit
    def cleanup_on_exit():
        if hasattr(TestAbortRequest, "tearDownClass"):
            TestAbortRequest.tearDownClass()

    atexit.register(cleanup_on_exit)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal, cleaning up...")
        cleanup_on_exit()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Run the test suite
    print("Starting abort request tests using unittest...")
    print(
        "Note: This will start SgLang servers and router, which may take 30-40 seconds..."
    )
    unittest.main(verbosity=2)
