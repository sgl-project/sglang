import asyncio
import os
import re
import unittest
from concurrent.futures import ThreadPoolExecutor

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests,
    send_generate_requests,
)


class TestMaxQueuedRequests(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",  # Enforce max request concurrency is 1
                "1",
                "--max-queued-requests",  # Enforce max queued request number is 1
                "1",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_max_queued_requests_validation_with_serial_requests(self):
        """Verify request is not throttled when the max concurrency is 1."""
        status_codes = send_generate_requests(
            self.base_url,
            num_requests=10,
        )

        for status_code in status_codes:
            assert status_code == 200  # request shouldn't be throttled

    def test_max_queued_requests_validation_with_concurrent_requests(self):
        """Verify request throttling with concurrent requests."""
        status_codes = asyncio.run(
            send_concurrent_generate_requests(self.base_url, num_requests=10)
        )

        expected_status_codes = [200, 200, 503, 503, 503, 503, 503, 503, 503, 503]
        assert status_codes == expected_status_codes

    def test_max_running_requests_and_max_queued_request_validation(self):
        """Verify running request and queued request numbers based on server logs."""
        rr_pattern = re.compile(r"#running-req:\s*(\d+)")
        qr_pattern = re.compile(r"#queue-req:\s*(\d+)")

        with open(STDERR_FILENAME) as lines:
            for line in lines:
                rr_match, qr_match = rr_pattern.search(line), qr_pattern.search(line)
                if rr_match:
                    assert int(rr_match.group(1)) <= 1
                if qr_match:
                    assert int(qr_match.group(1)) <= 1


if __name__ == "__main__":
    unittest.main()
