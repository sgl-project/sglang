"""Producer teardown must not poison later requests in the shared test Catalog."""

import socket
import unittest
from http.client import HTTPConnection

from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCatalogFixture(CustomTestCase):
    def setUp(self):
        self.catalog = TestCaptureCatalog()
        self.addCleanup(self.catalog.close)

    def test_disconnect_before_body_completion_does_not_poison_the_next_producer(self):
        """A killed producer may send POST headers but none or only part of its body."""
        address = self.catalog.server.server_address
        for body in (b"", b'{"dataset_id":'):
            with self.subTest(body=body):
                with socket.create_connection(address, timeout=5) as client:
                    client.sendall(
                        b"POST /captures:begin HTTP/1.1\r\n"
                        b"Host: localhost\r\nContent-Length: 100\r\n\r\n" + body
                    )
                    client.shutdown(socket.SHUT_WR)
                    with client.makefile("rb") as response:
                        self.assertEqual(response.read(), b"")
                self.assertFalse(self.catalog.errors)
                self.assertFalse(self.catalog.captures)
        lease = HTTPCaptureCatalog(self.catalog.endpoint).begin(
            {
                "dataset_id": "fixture",
                "sample_id": "next-producer",
                "generation_id": "generation",
                "idempotency_key": "next-begin",
            }
        )
        self.assertEqual(lease.sample_id, "next-producer")
        self.assertEqual(len(self.catalog.captures), 1)
        self.assertFalse(self.catalog.errors)

    def test_complete_malformed_json_still_reports_a_protocol_failure(self):
        """Disconnect handling must not hide a fully transmitted invalid payload."""
        client = HTTPConnection(*self.catalog.server.server_address, timeout=5)
        try:
            client.request("POST", "/captures:begin", body=b"!")
            response = client.getresponse()
            self.assertEqual(response.status, 409)
            response.read()
        finally:
            client.close()
        self.assertEqual(len(self.catalog.errors), 1)
        self.assertIn("JSONDecodeError", self.catalog.errors[0])
        self.assertFalse(self.catalog.captures)


if __name__ == "__main__":
    unittest.main()
