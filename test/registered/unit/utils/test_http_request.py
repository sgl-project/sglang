"""Regression coverage for frontend streaming TLS settings."""

import os
import shutil
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import requests
from urllib3 import HTTPResponse
from urllib3.connectionpool import HTTPSConnectionPool

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import http_request

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestHttpRequest(CustomTestCase):
    def test_streaming_ca_bundle(self):
        """Custom CAs must reach the TLS pool; defaults must keep verification."""
        tls_settings = []

        def respond(pool, **kwargs):
            tls_settings.append((pool.ca_certs, pool.cert_reqs))
            return HTTPResponse(
                body=BytesIO(b"data: [DONE]\n\n"), status=200, preload_content=False
            )

        # Keep requests' TLS setup; replace only the network I/O.
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                HTTPSConnectionPool, "urlopen", autospec=True, side_effect=respond
            ),
        ):
            custom_ca = str(Path(directory) / "ca.pem")
            shutil.copyfile(requests.certs.where(), custom_ca)
            for verify in (custom_ca, None):
                with (
                    self.subTest(verify=verify),
                    http_request(
                        "https://localhost/generate", stream=True, verify=verify
                    ) as response,
                ):
                    self.assertEqual(
                        list(response.iter_lines()), [b"data: [DONE]", b""]
                    )
            self.assertEqual(
                tls_settings,
                [
                    (custom_ca, "CERT_REQUIRED"),
                    (requests.certs.where(), "CERT_REQUIRED"),
                ],
            )


if __name__ == "__main__":
    unittest.main()
