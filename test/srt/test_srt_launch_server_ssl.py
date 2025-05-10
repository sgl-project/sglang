"""
Usage:
python3 -m unittest test_srt_launch_server_ssl.TestSRTLaunchServerSSL
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)


class TestSRTLaunchServerSSL(CustomTestCase):

    def test_start_server_with_ssl_cert(self):
        # Generate a self-signed certificate

        # Generate a private key
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        # Build certificate subject and issuer
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
                x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MyCompany"),
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ]
        )

        # Build certificate
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.now(timezone.utc))
            .not_valid_after(datetime.now(timezone.utc) + timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("localhost")]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        with tempfile.NamedTemporaryFile(
            delete=False, mode="wb", suffix=".crt"
        ) as cert_file, tempfile.NamedTemporaryFile(
            delete=False, mode="wb", suffix=".key"
        ) as key_file:
            key_file.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )

            cert_file.write(cert.public_bytes(serialization.Encoding.PEM))

        process = popen_launch_server(
            model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            base_url="https://localhost:8000",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--ssl-certfile",
                cert_file.name,
                "--ssl-keyfile",
                key_file.name,
                "--ssl-self-signed-cert",
                "--max-total-tokens",
                "256",
            ],
        )

        print(f"Process PID: {process.pid}")
        assert process is not None

        kill_process_tree(process.pid)

        # Clean up
        os.remove(cert_file.name)
        os.remove(key_file.name)


if __name__ == "__main__":
    unittest.main()
