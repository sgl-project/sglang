import datetime
import os
import socket
import subprocess
import sys
import time
from contextlib import closing

import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from urllib3.exceptions import InsecureRequestWarning

# Suppress insecure request warnings due to self-signed cert
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def find_free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def generate_self_signed_cert(cert_path: str, key_path: str) -> None:
    """Generate a self-signed certificate and private key for localhost."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SGLang Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=10))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost")]), critical=False
        )
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def test_tls_server() -> None:
    """End-to-end test for TLS-enabled router startup and basic endpoints."""
    cert_path = "cert.pem"
    key_path = "key.pem"
    generate_self_signed_cert(cert_path, key_path)

    port = find_free_port()

    cmd = [
        sys.executable,
        "-m",
        "sglang_router.launch_router",
        "--worker-urls",
        "http://127.0.0.1:9999",  # Dummy worker
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tls-cert-path",
        cert_path,
        "--tls-key-path",
        key_path,
        "--log-level",
        "info",
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    try:
        # Wait for server to start and respond to health check
        start_time = time.time()
        while time.time() - start_time < 15:
            try:
                response = requests.get(
                    f"https://localhost:{port}/health", verify=False, timeout=2
                )
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if proc.poll() is not None:
                stdout, stderr = proc.communicate()
                raise RuntimeError(
                    f"Router process died early.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )

            time.sleep(0.5)
        else:
            raise TimeoutError("Server did not become healthy within 15 seconds")

        # Verify basic endpoints work over TLS
        models_resp = requests.get(
            f"https://localhost:{port}/v1/models", verify=False, timeout=2
        )
        assert models_resp.status_code in (
            200,
            503,
        )  # 503 expected with no healthy workers

        # Minimal generate request (should be rejected or queued)
        gen_payload = {"model": "dummy", "prompt": "test", "max_new_tokens": 1}
        gen_resp = requests.post(
            f"https://localhost:{port}/generate",
            json=gen_payload,
            verify=False,
            timeout=2,
        )
        assert gen_resp.status_code in (
            200,
            400,
            503,
        )  # Various valid responses with dummy worker

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        for path in (cert_path, key_path):
            if os.path.exists(path):
                os.remove(path)
