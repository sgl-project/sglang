"""
Generate self-signed certificates for mTLS integration testing.
Creates a Certificate Authority (CA), server certificates, and client certificates.
"""

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate_private_key():
    """Generate an RSA private key."""
    return rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )


def generate_ca_certificate():
    """Generate a self-signed CA certificate."""
    private_key = generate_private_key()

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Test"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Test"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SGLang Test"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Test CA"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                key_encipherment=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(private_key, hashes.SHA256())
    )

    return private_key, cert


def generate_server_certificate(ca_key, ca_cert):
    """Generate a server certificate signed by the CA."""
    private_key = generate_private_key()

    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Test"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Test"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SGLang Test"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
                ]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return private_key, cert


def generate_client_certificate(ca_key, ca_cert):
    """Generate a client certificate signed by the CA."""
    private_key = generate_private_key()

    subject = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Test"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Test"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "SGLang Test"),
            x509.NameAttribute(NameOID.ORGANIZATIONAL_UNIT_NAME, "Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, "test-client"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                key_cert_sign=False,
                crl_sign=False,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage(
                [
                    x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
                ]
            ),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return private_key, cert


def save_key(key, path: Path):
    """Save private key to PEM file."""
    with open(path, "wb") as f:
        f.write(
            key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )


def save_cert(cert, path: Path):
    """Save certificate to PEM file."""
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


def generate_all_certificates(output_dir: Path):
    """Generate all certificates and keys for mTLS testing."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("==> Generating CA certificate...")
    ca_key, ca_cert = generate_ca_certificate()
    save_key(ca_key, output_dir / "ca-key.pem")
    save_cert(ca_cert, output_dir / "ca-cert.pem")

    print("==> Generating server certificate...")
    server_key, server_cert = generate_server_certificate(ca_key, ca_cert)
    save_key(server_key, output_dir / "server-key.pem")
    save_cert(server_cert, output_dir / "server-cert.pem")

    print("==> Generating client certificate...")
    client_key, client_cert = generate_client_certificate(ca_key, ca_cert)
    save_key(client_key, output_dir / "client-key.pem")
    save_cert(client_cert, output_dir / "client-cert.pem")

    print(f"==> Certificates generated successfully in {output_dir}")
    print()
    print("Files created:")
    print("  - ca-cert.pem       : CA certificate (for verifying server/client certs)")
    print("  - ca-key.pem        : CA private key")
    print("  - server-cert.pem   : Server certificate")
    print("  - server-key.pem    : Server private key")
    print("  - client-cert.pem   : Client certificate")
    print("  - client-key.pem    : Client private key")
    print()
    print("Test server can use: server-cert.pem + server-key.pem")
    print("Test router can use: client-cert.pem + client-key.pem + ca-cert.pem")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    certs_dir = script_dir / "test_certs"
    generate_all_certificates(certs_dir)
