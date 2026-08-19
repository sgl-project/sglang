"""
FractalMesh Security Core
HMAC-SHA256 payload fingerprinting and verification
Samuel James Hiotis | ABN 56 628 117 363
"""
import hmac
import hashlib
import json
import os


def generate_fingerprint(payload_dict: dict) -> tuple[str, bytes]:
    """
    Sign a dict payload with the BUS_SECRET.
    Returns (hex_signature, canonical_json_bytes).
    """
    secret        = os.getenv("BUS_SECRET", "fallback").encode("utf-8")
    payload_bytes = json.dumps(payload_dict, sort_keys=True).encode("utf-8")
    signature     = hmac.new(secret, payload_bytes, hashlib.sha256).hexdigest()
    return signature, payload_bytes


def verify_fingerprint(signature: str, payload_bytes: bytes) -> bool:
    """
    Verify an HMAC-SHA256 signature against raw payload bytes.
    Returns False (not just raises) if signature is missing or invalid.
    """
    if not signature:
        return False
    secret   = os.getenv("BUS_SECRET", "fallback").encode("utf-8")
    expected = hmac.new(secret, payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
