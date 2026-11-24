"""Authenticate the encoder, its reachable endpoint, and the full staged layout."""

import hashlib
import hmac
from functools import lru_cache
from urllib.parse import urlsplit

from sglang.srt.environ import envs


@lru_cache(maxsize=1)
def _read_key(path: str) -> bytes:
    with open(path, "rb") as key_file:
        key = key_file.read(33)
    if len(key) != 32:
        raise ValueError("The encoder handoff key file must contain exactly 32 bytes")
    return key


def require_handoff_key() -> bytes:
    path = envs.SGLANG_ENCODER_HANDOFF_KEY_FILE.get()
    if not path:
        raise ValueError(
            "External encoder handoff requires SGLANG_ENCODER_HANDOFF_KEY_FILE"
        )
    return _read_key(path)


def handoff_key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()


def _mac(key, peer, transfer, instance_id):
    # Protobuf preserves optional-field presence and the IEEE bits of timings.
    # Deterministic serialization binds every layout field without relying on
    # the JSON spelling or key order chosen by the router.
    message = (
        b"sglang.encoder.handoff.v1\x00"
        + peer.encode("ascii")
        + b"\x00"
        + instance_id
        + transfer.SerializeToString(deterministic=True)
    )
    return hmac.digest(key, message, "sha256")


class EncoderHandoffSigner:
    def __init__(self, peer: str, key: bytes):
        endpoint = urlsplit(peer)
        if (
            endpoint.scheme != "grpc"
            or not endpoint.hostname
            or endpoint.hostname in ("0.0.0.0", "::", "*")
            or not endpoint.port
            or endpoint.username is not None
            or endpoint.password is not None
            or endpoint.path
            or endpoint.query
            or endpoint.fragment
            or not peer.isascii()
            or any(ord(ch) <= 32 or ord(ch) == 127 for ch in peer)
        ):
            raise ValueError(
                "Configure a reachable grpc://host:port for the staged encoder"
            )
        self.peer = peer
        self._key = key
        self.key_id = handoff_key_id(key)

    @classmethod
    def from_config(cls):
        if not envs.SGLANG_ENCODER_HANDOFF_KEY_FILE.get():
            return None
        peer = envs.SGLANG_ENCODER_HANDOFF_ENDPOINT.get()
        if not peer:
            from sglang.srt.runtime_context import get_serving
            from sglang.srt.utils.network import NetworkAddress

            serving = get_serving()
            peer = NetworkAddress(serving.host, serving.port).to_url("grpc")
        return cls(peer, require_handoff_key())

    def sign(self, transfer, instance_id: bytes) -> str:
        return _mac(self._key, self.peer, transfer, instance_id).hex()


def verify_handoff(handoff, key: bytes) -> None:
    if len(handoff.signature) == 64 and len(handoff.instance_id) == 32:
        try:
            signature = bytes.fromhex(handoff.signature)
            instance = bytes.fromhex(handoff.instance_id)
            expected = _mac(key, handoff.peer, handoff.native_transfer(), instance)
            if len(instance) == 16 and hmac.compare_digest(signature, expected):
                return
        except (ValueError, OverflowError):
            pass
    raise ValueError("Invalid staged encoder handoff signature")
