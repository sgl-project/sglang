import hashlib
import struct
from types import SimpleNamespace

import pytest

from sglang.srt.disaggregation.rust_pd_contract import (
    LAYOUT_DIGEST,
    NATIVE_DIGEST,
    layout_fingerprint,
    manifest_digest,
    native_abi_digest,
)
from sglang.srt.disaggregation.utils import prepare_abort


def test_manifest_uses_fixed_pd_canon_fields_and_rejects_missing_files(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "utf8-ß.bin"
    first.write_bytes(b"alpha")
    second.write_bytes(b"\x00\xff")

    domain = "SGLANG-PD-TEST-MANIFEST-V1"
    body = bytearray(domain.encode("ascii") + b"\0")
    body.extend(struct.pack(">I", 2))
    for path in (first, second):
        name = path.name.encode()
        payload = path.read_bytes()
        body.extend(struct.pack(">I", len(name)))
        body.extend(name)
        body.extend(struct.pack(">Q", len(payload)))
        body.extend(hashlib.sha256(payload).digest())
    expected = hashlib.sha256(body).hexdigest()

    assert manifest_digest(domain, [second, first]) == expected
    with pytest.raises(ValueError, match="PD_UNSUPPORTED"):
        manifest_digest(domain, [])


def test_layout_and_native_fingerprints_are_frozen():
    assert layout_fingerprint() == LAYOUT_DIGEST
    assert native_abi_digest() == NATIVE_DIGEST


def test_prepare_abort_serializes_pd_reason_separately_from_message():
    request = SimpleNamespace(return_logprob=False)

    prepare_abort(
        request,
        "message text is not a protocol discriminator",
        status_code=500,
        pd_reason="PD_TRANSFER_TIMEOUT",
    )

    assert request.finished_reason.to_json() == {
        "type": "abort",
        "message": "message text is not a protocol discriminator",
        "status_code": 500,
        "err_type": None,
        "pd_reason": "PD_TRANSFER_TIMEOUT",
    }
