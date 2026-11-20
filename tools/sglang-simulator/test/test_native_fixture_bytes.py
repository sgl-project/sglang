"""Opaque original source descriptors must never be changed by text tooling."""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).parent / "fixtures" / "simopus-native-source"


@pytest.mark.parametrize(
    "path", sorted(ROOT.glob("tp*/_meta/*.json")), ids=lambda p: p.name
)
def test_original_source_blob_matches_its_address(path):
    expected = path.stem.rsplit("-", 1)[-1]
    assert len(expected) == 64
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
