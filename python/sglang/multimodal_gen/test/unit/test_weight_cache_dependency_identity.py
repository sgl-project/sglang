# SPDX-License-Identifier: Apache-2.0
"""Published native code and selected attention providers are cache identity."""

import importlib.metadata
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.multimodal_gen.runtime.weight_cache import identity


@pytest.fixture
def publications(monkeypatch):
    monkeypatch.setenv("SGLANG_USE_SGL_FA3_KERNEL", "1")
    monkeypatch.delenv("SGLANG_INKLING_FA4_USE_PIP", raising=False)
    records = {}

    def distribution(name):
        value = records.get(name, ("1.0", "published RECORD"))
        if value is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(version=value[0], read_text=lambda _: value[1])

    with patch.object(
        identity.importlib.metadata, "distribution", side_effect=distribution
    ):
        yield records


def fingerprint():
    return identity.dependency_identity(
        SimpleNamespace(weight_cache_allow_unverified_build=False)
    )


@pytest.mark.parametrize(
    "provider",
    [
        "sglang-kernel",
        "flash-attn-4",
        "triton",
        "nvidia-cutlass-dsl",
        "nvidia-cutlass-dsl-libs-cu13",
        "flashinfer-cubin",
    ],
)
@pytest.mark.parametrize(
    "changed", [("2.0", "published RECORD"), ("1.0", "different binary RECORD")]
)
def test_publication_changes_identity(publications, provider, changed):
    before = fingerprint()
    publications[provider] = changed
    assert fingerprint() != before
    assert "sglang-kernel" in before["distributions"]
    assert "sgl-kernel" not in before["distributions"]


@pytest.mark.parametrize("provider", ["torch", "sglang-kernel", "nvidia-cutlass-dsl"])
def test_required_provider_missing_fails_closed(publications, provider):
    publications[provider] = None
    with pytest.raises(ValueError, match="required provider is missing"):
        fingerprint()


def test_fa4_selection_and_required_external_provider(publications, monkeypatch):
    before = fingerprint()
    monkeypatch.setenv("SGLANG_INKLING_FA4_USE_PIP", "1")
    assert fingerprint() != before
    publications["flash-attn-4"] = None
    with pytest.raises(ValueError, match="flash-attn-4"):
        fingerprint()
    monkeypatch.delenv("SGLANG_INKLING_FA4_USE_PIP")
    assert fingerprint()["distributions"]["flash-attn-4"] is None


def test_unpublished_record_requires_explicit_development_bypass(publications, caplog):
    publications["sglang-kernel"] = ("1.0", None)
    with pytest.raises(ValueError, match="lacks published RECORD"):
        fingerprint()
    result = identity.dependency_identity(
        SimpleNamespace(weight_cache_allow_unverified_build=True)
    )
    assert result["distributions"]["sglang-kernel"]["record"] is None
    assert "UNVERIFIED provider" in caplog.text
    publications["sglang-kernel"] = None
    with pytest.raises(ValueError, match="required provider"):
        identity.dependency_identity(
            SimpleNamespace(weight_cache_allow_unverified_build=True)
        )


def test_dynamic_fa3_artifact_provider_is_not_admitted(publications, monkeypatch):
    monkeypatch.setenv("SGLANG_USE_SGL_FA3_KERNEL", "0")
    with pytest.raises(ValueError, match="SGLANG_USE_SGL_FA3_KERNEL"):
        fingerprint()


def test_environment_includes_provider_receipt(publications):
    args = SimpleNamespace(
        weight_cache_allow_unverified_build=False, gpu_ids=None, base_gpu_id=0
    )
    with (
        patch.object(identity, "source_digest", return_value="source"),
        patch.object(identity, "compute_env_stamp", return_value={}),
        patch.object(
            identity.current_platform,
            "get_device_capability",
            return_value=SimpleNamespace(major=9, minor=0),
        ),
    ):
        before = identity.environment_identity(args)
        publications["sglang-kernel"] = ("1.0", "changed RECORD")
        assert identity.environment_identity(args) != before


def test_source_hash_failure_cannot_collapse_different_development_builds():
    with patch.object(
        identity, "source_digest", side_effect=ValueError("source changed")
    ):
        with pytest.raises(ValueError, match="source changed"):
            identity.environment_identity(
                SimpleNamespace(weight_cache_allow_unverified_build=True)
            )
