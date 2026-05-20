"""Unit tests for codec_version — version negotiation, opt-on stages,
graceful downgrade, and the 426 builder.

Run:
    pytest -xvs python/sglang/srt/entrypoints/test_codec_version.py

Tests intentionally manipulate the environment via monkeypatch to verify
both the default-off path (every test starts with capabilities disabled)
and the opt-on / opt-enforce paths.
"""

from __future__ import annotations

import json

import pytest

from sglang.srt.entrypoints.codec_version import (
    HEADER_VERSION_INTRODUCED,
    any_v04_mandatory,
    collect_required_features,
    filter_codec_headers,
    make_426_response,
    needs_upgrade,
    parse_client_version,
    safety_policy_enabled,
    safety_policy_required,
    should_emit_header,
    version_ge,
    version_lt,
    version_policy_document,
    version_policy_mode,
)

# ── parse helpers ────────────────────────────────────────────────────────────


def test_version_ge_basic():
    assert version_ge("0.4", "0.3")
    assert version_ge("0.4", "0.4")
    assert not version_ge("0.3", "0.4")


def test_version_ge_tolerates_leading_v_and_patch():
    assert version_ge("v0.4", "0.4")
    assert version_ge("0.4.5", "0.4")  # patch ignored
    assert version_ge("0.4", "0.4.7")


def test_version_lt_inverse_of_ge():
    assert version_lt("0.3", "0.4")
    assert not version_lt("0.4", "0.4")


# ── client-version header parsing ────────────────────────────────────────────


def _request_with_headers(**headers):
    """Build a minimal Request-like object exposing .headers like Starlette."""

    class _H:
        def __init__(self, d):
            self._d = {k.lower(): v for k, v in d.items()}

        def get(self, key, default=""):
            return self._d.get(key.lower(), default)

    class _R:
        def __init__(self, h):
            self.headers = _H(h)

    return _R(headers)


def test_parse_client_version_present():
    req = _request_with_headers(**{"Codec-Client-Version": "0.4"})
    assert parse_client_version(req) == "0.4"


def test_parse_client_version_absent_defaults_to_0_2():
    """Spec: absent header = treat as v0.2 baseline (most conservative)."""
    req = _request_with_headers()
    assert parse_client_version(req) == "0.2"


def test_parse_client_version_strips_v_prefix():
    req = _request_with_headers(**{"Codec-Client-Version": "v0.3"})
    assert parse_client_version(req) == "0.3"


# ── should_emit_header / floor table ─────────────────────────────────────────


def test_v04_client_sees_all_headers():
    for header in HEADER_VERSION_INTRODUCED:
        assert should_emit_header(header, "0.4"), header


def test_v03_client_sees_v02_and_v03_only():
    assert should_emit_header("Codec-Zstd-Dict", "0.3")  # v0.2
    assert should_emit_header("Codec-Tokenizer-Map", "0.3")  # v0.2
    assert should_emit_header("Codec-Latent-Map", "0.3")  # v0.3
    assert not should_emit_header("Codec-Safety-Policy", "0.3")  # v0.4
    assert not should_emit_header("Codec-Safety-Policy-Hash", "0.3")  # v0.4


def test_v02_client_sees_v02_only():
    assert should_emit_header("Codec-Zstd-Dict", "0.2")
    assert not should_emit_header("Codec-Latent-Map", "0.2")
    assert not should_emit_header("Codec-Safety-Policy", "0.2")


def test_unknown_header_passes_through():
    """Standard HTTP headers (Content-Type, Vary, etc.) aren't in the
    table — must default to True so the filter doesn't strip them."""
    assert should_emit_header("Content-Type", "0.0")
    assert should_emit_header("Vary", "0.0")


def test_filter_codec_headers_v03_strips_safety():
    headers = {
        "Vary": "Accept-Encoding",
        "Content-Encoding": "zstd",
        "Codec-Zstd-Dict": "sha256:abc",
        "Codec-Safety-Policy": "acme/strict",
        "Codec-Safety-Policy-Hash": "sha256:def",
    }
    out = filter_codec_headers(headers, "0.3")
    assert "Vary" in out
    assert "Content-Encoding" in out
    assert "Codec-Zstd-Dict" in out
    assert "Codec-Safety-Policy" not in out
    assert "Codec-Safety-Policy-Hash" not in out


# ── capability enable / enforce ──────────────────────────────────────────────


def test_default_off_all_capabilities(monkeypatch):
    """Spec: ship default OFF. With no env set, no capability is
    enabled, no capability is enforced, no v0.4 wire bytes leak."""
    monkeypatch.delenv("CODEC_SAFETY_POLICY", raising=False)
    monkeypatch.delenv("CODEC_SAFETY_POLICY_REQUIRED", raising=False)
    monkeypatch.delenv("CODEC_VERSION_POLICY", raising=False)
    assert not safety_policy_enabled()
    assert not safety_policy_required()
    assert version_policy_mode() == "off"
    assert not any_v04_mandatory()
    assert collect_required_features() == []


def test_safety_enable_without_enforce(monkeypatch):
    """Stage-1 only: capability advertised to v0.4 clients, v0.3 clients
    still served — no 426."""
    monkeypatch.setenv("CODEC_SAFETY_POLICY", "acme/strict-v3")
    monkeypatch.delenv("CODEC_SAFETY_POLICY_REQUIRED", raising=False)
    assert safety_policy_enabled()
    assert not safety_policy_required()
    assert not any_v04_mandatory()
    assert not needs_upgrade("0.3")
    assert not needs_upgrade("0.4")


def test_safety_enforce_requires_enable(monkeypatch):
    """Setting _REQUIRED without enabling is a no-op — enforcement
    cannot fire if the capability isn't even configured."""
    monkeypatch.delenv("CODEC_SAFETY_POLICY", raising=False)
    monkeypatch.setenv("CODEC_SAFETY_POLICY_REQUIRED", "1")
    assert not safety_policy_required()
    assert not any_v04_mandatory()


def test_safety_enforce_triggers_for_old_clients(monkeypatch):
    monkeypatch.setenv("CODEC_SAFETY_POLICY", "acme/strict-v3")
    monkeypatch.setenv("CODEC_SAFETY_POLICY_REQUIRED", "1")
    assert any_v04_mandatory()
    assert "safety-policy-enforcement" in collect_required_features()
    assert needs_upgrade("0.2")
    assert needs_upgrade("0.3")
    assert not needs_upgrade("0.4")
    assert not needs_upgrade("0.5")


def test_version_policy_strict_triggers_426(monkeypatch):
    """CODEC_VERSION_POLICY=strict alone (no safety) still requires
    a v0.4+ client — it's the simplest case of "enforce version
    floor without enforcing a specific feature"."""
    monkeypatch.delenv("CODEC_SAFETY_POLICY", raising=False)
    monkeypatch.setenv("CODEC_VERSION_POLICY", "strict")
    assert any_v04_mandatory()
    assert needs_upgrade("0.3")
    assert not needs_upgrade("0.4")


def test_version_policy_advisory_does_not_trigger_426(monkeypatch):
    """`advisory` mode means the server may emit Codec-Min-Version on
    responses but never returns 426 for version reasons."""
    monkeypatch.setenv("CODEC_VERSION_POLICY", "advisory")
    assert version_policy_mode() == "advisory"
    assert not any_v04_mandatory()
    assert not needs_upgrade("0.3")


# ── 426 response ─────────────────────────────────────────────────────────────


def test_426_body_shape(monkeypatch):
    monkeypatch.setenv("CODEC_SAFETY_POLICY", "acme/strict-v3")
    monkeypatch.setenv("CODEC_SAFETY_POLICY_REQUIRED", "1")
    monkeypatch.setenv("CODEC_DEPLOYMENT_ID", "test-deploy-1")

    resp = make_426_response(client_version="0.3")
    assert resp.status_code == 426
    body = json.loads(resp.body)
    assert body["error"] == "codec_version_required"
    assert body["minimum_version"] == "0.4"
    assert body["client_version"] == "0.3"
    assert "safety-policy-enforcement" in body["required_features"]
    assert body["deployment_id"] == "test-deploy-1"
    assert "docs_url" in body
    # Headers from the spec § HTTP-transport shape.
    assert resp.headers["Codec-Min-Version"] == "0.4"
    assert "safety-policy-enforcement" in resp.headers["Codec-Required-Features"]


def test_426_body_renders_as_string_for_old_clients(monkeypatch):
    """Spec § graceful degradation: a v0.3 client that doesn't know the
    JSON shape can still surface body['error'] as a string. Verify the
    body is valid JSON with a top-level `error` string."""
    monkeypatch.setenv("CODEC_VERSION_POLICY", "strict")
    resp = make_426_response(client_version="0.3")
    body = json.loads(resp.body)
    assert isinstance(body["error"], str)
    # No nested required structure — flat scalars + one list.
    for k in ("minimum_version", "client_version", "docs_url"):
        assert isinstance(body[k], str)


# ── well-known version-policy doc ────────────────────────────────────────────


def test_version_policy_doc_absent_when_nothing_mandatory(monkeypatch):
    """Spec § Version policy (v0.4+): a deployment without mandatory
    features SHOULD NOT publish this document."""
    monkeypatch.delenv("CODEC_SAFETY_POLICY", raising=False)
    monkeypatch.delenv("CODEC_VERSION_POLICY", raising=False)
    assert version_policy_document() is None


def test_version_policy_doc_present_when_required(monkeypatch):
    monkeypatch.setenv("CODEC_SAFETY_POLICY", "acme/strict-v3")
    monkeypatch.setenv("CODEC_SAFETY_POLICY_REQUIRED", "1")
    monkeypatch.setenv("CODEC_DEPLOYMENT_ID", "well-known-test")
    doc = version_policy_document()
    assert doc is not None
    assert doc["minimum_version"] == "0.4"
    assert "safety-policy-enforcement" in doc["required_features"]
    assert doc["deployment_id"] == "well-known-test"


# ── default-off conformance sanity (HTTP-level, lightweight) ─────────────────


def test_426_route_helper_returns_proper_response_object():
    """Spot-check that the response is constructible without FastAPI
    routing — useful for unit tests of the helper itself."""
    resp = make_426_response(client_version="0.2")
    assert resp.status_code == 426
    assert resp.media_type == "application/json"


# ── Full matrix: (client_version × server_config) → wire outcome ──────────────
#
# Spec § Capabilities are opt-on at the server enumerates the configurations
# we ship. Tests below exercise the cartesian product so every combination
# of client/server state has a documented expected behavior.


# Client versions we support today. Includes the absent-header case
# (which parses as DEFAULT_CLIENT_VERSION = "0.2") and a future version
# (0.5) to confirm forward-compat.
CLIENT_VERSIONS = ["0.2", "0.3", "0.4", "0.5"]


def _clear_codec_env(monkeypatch):
    for v in (
        "CODEC_SAFETY_POLICY",
        "CODEC_SAFETY_POLICY_REQUIRED",
        "CODEC_VERSION_POLICY",
        "CODEC_DEPLOYMENT_ID",
    ):
        monkeypatch.delenv(v, raising=False)


def _apply(monkeypatch, env: dict[str, str]):
    _clear_codec_env(monkeypatch)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


# Each server config + (client_version → expected outcome) table.
#
# `426` means needs_upgrade() is True and the client would be refused.
# `200` means the request would proceed; the test then asserts that
# only the headers permitted by the client's version reach the wire.
SERVER_CONFIGS = [
    {
        "name": "default-off",
        "env": {},
        # No capability mandatory; every client served. v0.4 wire only
        # for v0.4+ clients per filter_codec_headers, but no v0.4 headers
        # would be added in the first place because no capability is on.
        "expected_426": {v: False for v in CLIENT_VERSIONS},
        "expects_well_known": False,
    },
    {
        "name": "safety-enabled-not-enforced",
        "env": {"CODEC_SAFETY_POLICY": "acme/strict-v3"},
        # Capability on; nothing required. v0.3 clients still served.
        "expected_426": {v: False for v in CLIENT_VERSIONS},
        "expects_well_known": False,
    },
    {
        "name": "safety-enforced",
        "env": {
            "CODEC_SAFETY_POLICY": "acme/strict-v3",
            "CODEC_SAFETY_POLICY_REQUIRED": "1",
        },
        # v0.4+ clients OK; v0.3- refused.
        "expected_426": {"0.2": True, "0.3": True, "0.4": False, "0.5": False},
        "expects_well_known": True,
    },
    {
        "name": "version-policy-advisory",
        "env": {"CODEC_VERSION_POLICY": "advisory"},
        # Advisory means "header set on responses but no 426". No 426 for any client.
        "expected_426": {v: False for v in CLIENT_VERSIONS},
        "expects_well_known": False,
    },
    {
        "name": "version-policy-strict",
        "env": {"CODEC_VERSION_POLICY": "strict"},
        # Strict means 426 for clients below 0.4 regardless of which
        # specific capability triggered it.
        "expected_426": {"0.2": True, "0.3": True, "0.4": False, "0.5": False},
        "expects_well_known": True,
    },
    {
        "name": "safety-enabled-version-strict",
        "env": {
            "CODEC_SAFETY_POLICY": "acme/strict-v3",
            "CODEC_VERSION_POLICY": "strict",
        },
        # Safety enabled (no enforce) + version strict = 426 by version
        # but body would NOT include safety-policy-enforcement in
        # required_features (safety isn't enforced).
        "expected_426": {"0.2": True, "0.3": True, "0.4": False, "0.5": False},
        "expects_well_known": True,
    },
    {
        "name": "safety-required-version-strict",
        "env": {
            "CODEC_SAFETY_POLICY": "acme/strict-v3",
            "CODEC_SAFETY_POLICY_REQUIRED": "1",
            "CODEC_VERSION_POLICY": "strict",
        },
        # Full enforce. v0.4+ served; v0.3- refused with both signals.
        "expected_426": {"0.2": True, "0.3": True, "0.4": False, "0.5": False},
        "expects_well_known": True,
    },
    {
        "name": "safety-required-version-off",
        "env": {
            "CODEC_SAFETY_POLICY": "acme/strict-v3",
            "CODEC_SAFETY_POLICY_REQUIRED": "1",
        },
        # Same as safety-enforced — proves version-policy alone is not
        # required for safety enforcement to fire.
        "expected_426": {"0.2": True, "0.3": True, "0.4": False, "0.5": False},
        "expects_well_known": True,
    },
]


@pytest.mark.parametrize("cfg", SERVER_CONFIGS, ids=[c["name"] for c in SERVER_CONFIGS])
@pytest.mark.parametrize("client_version", CLIENT_VERSIONS)
def test_matrix_needs_upgrade(monkeypatch, cfg, client_version):
    """For every (server config × client version) cell, needs_upgrade()
    matches the documented expectation."""
    _apply(monkeypatch, cfg["env"])
    expected = cfg["expected_426"][client_version]
    actual = needs_upgrade(client_version)
    assert (
        actual == expected
    ), f"server={cfg['name']} client={client_version} expected 426={expected} got 426={actual}"


@pytest.mark.parametrize("cfg", SERVER_CONFIGS, ids=[c["name"] for c in SERVER_CONFIGS])
def test_matrix_well_known_presence(monkeypatch, cfg):
    """version_policy_document() returns a doc iff any capability is
    mandatory — the well-known file MUST mirror the runtime 426
    behavior per spec."""
    _apply(monkeypatch, cfg["env"])
    doc = version_policy_document()
    if cfg["expects_well_known"]:
        assert doc is not None, f"server={cfg['name']} should publish well-known"
        assert doc["minimum_version"] == "0.4"
    else:
        assert doc is None, f"server={cfg['name']} should NOT publish well-known"


@pytest.mark.parametrize("client_version", CLIENT_VERSIONS)
def test_matrix_header_filter_safety_block(monkeypatch, client_version):
    """When the safety capability is ENABLED (stage-1), the server
    may emit Codec-Safety-Policy headers. The filter MUST strip them
    for v0.3- clients (graceful downgrade) but pass them through for
    v0.4+ clients."""
    _apply(
        monkeypatch,
        {
            "CODEC_SAFETY_POLICY": "acme/strict-v3",
        },
    )
    raw_headers = {
        "Vary": "Accept-Encoding",
        "Content-Encoding": "zstd",
        "Codec-Zstd-Dict": "sha256:abc",  # v0.2 — always emitted
        "Codec-Tokenizer-Map": "sha256:def",  # v0.2 — always emitted
        "Codec-Latent-Map": "sha256:ghi",  # v0.3 — emitted to v0.3+
        "Codec-Safety-Policy": "acme/strict-v3",  # v0.4 — emitted to v0.4+
        "Codec-Safety-Policy-Hash": "sha256:jkl",  # v0.4 — emitted to v0.4+
    }
    filtered = filter_codec_headers(raw_headers, client_version)

    # v0.2 headers + standard HTTP always pass.
    assert "Vary" in filtered
    assert "Content-Encoding" in filtered
    assert "Codec-Zstd-Dict" in filtered
    assert "Codec-Tokenizer-Map" in filtered

    # v0.3 header gated.
    if version_ge(client_version, "0.3"):
        assert "Codec-Latent-Map" in filtered
    else:
        assert "Codec-Latent-Map" not in filtered

    # v0.4 headers gated.
    if version_ge(client_version, "0.4"):
        assert "Codec-Safety-Policy" in filtered
        assert "Codec-Safety-Policy-Hash" in filtered
    else:
        assert "Codec-Safety-Policy" not in filtered
        assert "Codec-Safety-Policy-Hash" not in filtered


@pytest.mark.parametrize("cfg", SERVER_CONFIGS, ids=[c["name"] for c in SERVER_CONFIGS])
def test_matrix_426_body_features_match_config(monkeypatch, cfg):
    """The required_features field on a 426 body MUST list exactly the
    capabilities that are ENFORCED — not the ones merely enabled."""
    _apply(monkeypatch, cfg["env"])
    # Skip configs that don't 426; their body shape is moot.
    if not any_v04_mandatory():
        return
    resp = make_426_response(client_version="0.3")
    body = json.loads(resp.body)
    features = set(body["required_features"])
    expected = set(collect_required_features())
    assert (
        features == expected
    ), f"server={cfg['name']} expected required_features={expected} got {features}"


def test_matrix_default_off_emits_zero_v04_wire(monkeypatch):
    """Default-off ship state MUST emit zero v0.4 headers to any client
    — regardless of the client's advertised version. The spec's hard
    rule: a controlled fleet pays zero v0.4 wire cost when the server
    has no capability enabled."""
    _clear_codec_env(monkeypatch)
    # Even a v0.4 client should not see v0.4 headers when the server
    # has nothing enabled — the headers aren't generated in the first
    # place (no capability runtime, no policy to advertise).
    assert not any_v04_mandatory()
    # The filter passes whatever the caller passed in; the contract is
    # that the SERVER doesn't ATTACH v0.4 headers when the capability
    # is off. Simulate that by passing only the v0.2 set.
    filtered = filter_codec_headers(
        {"Codec-Zstd-Dict": "sha256:x"},
        "0.4",
    )
    assert filtered == {"Codec-Zstd-Dict": "sha256:x"}


def test_matrix_graceful_downgrade_v03_never_sees_v04(monkeypatch):
    """Even when the server has safety ENABLED + ENFORCED, a v0.3 client
    that somehow received headers must not see them post-filter."""
    _apply(
        monkeypatch,
        {
            "CODEC_SAFETY_POLICY": "acme/strict-v3",
            "CODEC_SAFETY_POLICY_REQUIRED": "1",
        },
    )
    # In practice the v0.3 client would have been 426'd before any
    # response body. But if a server bug allowed the 2xx path to fire,
    # graceful downgrade is the last line of defense.
    raw = {
        "Codec-Zstd-Dict": "sha256:abc",
        "Codec-Safety-Policy": "acme/strict-v3",
        "Codec-Safety-Policy-Hash": "sha256:jkl",
    }
    filtered = filter_codec_headers(raw, "0.3")
    assert filtered == {"Codec-Zstd-Dict": "sha256:abc"}
