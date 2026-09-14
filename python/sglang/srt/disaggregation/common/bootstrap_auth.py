"""Shared secret for PD bootstrap mutating HTTP endpoints.

``PUT /route`` and ``POST /register_dp_rank`` write attacker-controlled
addresses into the KV-transfer routing table. Decode then trusts those
entries, so unauthenticated writes are route poisoning. Writes require a
bearer token; GET ``/route``, POST ``/query_dp_ranks``, and ``/health`` stay
unauthenticated so decode workers can still rendezvous.

Token resolution (first match wins):

1. ``SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN``
2. ``--api-key`` (``get_serving().api_key``)
3. unset — mutating endpoints fail closed (401)

``ensure_bootstrap_auth_token`` must run in the parent process before scheduler
spawn so every child inherits the same secret. Multi-node prefill additionally
broadcasts rank-0's token before remote ranks PUT.
"""

from __future__ import annotations

import secrets
from typing import Dict, Optional

from sglang.srt.environ import envs


def get_bootstrap_auth_token() -> Optional[str]:
    token = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.get()
    if token:
        return token
    try:
        from sglang.srt.runtime_context import get_serving

        return get_serving().api_key or None
    except Exception:
        return None


def ensure_bootstrap_auth_token(api_key: Optional[str] = None) -> str:
    """Persist a bootstrap write token in the process environment.

    Idempotent: an already-set env token is left untouched. Called from the
    engine parent before subprocess spawn so tokenizer (bootstrap server) and
    scheduler (``register_to_bootstrap``) share the secret.
    """
    existing = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.get()
    if existing:
        return existing
    token = api_key or secrets.token_urlsafe(32)
    envs.SGLANG_DISAGGREGATION_BOOTSTRAP_AUTH_TOKEN.set(token)
    return token


def bootstrap_auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    if token is None:
        token = get_bootstrap_auth_token()
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def is_bootstrap_write_authorized(
    authorization_header: Optional[str],
    expected_token: Optional[str],
) -> bool:
    """Fail closed: missing expected token or missing/wrong bearer is denied."""
    if not expected_token:
        return False
    if not authorization_header:
        return False
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    try:
        return secrets.compare_digest(parts[1], expected_token)
    except Exception:
        return False
