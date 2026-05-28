"""
flags.py — Vercel Flags SDK integration for FractalMesh FastAPI
Exposes /.well-known/vercel/flags for the Vercel Flags toolbar/dashboard,
verifies access using HMAC-SHA256, and evaluates flag overrides from the
vercel-flag-overrides JWE cookie.
Credentials from ~/.secrets/fractal.env via os.environ.setdefault.
Samuel James Hiotis | ABN 56 628 117 363
"""
import base64
import hashlib
import hmac as _hmac
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

# ── Credentials (vault pattern — never hardcoded) ─────────────────────────────
# Set FLAGS_SECRET in ~/.secrets/fractal.env or Vercel project env vars.
FLAGS_SECRET: str = os.environ.get("FLAGS_SECRET", "")

# ── Flag definitions ──────────────────────────────────────────────────────────
# Description, defaultValue, and options sourced from:
#   vercel flags inspect ggh --scope samuel-hiotis-projects
#
# ggh kind: boolean   variants: false (off) | true (on)
FLAGS_DEFINITIONS: dict[str, dict[str, Any]] = {
    "ggh": {
        "description": "GGH feature rollout for FractalMesh sovereign deployment",
        "origin": (
            "https://vercel.com/samuel-hiotis-projects"
            "/fractalmeshsovereigndeployment/settings/flags"
        ),
        "options": [
            {"value": False, "label": "Disabled"},
            {"value": True,  "label": "Enabled"},
        ],
        "defaultValue": False,
    },
}

# ── Access verification ───────────────────────────────────────────────────────

def _verify_access(authorization: str) -> bool:
    """
    Verify the Vercel Flags dashboard Authorization header.
    Expected: Bearer HMAC-SHA256(FLAGS_SECRET, "/.well-known/vercel/flags")
    encoded as base64url without padding.
    """
    if not FLAGS_SECRET:
        return True  # open when no secret configured
    if not authorization or not authorization.startswith("Bearer "):
        return False
    token = authorization.removeprefix("Bearer ").strip()
    digest = _hmac.new(
        FLAGS_SECRET.encode(),
        b"/.well-known/vercel/flags",
        hashlib.sha256,
    ).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return _hmac.compare_digest(expected, token)

# ── Override cookie decoding ──────────────────────────────────────────────────

def _decode_overrides(cookie: str) -> dict[str, Any]:
    """
    Decode a Vercel flag-override cookie (JWE encrypted with FLAGS_SECRET).
    Falls back gracefully to empty dict on any error.
    """
    if not cookie or not FLAGS_SECRET:
        return {}
    try:
        from jose import jwe, jwt  # python-jose[cryptography]

        # Try JWE (encrypted overrides) first, then fall back to JWT (signed)
        try:
            raw = jwe.decrypt(cookie, FLAGS_SECRET)
            return json.loads(raw).get("overrides", {})
        except Exception:
            claims = jwt.decode(cookie, FLAGS_SECRET, algorithms=["HS256"])
            return claims.get("overrides", {})
    except Exception:
        logger.debug("Could not decode flag overrides cookie", exc_info=True)
        return {}

# ── Flag evaluation ───────────────────────────────────────────────────────────

def evaluate(name: str, request: Request) -> Any:
    """
    Evaluate a flag for the given request.
    Override cookie takes precedence over the flag's defaultValue.
    """
    definition = FLAGS_DEFINITIONS.get(name)
    if definition is None:
        raise ValueError(f"Unknown flag: {name!r}")
    default = definition.get("defaultValue")
    overrides = _decode_overrides(request.cookies.get("vercel-flag-overrides", ""))
    return overrides.get(name, default)


def ggh(request: Request) -> bool:
    """Evaluate the ggh feature flag. Returns True when the flag is enabled."""
    return bool(evaluate("ggh", request))

# ── Well-known router ─────────────────────────────────────────────────────────

router = APIRouter(tags=["flags"])


@router.get(
    "/.well-known/vercel/flags",
    summary="Vercel Flags manifest",
    include_in_schema=False,
)
async def well_known_flags(request: Request) -> JSONResponse:
    """
    Endpoint consumed by the Vercel Flags toolbar and dashboard.
    Returns the flag definitions manifest after verifying the HMAC bearer token.
    """
    auth = request.headers.get("Authorization", "")
    if not _verify_access(auth):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return JSONResponse(
        {
            "definitions": FLAGS_DEFINITIONS,
            "hints": [],
        }
    )
