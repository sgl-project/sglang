"""Auth utilities for HTTP servers.

This module is intentionally lightweight (no torch import) so it can be used in unit tests.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    error_status_code: int = 401  # Only meaningful when allowed=False


class AuthLevel(str, Enum):
    """Per-endpoint auth level (attached to endpoint function via `@auth_level`)."""

    NORMAL = "normal"
    ADMIN_OPTIONAL = "admin_optional"
    ADMIN_FORCE = "admin_force"


def auth_level(level: AuthLevel):
    """Mark endpoint with auth level (stored in endpoint metadata)."""

    def decorator(func):
        func._auth_level = level
        return func

    return decorator


def _get_auth_level_from_app_and_scope(app: Any, scope: dict) -> AuthLevel:
    """Best-effort resolve auth level by matching the request to a route."""
    # Import lazily to keep this module unit-test friendly (FastAPI/Starlette are not
    # required unless you actually use the middleware / route matching).
    from starlette.routing import Match

    # Prefer app.router.routes when available; fall back to app.routes.
    routes = getattr(getattr(app, "router", None), "routes", None) or getattr(
        app, "routes", []
    )

    for route in routes:
        try:
            match, child_scope = route.matches(scope)
        except Exception:
            continue
        if match == Match.FULL:
            endpoint = child_scope.get("endpoint") or getattr(route, "endpoint", None)
            level = getattr(endpoint, "_auth_level", None)
            return level if isinstance(level, AuthLevel) else AuthLevel.NORMAL

    return AuthLevel.NORMAL


def app_has_admin_force_endpoints(app: Any) -> bool:
    """Return True if any route endpoint is marked as ADMIN_FORCE."""
    routes = getattr(getattr(app, "router", None), "routes", None) or getattr(
        app, "routes", []
    )
    for route in routes:
        endpoint = getattr(route, "endpoint", None)
        if getattr(endpoint, "_auth_level", None) == AuthLevel.ADMIN_FORCE:
            return True
    return False


def decide_request_auth(
    *,
    method: str,
    path: str,
    authorization_header: Optional[str],
    api_key: Optional[str],
    admin_api_key: Optional[str],
    auth_level: AuthLevel,
) -> AuthDecision:
    """Pure auth decision function (easy to unit test).

    Auth levels:
    - NORMAL: legacy behavior (api_key protects all endpoints when configured)
    - ADMIN_OPTIONAL: can be accessed without any key (if no keys configured),
      or with api_key/admin_api_key depending on server config.
    - ADMIN_FORCE: requires admin_api_key; if admin_api_key is NOT configured,
      it must be rejected (403) even if api_key is provided.

    NOTE :
    - Health/metrics endpoints are always allowed (even when api_key/admin_api_key is set),
      to support k8s/liveness/readiness and Prometheus scraping without embedding secrets.
    - We match them by prefix to cover common variants like /health_generate.
    """
    if method == "OPTIONS":
        return AuthDecision(allowed=True)

    if path.startswith("/health") or path.startswith("/metrics"):
        return AuthDecision(allowed=True)

    def _check_bearer_token(
        authorization_header: Optional[str], expected_token: str
    ) -> bool:
        """Check bearer token with constant-time comparison."""
        if not authorization_header:
            return False
        parts = authorization_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        return secrets.compare_digest(parts[1], expected_token)

    # Force-auth endpoints: only admin_api_key can unlock them; if admin_api_key is unset,
    # reject them unconditionally (explicitly "not allowed").
    if auth_level == AuthLevel.ADMIN_FORCE:
        if not admin_api_key:
            return AuthDecision(allowed=False, error_status_code=403)
        if not _check_bearer_token(authorization_header, admin_api_key):
            return AuthDecision(allowed=False)
        return AuthDecision(allowed=True)

    # Optional-auth endpoints:
    # - no keys configured: allow
    # - only api_key: require api_key
    # - only admin_api_key: require admin_api_key
    # - both: require admin_api_key (api_key is NOT accepted)
    if auth_level == AuthLevel.ADMIN_OPTIONAL:
        if admin_api_key:
            return AuthDecision(
                allowed=_check_bearer_token(authorization_header, admin_api_key)
            )
        elif api_key:
            return AuthDecision(
                allowed=_check_bearer_token(authorization_header, api_key)
            )
        else:
            return AuthDecision(allowed=True)

    # Normal endpoints:
    # - if api_key is configured, require api_key (even if admin_api_key is also configured)
    # - otherwise allow (including the "admin_api_key only" case)
    if api_key:
        return AuthDecision(allowed=_check_bearer_token(authorization_header, api_key))

    return AuthDecision(allowed=True)


def add_api_key_middleware(
    app,
    *,
    api_key: Optional[str],
    admin_api_key: Optional[str],
):
    """Add middleware for three endpoint auth levels: normal/admin_optional/admin_force."""
    # Import lazily so `decide_request_auth()` can be unit-tested without FastAPI installed.
    from fastapi.responses import ORJSONResponse

    @app.middleware("http")
    async def authentication(request, call_next):
        path = request.url.path
        authz = request.headers.get("Authorization")
        level = _get_auth_level_from_app_and_scope(request.app, request.scope)
        decision = decide_request_auth(
            method=request.method,
            path=path,
            authorization_header=authz,
            api_key=api_key,
            admin_api_key=admin_api_key,
            auth_level=level,
        )

        if not decision.allowed:
            return ORJSONResponse(
                content={
                    "error": (
                        "Unauthorized"
                        if decision.error_status_code == 401
                        else "Forbidden"
                    )
                },
                status_code=decision.error_status_code,
            )

        return await call_next(request)
