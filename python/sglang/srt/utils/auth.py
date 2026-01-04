"""Auth utilities for HTTP servers.

This module is intentionally lightweight (no torch import) so it can be used in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Optional

from fastapi.responses import ORJSONResponse


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    status_code: int = 200


def decide_request_auth(
    *,
    method: str,
    path: str,
    authorization_header: Optional[str],
    api_key: Optional[str],
    admin_api_key: Optional[str],
    admin_optional_auth_paths: AbstractSet[str],
    admin_force_auth_paths: AbstractSet[str],
) -> AuthDecision:
    """Pure auth decision function (easy to unit test).

    Path categories:
    - normal: any path not in admin_optional_auth_paths or admin_force_auth_paths
    - optional-auth: can be accessed without any key (if no keys configured),
      or with api_key/admin_api_key depending on server config.
    - force-auth: requires admin_api_key when configured; if admin_api_key is NOT configured,
      it must be rejected (403) even if api_key is provided.
    """
    if method == "OPTIONS":
        return AuthDecision(allowed=True)
    if path.startswith("/health") or path.startswith("/metrics"):
        return AuthDecision(allowed=True)

    def _is_bearer(token: str) -> bool:
        return authorization_header == "Bearer " + token

    # Force-auth endpoints: only admin_api_key can unlock them; if admin_api_key is unset,
    # reject them unconditionally (explicitly "not allowed").
    if path in admin_force_auth_paths:
        if not admin_api_key:
            return AuthDecision(allowed=False, status_code=403)
        if not _is_bearer(admin_api_key):
            return AuthDecision(allowed=False, status_code=401)
        return AuthDecision(allowed=True)

    # Optional-auth endpoints:
    # - no keys configured: allow
    # - only api_key: require api_key
    # - only admin_api_key: require admin_api_key
    # - both: require admin_api_key (api_key is NOT accepted)
    if path in admin_optional_auth_paths:
        if admin_api_key:
            return AuthDecision(allowed=_is_bearer(admin_api_key), status_code=401)
        elif api_key:
            return AuthDecision(allowed=_is_bearer(api_key), status_code=401)
        else:
            return AuthDecision(allowed=True)

    # Normal endpoints:
    # - if api_key is configured, require api_key (even if admin_api_key is also configured)
    # - otherwise allow (including the "admin_api_key only" case)
    if api_key:
        return AuthDecision(allowed=_is_bearer(api_key), status_code=401)

    return AuthDecision(allowed=True)


def add_api_key_middleware(
    app,
    *,
    api_key: Optional[str],
    admin_api_key: Optional[str],
    admin_optional_auth_paths: AbstractSet[str],
    admin_force_auth_paths: AbstractSet[str],
):
    """Add middleware for three endpoint categories: normal/optional-auth/force-auth."""

    @app.middleware("http")
    async def authentication(request, call_next):
        path = request.url.path
        authz = request.headers.get("Authorization")
        decision = decide_request_auth(
            method=request.method,
            path=path,
            authorization_header=authz,
            api_key=api_key,
            admin_api_key=admin_api_key,
            admin_optional_auth_paths=admin_optional_auth_paths,
            admin_force_auth_paths=admin_force_auth_paths,
        )

        if not decision.allowed:
            return ORJSONResponse(
                content={
                    "error": (
                        "Unauthorized" if decision.status_code == 401 else "Forbidden"
                    )
                },
                status_code=decision.status_code,
            )

        return await call_next(request)
