"""Auth utilities for HTTP servers.

This module is intentionally lightweight (no torch import) so it can be used in unit tests.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from sglang.srt.environ import envs

# The Rust server answers with the same text (`ADMIN_KEY_MISSING`).
ADMIN_API_KEY_MISSING_MESSAGE = (
    "This endpoint requires admin API key, but this server was started "
    "without one (admin-api-key). Restart with --admin-api-key to enable."
)


def bearer_auth_headers(key: Optional[str]) -> dict:
    """`Authorization: Bearer <key>`, or no header when there is no key."""
    return {"Authorization": f"Bearer {key}"} if key else {}


def admin_auth_headers(admin_api_key: Optional[str] = None) -> dict:
    """Bearer headers for the admin endpoints: `admin_api_key`, else `SGLANG_ADMIN_API_KEY`."""
    return bearer_auth_headers(admin_api_key or envs.SGLANG_ADMIN_API_KEY.get())


@dataclass(frozen=True)
class AuthDecision:
    allowed: bool
    error_status_code: int = 401  # Only meaningful when allowed=False


class AuthLevel(str, Enum):
    """An endpoint's auth level, as `auth_levels.json` declares it."""

    NORMAL = "normal"
    ADMIN_OPTIONAL = "admin_optional"
    ADMIN_FORCE = "admin_force"


def _unique_keys(pairs: list) -> dict:
    keys = [key for key, _ in pairs]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"auth_levels.json declares {duplicates} twice")
    return dict(pairs)


# Every HTTP route sglang serves, by the Python, the Rust, and the diffusion
# server alike, and its auth level; the Rust server compiles the same file in. A
# route it does not declare (an extension's, or one whose path an env var sets)
# is NORMAL.
AUTH_LEVELS: dict[str, AuthLevel] = {
    path: AuthLevel(level)
    for path, level in json.loads(
        Path(__file__).with_name("auth_levels.json").read_text(),
        object_pairs_hook=_unique_keys,
    ).items()
}


def _iter_leaf_routes(routes, prefix: str = ""):
    """Yield `(prefix, route)`, descending into the routers FastAPI keeps nested on include."""
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is None:
            yield prefix, route
            continue
        context = getattr(route, "include_context", None)
        yield from _iter_leaf_routes(
            included.routes, prefix + (getattr(context, "prefix", "") or "")
        )


def undeclared_routes(app: Any) -> list[str]:
    """The routes of `app` that `auth_levels.json` does not declare."""
    return sorted(
        prefix + route.path
        for prefix, route in _iter_leaf_routes(app.router.routes)
        if prefix + route.path not in AUTH_LEVELS
    )


def _get_auth_level_from_app_and_scope(app: Any, scope: dict) -> AuthLevel:
    """Best-effort resolve the auth level of the route Starlette's router picks,
    a wrong method included, so it is refused at that route's level before 405."""
    # Import lazily to keep this module unit-test friendly (FastAPI/Starlette are not
    # required unless you actually use the middleware / route matching).
    from starlette.routing import Match

    partial = None
    for prefix, route in _iter_leaf_routes(app.router.routes):
        if not scope["path"].startswith(prefix):
            continue
        try:
            match, _ = route.matches(
                {**scope, "path": scope["path"][len(prefix) :]} if prefix else scope
            )
        except Exception:
            continue
        if match == Match.FULL:
            return AUTH_LEVELS.get(prefix + route.path, AuthLevel.NORMAL)
        if match == Match.PARTIAL and partial is None:
            partial = prefix + route.path

    if partial is None:
        return AuthLevel.NORMAL
    return AUTH_LEVELS.get(partial, AuthLevel.NORMAL)


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
    - Health/liveness/readiness/metrics endpoints are always allowed (even when
      api_key/admin_api_key is set), to support k8s probes and Prometheus
      scraping without embedding secrets.
    - We match them by prefix to cover common variants like /health_generate.
    """
    if method == "OPTIONS":
        return AuthDecision(allowed=True)

    if (
        path.startswith("/health")
        or path.startswith("/liveness")
        or path.startswith("/ready")
        or path.startswith("/metrics")
    ):
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


def add_api_key_middleware(app, *, get_keys: Callable[[], tuple]):
    """Enforce the endpoint auth levels; `get_keys` returns `(api_key, admin_api_key)` per request."""
    # Import lazily so `decide_request_auth()` can be unit-tested without FastAPI installed.
    from fastapi.responses import ORJSONResponse
    from starlette.datastructures import Headers

    class _ApiKeyASGIMiddleware:
        """ASGI-native middleware to preserve client disconnect events."""

        def __init__(self, app, *, fastapi_app):
            self.app = app
            self.fastapi_app = fastapi_app

        async def __call__(self, scope, receive, send):
            if scope["type"] not in ("http", "websocket"):
                await self.app(scope, receive, send)
                return

            api_key, admin_api_key = get_keys()
            decision = decide_request_auth(
                method=scope.get("method", "GET"),
                path=scope["path"],
                authorization_header=Headers(scope=scope).get("Authorization"),
                api_key=api_key,
                admin_api_key=admin_api_key,
                auth_level=_get_auth_level_from_app_and_scope(self.fastapi_app, scope),
            )

            if decision.allowed:
                await self.app(scope, receive, send)
                return

            if scope["type"] == "websocket":
                # Closing before accept makes the server answer the handshake with 403.
                await send({"type": "websocket.close", "code": 1008})
                return

            response = ORJSONResponse(
                content={
                    "error": (
                        "Unauthorized"
                        if decision.error_status_code == 401
                        else ADMIN_API_KEY_MISSING_MESSAGE
                    )
                },
                status_code=decision.error_status_code,
            )
            await response(scope, receive, send)

    app.add_middleware(_ApiKeyASGIMiddleware, fastapi_app=app)
