# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0

"""Optional fastapi-guard security middleware wiring for the HTTP server.

fastapi-guard (https://github.com/Guard-Core/fastapi-guard) provides IP
block/allow lists, rate limiting with auto-ban, user-agent blocking,
penetration-attempt detection, passive mode, optional security headers,
optional Redis-backed shared state, and optional IPInfo geo/cloud-provider
lookups.

Everything here is opt-in: with `SGLANG_GUARD_ENABLED` unset (the default),
`attach_guard` is a no-op and the server behaves exactly as before. The
module imports cleanly without torch so it stays unit-testable, mirroring
`srt/utils/auth.py`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt import environ

if TYPE_CHECKING:
    from fastapi import FastAPI
    from guard import SecurityConfig

DEFAULT_EXCLUDED_PATHS = (
    "/health,/health_generate,/get_server_info,/metrics,/docs,/openapi.json"
)
DEFAULT_TRUSTED_PROXIES = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def _build_guard_config() -> SecurityConfig:
    """Build the SecurityConfig from the SGLANG_GUARD_* environment variables."""
    from guard import SecurityConfig

    e = environ.envs
    kwargs: dict[str, Any] = {
        "enable_rate_limiting": True,
        "rate_limit": e.SGLANG_GUARD_RATE_LIMIT.get(),
        "rate_limit_window": e.SGLANG_GUARD_RATE_LIMIT_WINDOW.get(),
        "enable_ip_banning": True,
        "auto_ban_threshold": e.SGLANG_GUARD_AUTO_BAN_THRESHOLD.get(),
        "auto_ban_duration": e.SGLANG_GUARD_AUTO_BAN_DURATION.get(),
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable. With
        # --tokenizer-worker-num > 1 each worker is a separate process, so
        # set SGLANG_GUARD_REDIS_URL to share counters and bans.
        "enable_redis": False,
        "passive_mode": e.SGLANG_GUARD_PASSIVE_MODE.get(),
        "blacklist": _csv(e.SGLANG_GUARD_BLOCKED_IPS.get()),
        "blocked_user_agents": list(_csv(e.SGLANG_GUARD_BLOCKED_USER_AGENTS.get())),
        "trusted_proxies": _csv(e.SGLANG_GUARD_TRUSTED_PROXIES.get())
        or _csv(DEFAULT_TRUSTED_PROXIES),
        "trusted_proxy_depth": e.SGLANG_GUARD_TRUSTED_PROXY_DEPTH.get(),
        "exclude_paths": list(_csv(e.SGLANG_GUARD_EXCLUDED_PATHS.get()))
        or DEFAULT_EXCLUDED_PATHS.split(","),
    }

    if allowed_ips := _csv(e.SGLANG_GUARD_ALLOWED_IPS.get()):
        kwargs["whitelist"] = allowed_ips
    if blocked_countries := _csv(e.SGLANG_GUARD_BLOCKED_COUNTRIES.get()):
        kwargs["blocked_countries"] = frozenset(blocked_countries)
    if allowed_countries := _csv(e.SGLANG_GUARD_ALLOWED_COUNTRIES.get()):
        kwargs["whitelist_countries"] = frozenset(allowed_countries)
    if cloud_providers := _csv(e.SGLANG_GUARD_BLOCK_CLOUD_PROVIDERS.get()):
        kwargs["block_cloud_providers"] = frozenset(cloud_providers)

    if redis_url := e.SGLANG_GUARD_REDIS_URL.get():
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url
        kwargs["redis_prefix"] = "sglang_guard:"

    if ipinfo_token := e.SGLANG_GUARD_IPINFO_TOKEN.get():
        kwargs["ipinfo_token"] = ipinfo_token

    return SecurityConfig(**kwargs)


def attach_guard(app: FastAPI) -> None:
    """Attach the fastapi-guard middleware when SGLANG_GUARD_ENABLED is set.

    No-op unless the env flag is on. Fails loudly when the flag is on but
    the package is missing: a misconfiguration must never silently disable
    security.
    """
    if not environ.envs.SGLANG_GUARD_ENABLED.get():
        return
    try:
        from guard import SecurityMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "SGLANG_GUARD_ENABLED=true requires fastapi-guard. "
            'Install it with: pip install "sglang[guard]"'
        ) from exc

    app.add_middleware(SecurityMiddleware, config=_build_guard_config())
