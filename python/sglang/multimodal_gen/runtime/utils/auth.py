"""Auth utilities for the diffusion HTTP server.

Re-exports from the canonical implementation in ``sglang.srt.utils.auth`` so
that both the LLM engine and the diffusion engine share exactly the same
authentication logic and stay in sync automatically.
"""

from sglang.srt.utils.auth import (  # noqa: F401
    add_api_key_middleware,
    app_has_admin_force_endpoints,
    auth_level,
    AuthDecision,
    AuthLevel,
    decide_request_auth,
)
