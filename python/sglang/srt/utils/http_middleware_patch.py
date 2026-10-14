"""
Fix @app.middleware("http") whose BaseHTTPMiddleware call_next replaces
ASGI ``receive``, breaking request.is_disconnected() and preventing
non-streaming request abort on client disconnect.

patch_app_http_middleware(app) replaces @app.middleware("http") with a
version whose call_next passes ``receive`` through untouched.
"""

from __future__ import annotations

from starlette.requests import Request


class _SentResponse:
    """Response proxy returned after the real response was already sent."""

    def __init__(self, status_code: int):
        self.status_code = status_code


class _PureASGIDispatch:
    """Pure ASGI middleware providing a fixed call_next that passes
    ``receive`` through untouched (unlike BaseHTTPMiddleware)."""

    def __init__(self, app, dispatch):
        self.app = app
        self.dispatch = dispatch

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        status_code = 500

        async def call_next(_request):
            nonlocal status_code

            async def send_and_capture(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            await self.app(scope, receive, send_and_capture)
            return _SentResponse(status_code)

        await self.dispatch(request, call_next)


def patch_app_http_middleware(app):
    """Replace @app.middleware("http") with a fixed-call_next version."""
    _orig = app.middleware

    def _fixed(middleware_type):
        if middleware_type == "http":

            def decorator(fn):
                app.add_middleware(_PureASGIDispatch, dispatch=fn)
                return fn

            return decorator
        return _orig(middleware_type)

    app.middleware = _fixed
