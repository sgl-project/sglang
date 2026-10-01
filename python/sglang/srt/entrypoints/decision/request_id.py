import uuid

from starlette.datastructures import Headers, MutableHeaders

TYPESAFE_REQUEST_ID_HEADER = "x-typesafe-request-id"
DECISION_ROUTES = frozenset({"/v1/decisions", "/v1/jev", "/v1/systemone"})


class TypesafeRequestIdMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not _is_decision_route(scope):
            return await self.app(scope, receive, send)
        request_id = Headers(scope=scope).get(TYPESAFE_REQUEST_ID_HEADER)
        request_id = request_id or uuid.uuid4().hex

        async def send_with_request_id(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)[TYPESAFE_REQUEST_ID_HEADER] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


def install_typesafe_request_id(app) -> None:
    build_middleware_stack = app.build_middleware_stack
    # Wrapping the built stack, unlike add_middleware, also covers auth and unhandled 500s;
    # Starlette builds it on the first call, after all middleware is added.
    app.build_middleware_stack = lambda: TypesafeRequestIdMiddleware(
        build_middleware_stack()
    )


def _is_decision_route(scope) -> bool:
    return scope["path"].removeprefix(scope.get("root_path", "")) in DECISION_ROUTES
