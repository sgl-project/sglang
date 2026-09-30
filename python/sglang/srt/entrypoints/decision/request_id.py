"""Pure-ASGI middleware that tags every response of the decision routes with the TypeSafe request id."""

import uuid

from starlette.datastructures import Headers, MutableHeaders

TYPESAFE_REQUEST_ID_HEADER = "x-typesafe-request-id"
DECISION_ROUTES = frozenset({"/v1/decisions", "/v1/jev", "/v1/systemone"})


class TypesafeRequestIdMiddleware:
    """Echo the client's x-typesafe-request-id, or generate one, on success and error responses alike."""

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


def _is_decision_route(scope) -> bool:
    return scope["path"].removeprefix(scope.get("root_path", "")) in DECISION_ROUTES
