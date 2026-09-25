"""Opt-in CTX-affine admission edge for a fixed fleet of native PD routers.

New sessions choose the least in-flight request-body bytes. Existing sessions
keep their canonical CTX worker URL. This is neither a KV index nor a compute
cost model. Run one owner per session namespace; state is not replicated.
"""

import argparse
import asyncio
import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import aiohttp
from aiohttp import web
from multidict import CIMultiDict
from yarl import URL

SESSION_HEADER = "X-SMG-Routing-Key"
TARGET_HEADER = "X-SMG-Target-Worker"
INFERENCE_PATHS = {"/v1/chat/completions", "/v1/completions", "/generate"}
HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
}
logger = logging.getLogger(__name__)


def forwarded_headers(headers):
    excluded = HOP_HEADERS | {
        token.strip().lower()
        for value in headers.getall("Connection", [])
        for token in value.split(",")
    }
    return CIMultiDict((k, v) for k, v in headers.items() if k.lower() not in excluded)


def backend_index(key, count):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % count


def inventory_identity(snapshot):
    workers = snapshot["workers"]
    urls = [worker["url"] for worker in workers]
    if len(urls) != len(set(urls)):
        raise ValueError("Duplicate canonical worker URLs")
    if any(
        worker.get("worker_type") not in {"prefill", "decode"} for worker in workers
    ):
        raise ValueError("Only disaggregated prefill/decode workers are supported")
    return sorted((w["url"], w.get("id", ""), w["worker_type"]) for w in workers)


def inventory_ranks(inventories, expected_ctx, expected_gen):
    if not inventories or min(expected_ctx, expected_gen) < 1:
        raise ValueError(
            "Nonempty inventories and positive expected rank counts required"
        )
    memberships = []
    for snapshot in inventories:
        inventory_identity(snapshot)
        workers = snapshot["workers"]
        ctx = sorted(w["url"] for w in workers if w["worker_type"] == "prefill")
        gen = sorted(w["url"] for w in workers if w["worker_type"] == "decode")
        if len(ctx) != expected_ctx or len(gen) != expected_gen:
            raise ValueError("Unexpected CTX/GEN rank counts")
        if not all(w.get("is_healthy") is True for w in workers):
            raise ValueError("Initial inventory must contain only healthy workers")
        memberships.append((ctx, gen))
    if any(membership != memberships[0] for membership in memberships):
        raise ValueError("Native frontends disagree on CTX/GEN membership")
    return memberships[0][0]


@dataclass
class Reservation:
    owner: str
    weight: int
    released: bool = False


class SessionAdmission:
    """Single-event-loop selection/reservation, with no intervening awaits.

    Bytes remain reserved until response completion/disconnect, not CTX finish.
    Session ownership persists until process restart. Exhaustion fails closed:
    silently evicting an owner would discard the cache-affinity guarantee.
    """

    def __init__(self, ranks, max_sessions=100000):
        if not ranks or len(set(ranks)) != len(ranks) or max_sessions < 1:
            raise ValueError("Distinct ranks and a positive session bound required")
        self.ranks = tuple(sorted(ranks))
        self.max_sessions = max_sessions
        self.owners = {}
        self.bytes = Counter()
        self.active = Counter()
        self.assigned = Counter()

    def reserve(self, key, weight, eligible):
        if (
            not key
            or isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight < 1
        ):
            raise ValueError("Nonempty session key and positive byte weight required")
        if key not in self.owners:
            if len(self.owners) >= self.max_sessions:
                raise OverflowError("Session affinity capacity reached")
            choices = [rank for rank in self.ranks if rank in eligible]
            if not choices:
                raise OverflowError("No healthy CTX owner for new session")
            owner = min(
                choices,
                key=lambda r: (self.bytes[r], self.active[r], self.assigned[r], r),
            )
            self.owners[key] = owner
            self.assigned[owner] += 1
        owner = self.owners[key]
        if owner not in eligible:
            raise OverflowError("Session CTX owner unavailable; no affinity fallback")
        self.bytes[owner] += weight
        self.active[owner] += 1
        return Reservation(owner, weight)

    def release(self, reservation):
        owner = reservation.owner
        if (
            reservation.released
            or self.active[owner] < 1
            or self.bytes[owner] < reservation.weight
        ):
            raise ValueError("Invalid or already released admission reservation")
        self.active[owner] -= 1
        self.bytes[owner] -= reservation.weight
        reservation.released = True

    def snapshot(self):
        return {
            "policy": "new-session-least-inflight-wire-bytes-sticky-owner",
            "session_count": len(self.owners),
            "ranks": {
                r: {
                    "inflight_wire_bytes": self.bytes[r],
                    "inflight_requests": self.active[r],
                    "assigned_sessions": self.assigned[r],
                }
                for r in self.ranks
            },
        }


def make_app(
    backends,
    inventories,
    expected_ctx,
    expected_gen,
    *,
    inventory_guard_interval=5.0,
    max_sessions=100000,
):
    backends = tuple(backend.rstrip("/") for backend in backends)
    if len(backends) != len(inventories) or len(set(backends)) != len(backends):
        raise ValueError("One inventory per distinct native frontend required")
    if inventory_guard_interval <= 0:
        raise ValueError("Positive inventory guard interval required")
    for backend in backends:
        url = URL(backend)
        if (
            url.scheme != "http"
            or not url.host
            or url.path not in ("", "/")
            or url.query_string
            or url.fragment
            or url.user is not None
        ):
            raise ValueError("Backend must be a trusted plain HTTP origin")
    policy = SessionAdmission(
        inventory_ranks(inventories, expected_ctx, expected_gen), max_sessions
    )
    identities = [inventory_identity(snapshot) for snapshot in inventories]
    current = list(inventories)
    state = {"inventory_valid": True, "inventory_failure": None}
    app = web.Application()

    async def session_context(app):
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=30),
            auto_decompress=False,
            trust_env=False,
            skip_auto_headers={
                "Accept",
                "Accept-Encoding",
                "User-Agent",
                "Content-Type",
            },
        ) as session:
            app["upstream_session"] = session

            async def verify():
                async def get(index):
                    async with session.get(
                        backends[index] + "/workers",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as response:
                        response.raise_for_status()
                        snapshot = await response.json()
                    if inventory_identity(snapshot) != identities[index]:
                        raise ValueError("Native worker identity/membership changed")
                    current[index] = snapshot

                await asyncio.gather(*(get(i) for i in range(len(backends))))

            await verify()

            async def guard():
                while True:
                    await asyncio.sleep(inventory_guard_interval)
                    try:
                        await verify()
                    except (
                        aiohttp.ClientError,
                        OSError,
                        asyncio.TimeoutError,
                        ValueError,
                        KeyError,
                        TypeError,
                    ) as error:
                        state.update(
                            inventory_valid=False, inventory_failure=str(error)
                        )
                        logger.error("CTX admission latched closed: %s", error)
                        return

            task = asyncio.create_task(guard())
            try:
                yield
            finally:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass  # Expected shutdown of our owned inventory watcher.

    app.cleanup_ctx.append(session_context)

    async def routing_state(request):
        return web.json_response({**policy.snapshot(), **state})

    async def forward(request):
        keys = request.headers.getall(SESSION_HEADER, [])
        if len(keys) > 1 or (keys and not keys[0]):
            raise web.HTTPBadRequest(
                text="Exactly one nonempty session routing key required"
            )
        inference = request.method == "POST" and request.path in INFERENCE_PATHS
        if inference and not keys:
            raise web.HTTPBadRequest(text="Missing stable X-SMG-Routing-Key")
        if TARGET_HEADER in request.headers:
            raise web.HTTPBadRequest(text="CTX target is owned by session admission")
        # Preserve framing even when stripping Connection-nominated hop headers.
        if "content-length" in {
            token.strip().lower()
            for value in request.headers.getall("Connection", [])
            for token in value.split(",")
        }:
            raise web.HTTPBadRequest(text="Connection must not nominate Content-Length")
        index = backend_index(keys[0], len(backends)) if keys else 0
        headers = forwarded_headers(request.headers)
        reservation = None
        response = None
        if inference:
            if not state["inventory_valid"]:
                raise web.HTTPServiceUnavailable(
                    text="CTX inventory invalidated; no remapping"
                )
            if request.content_length is None:
                raise web.HTTPLengthRequired(
                    text="Fixed Content-Length required for admission"
                )
            eligible = {
                w["url"]
                for w in current[index]["workers"]
                if w["worker_type"] == "prefill" and w.get("is_healthy") is True
            }
            try:
                reservation = policy.reserve(
                    keys[0], max(1, request.content_length), eligible
                )
            except OverflowError as error:
                raise web.HTTPServiceUnavailable(text=str(error)) from error
            headers[TARGET_HEADER] = reservation.owner
        try:
            body = request.content.iter_any() if request.can_read_body else None
            async with app["upstream_session"].request(
                request.method,
                URL(backends[index] + request.raw_path, encoded=True),
                data=body,
                headers=headers,
                allow_redirects=False,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=forwarded_headers(upstream.headers),
                )
                await response.prepare(request)
                async for chunk in upstream.content.iter_any():
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as error:
            if response is not None and response.prepared:
                if request.transport is not None:
                    request.transport.close()
                raise  # Never turn a truncated stream into a successful response.
            raise web.HTTPBadGateway(
                text="Upstream request failed; not retried"
            ) from error
        finally:
            if reservation is not None:
                policy.release(reservation)

    app.router.add_get("/ctx-routing-state", routing_state)
    app.router.add_route("*", "/{path:.*}", forward)
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--inventory", action="append", required=True)
    parser.add_argument("--expected-ctx", type=int, required=True)
    parser.add_argument("--expected-gen", type=int, required=True)
    parser.add_argument("--max-sessions", type=int, default=100000)
    parser.add_argument("--inventory-guard-interval", type=float, default=5.0)
    args = parser.parse_args()
    inventories = [json.loads(Path(path).read_text()) for path in args.inventory]
    app = make_app(
        args.backend,
        inventories,
        args.expected_ctx,
        args.expected_gen,
        max_sessions=args.max_sessions,
        inventory_guard_interval=args.inventory_guard_interval,
    )
    web.run_app(
        app,
        host=args.host,
        port=args.port,
        handler_cancellation=True,
        auto_decompress=False,
        access_log=None,
    )


if __name__ == "__main__":
    main()
