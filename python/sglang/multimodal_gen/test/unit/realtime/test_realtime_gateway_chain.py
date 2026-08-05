# SPDX-License-Identifier: Apache-2.0

import asyncio
import socket
import time
from urllib.parse import parse_qs, urlsplit

import httpx
import uvicorn
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedOK

from sglang.multimodal_gen.runtime.entrypoints.realtime_gateway_server import (
    create_app,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    decode_message,
    encode_message,
)
from sglang.multimodal_gen.runtime.realtime.coordinator import (
    SessionAssignment,
    WorkerSlot,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _Coordinator:
    def __init__(self, denoiser_endpoint: str):
        self.denoiser_endpoint = denoiser_endpoint
        self.admitted = []
        self.released = []
        self.released_at = None

    async def health(self):
        return {"status": "ready"}

    async def admit(self, **request):
        self.admitted.append(request)
        return SessionAssignment(
            user_id=request["user_id"],
            session_id=request["session_id"],
            generation_id=request["generation_id"],
            token="lease-token",
            expires_at=time.monotonic() + 60,
            denoiser=WorkerSlot(
                worker_id="denoiser-1",
                role="denoiser",
                endpoint=self.denoiser_endpoint,
                az="test-a",
                slot_index=0,
                model_revision="minwm-r1",
                vae_fingerprint="taew2_2",
            ),
            vae=WorkerSlot(
                worker_id="vae-1",
                role="vae",
                endpoint="ws://vae-1:18081/v1/realtime_vae/decode",
                az="test-a",
                slot_index=0,
                model_revision="",
                vae_fingerprint="taew2_2",
            ),
        )

    async def renew(self, assignment):
        return assignment

    async def release(self, assignment):
        self.released_at = time.monotonic()
        self.released.append(assignment)


class _TraceQuery:
    async def query(self, trace_id, **_kwargs):
        return {
            "trace_id": trace_id,
            "events": [{"event": "gateway.ws_accepted", "trace_seq": 1}],
            "next_cursor": 1,
        }


async def _run_gateway(app, port: int):
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            lifespan="off",
        )
    )
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.01)
    assert server.started
    return server, task


def test_gateway_routes_control_and_direct_vae_media_and_queries_trace_over_http():
    async def run():
        gateway_port = _free_port()
        denoiser_port = _free_port()
        denoiser_close_started_at = None

        async def denoiser(connection):
            nonlocal denoiser_close_started_at
            query = parse_qs(urlsplit(connection.request.path).query)
            session_id = query["session_id"][0]
            generation_id = query["generation_id"][0]
            output_url = query["gateway_output_url"][0]
            output_token = query["gateway_output_token"][0]

            await connection.send(encode_message("session_ready"))
            async with connect(output_url, max_size=None, compression=None) as output:
                await output.send(
                    encode_message(
                        "session_output_open",
                        session_id=session_id,
                        generation_id=generation_id,
                        token=output_token,
                    )
                )
                accepted = decode_message(await output.recv())
                assert accepted["type"] == "session_output_accepted"
                await output.send(
                    encode_message(
                        "frame_batch",
                        session_id=session_id,
                        generation_id=generation_id,
                        request_id="request-0",
                        chunk_index=0,
                        frame_batch_index=0,
                        payload_lengths=[4],
                        payload=b"webp",
                        content_type="image/webp",
                        width=8,
                        height=8,
                        num_frames=1,
                    )
                )
                denoiser_close_started_at = time.monotonic()
                await connection.close(code=1000, reason="generation complete")

        coordinator = _Coordinator(
            f"ws://127.0.0.1:{denoiser_port}/v1/realtime_video/generate"
        )
        app = create_app(
            coordinator,
            model_revision="minwm-r1",
            vae_fingerprint="taew2_2",
            internal_output_url=(
                f"ws://127.0.0.1:{gateway_port}/v1/internal/realtime_output"
            ),
            trace_query=_TraceQuery(),
            release_grace_s=0.05,
        )

        async with serve(denoiser, "127.0.0.1", denoiser_port):
            server, server_task = await _run_gateway(app, gateway_port)
            try:
                url = (
                    f"ws://127.0.0.1:{gateway_port}/v1/realtime_video/generate"
                    "?user_id=user-a&trace_id=trace-a"
                )
                async with connect(url, max_size=None, compression=None) as browser:
                    async def send_actions_until_closed():
                        event_id = 1
                        while True:
                            try:
                                await browser.send(
                                    encode_message(
                                        "camera_actions",
                                        event_id=event_id,
                                        actions=["w"],
                                    )
                                )
                            except ConnectionClosedOK:
                                return
                            event_id += 1
                            await asyncio.sleep(0)

                    action_task = asyncio.create_task(send_actions_until_closed())
                    messages = []
                    try:
                        while True:
                            messages.append(
                                    decode_message(
                                    await asyncio.wait_for(browser.recv(), 4)
                                )
                            )
                    except ConnectionClosedOK:
                        pass
                    await action_task
                assert {message["type"] for message in messages} == {
                    "session_ready",
                    "frame_batch",
                }
                assert all("trace" not in message["type"] for message in messages)

                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"http://127.0.0.1:{gateway_port}"
                        "/v1/realtime_video/traces/trace-a"
                    )
                assert response.status_code == 200
                assert response.json()["events"][0]["event"] == (
                    "gateway.ws_accepted"
                )
            finally:
                server.should_exit = True
                await server_task

        assert len(coordinator.admitted) == 1
        assert len(coordinator.released) == 1
        assert denoiser_close_started_at is not None
        assert coordinator.released_at - denoiser_close_started_at >= 0.045

    asyncio.run(run())
