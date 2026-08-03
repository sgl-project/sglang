# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from sglang.multimodal_gen.runtime.realtime.async_vae_protocol import (
    ProtocolViolation,
    decode_message,
    encode_message,
    latent_header_from_message,
    validate_payload,
)
from sglang.multimodal_gen.runtime.realtime.async_vae_worker import (
    AsyncVAEWorker,
    SessionOpen,
    TAEHVEngine,
)


def _tensor_from_payload(header, payload: bytes) -> torch.Tensor:
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[header.dtype]
    return torch.frombuffer(bytearray(payload), dtype=dtype).reshape(header.shape)


def create_app(worker: AsyncVAEWorker, *, max_message_bytes: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await worker.close_all()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return JSONResponse(
            {
                "status": "ok",
                "active_sessions": worker.active_sessions,
                "max_sessions": worker.max_sessions,
            }
        )

    @app.websocket("/v1/realtime_vae/decode")
    async def decode_socket(ws: WebSocket):
        await ws.accept()
        identity = None
        send_lock = asyncio.Lock()
        finish_tasks: set[asyncio.Task] = set()

        async def send(message: bytes):
            async with send_lock:
                await ws.send_bytes(message)

        try:
            while True:
                wire = await ws.receive_bytes()
                message = decode_message(wire, max_message_bytes=max_message_bytes)
                message_type = message["type"]
                if message_type == "session_open":
                    opened = SessionOpen(
                        session_id=str(message["session_id"]),
                        generation_id=str(message["generation_id"]),
                        output_format=str(message.get("output_format") or "webp"),
                        quality=int(message.get("quality") or 90),
                        preview_max_width=message.get("preview_max_width"),
                    )
                    await worker.open(opened)
                    identity = (opened.session_id, opened.generation_id)
                    await send(
                        encode_message(
                            "session_accepted",
                            session_id=opened.session_id,
                            generation_id=opened.generation_id,
                            credit_chunk_index=0,
                        )
                    )
                    continue
                if message_type == "abort":
                    break
                if message_type != "latent_chunk":
                    raise ProtocolViolation(f"unsupported message type: {message_type}")
                if identity is None:
                    raise ProtocolViolation("session_open must precede latent chunks")

                header = latent_header_from_message(message)
                if (header.session_id, header.generation_id) != identity:
                    raise ProtocolViolation("latent identity does not match WebSocket")
                payload = message.get("payload")
                if not isinstance(payload, bytes):
                    raise ProtocolViolation("latent payload is required")
                validate_payload(header, payload)
                latents = _tensor_from_payload(header, payload)

                async def on_frame_batch(frame_batch, *, header=header):
                    payload_lengths = [len(value) for value in frame_batch.payloads]
                    await send(
                        encode_message(
                            "frame_batch",
                            payload=b"".join(frame_batch.payloads),
                            session_id=header.session_id,
                            generation_id=header.generation_id,
                            request_id=header.request_id,
                            chunk_index=header.chunk_index,
                            event_id=header.event_id,
                            content_type=frame_batch.content_type,
                            width=frame_batch.width,
                            height=frame_batch.height,
                            payload_lengths=payload_lengths,
                            num_frames=frame_batch.num_frames,
                            frame_batch_index=frame_batch.frame_batch_index,
                            is_final_frame_batch=frame_batch.is_final,
                            encode_ms=frame_batch.encode_ms,
                        )
                    )

                future = await worker.submit(
                    header,
                    latents,
                    on_frame_batch=on_frame_batch,
                )
                await send(
                    encode_message(
                        "latent_accepted",
                        session_id=header.session_id,
                        generation_id=header.generation_id,
                        request_id=header.request_id,
                        chunk_index=header.chunk_index,
                        next_credit_chunk_index=header.chunk_index + 1,
                    )
                )

                async def finish_chunk(future=future, header=header):
                    try:
                        result = await future
                        await send(
                            encode_message(
                                "chunk_complete",
                                session_id=header.session_id,
                                generation_id=header.generation_id,
                                request_id=header.request_id,
                                chunk_index=header.chunk_index,
                                num_frames=result.num_frames,
                                queue_wait_ms=result.queue_wait_ms,
                                decode_ms=result.decode_ms,
                                encode_ms=result.encode_ms,
                            )
                        )
                    except Exception as exc:
                        await send(
                            encode_message(
                                "error",
                                request_id=header.request_id,
                                chunk_index=header.chunk_index,
                                error_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )

                task = asyncio.create_task(finish_chunk())
                finish_tasks.add(task)
                task.add_done_callback(finish_tasks.discard)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await send(
                    encode_message(
                        "error",
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
            except Exception:
                pass
        finally:
            for task in finish_tasks:
                task.cancel()
            if finish_tasks:
                await asyncio.gather(*finish_tasks, return_exceptions=True)
            if identity is not None:
                await worker.close(*identity)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="SGLang realtime TAEHV worker")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--max-sessions", type=int, default=8)
    parser.add_argument("--queue-depth-per-session", type=int, default=1)
    parser.add_argument("--max-message-mb", type=int, default=64)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    engine = TAEHVEngine(args.checkpoint_path, device=args.device, dtype=dtype)
    worker = AsyncVAEWorker(
        engine,
        max_sessions=args.max_sessions,
        queue_depth_per_session=args.queue_depth_per_session,
    )
    app = create_app(worker, max_message_bytes=args.max_message_mb * 1024 * 1024)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
