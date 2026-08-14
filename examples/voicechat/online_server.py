#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single-user WebSocket server for SGLang NemotronLabs VoiceChat.

The two autoregressive stages run in SGLang. A small NeMo sidecar supplies
the checkpoint's streaming perception encoder and audio codec.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoConfig, AutoTokenizer

from examples.voicechat.online_session import AsyncSGLangVoiceChatSession
from sglang import Engine
from sglang.srt.configs.eartts import EarTTSConfig  # noqa: F401

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise voice assistant."
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 22_050
FRAME_SAMPLES = 1_280


class AudioSidecarClient:
    def __init__(self, url: str):
        self.url = url.rstrip("/")

    def _request(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"Audio sidecar returned HTTP {error.code}: {detail}"
            ) from error

    def health(self):
        return self._request("GET", "/health")

    def start(self) -> str:
        return self._request("POST", "/session")["session_id"]

    def encode(self, session_id: str, pcm16: str) -> torch.Tensor:
        result = self._request(
            "POST", f"/session/{session_id}/encode", {"pcm16": pcm16}
        )
        embedding = np.frombuffer(
            base64.b64decode(result["embedding"]), dtype="<f4"
        ).copy()
        return torch.from_numpy(embedding.reshape(result["shape"]))

    def decode(self, session_id: str, codes: list[int]):
        return self._request("POST", f"/session/{session_id}/decode", {"codes": codes})

    def close(self, session_id: str):
        try:
            self._request("DELETE", f"/session/{session_id}")
        except Exception:
            pass


def _load_speaker(model_dir: Path, speaker_latent_path: str | None):
    if speaker_latent_path is None:
        candidates = sorted((model_dir / "speaker_latents").glob("*.pt"))
        if not candidates:
            raise FileNotFoundError("No converted speaker latent was found.")
        speaker_latent_path = str(candidates[0])
    speaker = torch.load(speaker_latent_path, map_location="cpu", weights_only=True)
    if speaker.ndim == 3 and speaker.shape[0] == 1:
        speaker = speaker.squeeze(0)
    return speaker


class VoiceChatRuntime:
    def __init__(self, args):
        self.sidecar = AudioSidecarClient(args.audio_sidecar)
        self.sidecar.health()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.duplex_model, trust_remote_code=False
        )
        self.config = AutoConfig.from_pretrained(
            args.duplex_model, trust_remote_code=False
        )
        self.speaker = _load_speaker(Path(args.eartts_model), args.speaker_latent)
        common = {
            "max_running_requests": 2,
            "skip_tokenizer_init": True,
            "enable_streaming_session": True,
            "log_level": args.log_level,
        }
        self.duplex = Engine(
            model_path=args.duplex_model,
            dtype="bfloat16",
            mem_fraction_static=args.duplex_memory_fraction,
            context_length=args.context_length,
            base_gpu_id=args.duplex_base_gpu_id,
            **common,
        )
        previous_tf32 = os.environ.get("NVIDIA_TF32_OVERRIDE")
        os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        try:
            self.eartts = Engine(
                model_path=args.eartts_model,
                dtype="float32",
                mem_fraction_static=args.eartts_memory_fraction,
                context_length=args.context_length,
                base_gpu_id=args.eartts_base_gpu_id,
                attention_backend=args.eartts_attention_backend,
                **common,
            )
        finally:
            if previous_tf32 is None:
                os.environ.pop("NVIDIA_TF32_OVERRIDE", None)
            else:
                os.environ["NVIDIA_TF32_OVERRIDE"] = previous_tf32
        self.connection_lock = asyncio.Lock()
        self.max_audio_queue_frames = args.max_audio_queue_frames
        self.warmup_frames = 0 if args.skip_warmup else args.warmup_frames
        self.warmup_duration_ms = None
        self.ready = False

    def prompt_ids(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return [self.config.bos_token_id, *ids, self.config.eos_token_id]

    def shutdown(self):
        self.duplex.shutdown()
        self.eartts.shutdown()

    async def warmup(self):
        if self.warmup_frames == 0:
            self.ready = True
            return

        started_at = time.perf_counter()
        audio_session = None
        model_session = None
        silence = base64.b64encode(bytes(FRAME_SAMPLES * 2)).decode()
        try:
            audio_session = await asyncio.to_thread(self.sidecar.start)
            model_session = await AsyncSGLangVoiceChatSession.create(
                self.duplex, self.eartts, capacity=8192
            )
            await model_session.start(
                self.prompt_ids(DEFAULT_SYSTEM_PROMPT),
                self.speaker,
                self.config.pad_token_id,
            )
            for _ in range(self.warmup_frames):
                embedding = await asyncio.to_thread(
                    self.sidecar.encode, audio_session, silence
                )
                result = await model_session.step(embedding)
                await asyncio.to_thread(
                    self.sidecar.decode, audio_session, result.audio_codes
                )
        finally:
            try:
                if model_session is not None:
                    await model_session.close()
            finally:
                if audio_session is not None:
                    await asyncio.to_thread(self.sidecar.close, audio_session)

        self.warmup_duration_ms = (time.perf_counter() - started_at) * 1000
        self.ready = True
        print(
            f"VoiceChat warm-up completed in {self.warmup_duration_ms:.1f} ms "
            f"using {self.warmup_frames} silent frames.",
            flush=True,
        )


def create_app(runtime: VoiceChatRuntime) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        try:
            await runtime.warmup()
            yield
        finally:
            runtime.ready = False
            runtime.shutdown()

    app = FastAPI(title="SGLang NVIDIA NemotronLabs VoiceChat", lifespan=lifespan)

    @app.get("/")
    async def discovery():
        return {
            "service": "sglang-nemotron-voicechat",
            "websocket": "/v1/realtime",
            "websocket_alias": "/realtime",
            "health": "/v1/realtime/health",
            "input_sample_rate": INPUT_SAMPLE_RATE,
            "output_sample_rate": OUTPUT_SAMPLE_RATE,
        }

    @app.get("/health")
    @app.get("/v1/realtime/health")
    async def health():
        return {
            "ready": runtime.ready,
            "input_format": "pcm16",
            "input_sample_rate": INPUT_SAMPLE_RATE,
            "output_format": "pcm16",
            "output_sample_rate": OUTPUT_SAMPLE_RATE,
            "frame_samples": FRAME_SAMPLES,
            "single_session": True,
            "max_audio_queue_frames": runtime.max_audio_queue_frames,
            "warmup": {
                "enabled": runtime.warmup_frames > 0,
                "frames": runtime.warmup_frames,
                "duration_ms": runtime.warmup_duration_ms,
            },
            "audio_sidecar": runtime.sidecar.health(),
        }

    @app.websocket("/v1/realtime")
    @app.websocket("/realtime")
    async def realtime(websocket: WebSocket):
        await websocket.accept()
        if runtime.connection_lock.locked():
            await websocket.send_json(
                {"type": "error", "error": {"message": "Server is busy."}}
            )
            await websocket.close(code=1013)
            return

        async with runtime.connection_lock:
            audio_session = None
            model_session = None
            started = False
            audio_enqueued = False
            prompt = DEFAULT_SYSTEM_PROMPT
            worker_tasks = []
            queues = [asyncio.Queue(runtime.max_audio_queue_frames) for _ in range(4)]
            send_lock = asyncio.Lock()
            frame_timings = []
            try:

                async def send(event):
                    async with send_lock:
                        await websocket.send_json(event)

                await send(
                    {
                        "type": "session.created",
                        "session": {
                            "input_audio_format": "pcm16",
                            "input_sample_rate": INPUT_SAMPLE_RATE,
                            "output_audio_format": "pcm16",
                            "output_sample_rate": OUTPUT_SAMPLE_RATE,
                            "frame_samples": FRAME_SAMPLES,
                        },
                    }
                )
                audio_session = await asyncio.to_thread(runtime.sidecar.start)
                model_session = await AsyncSGLangVoiceChatSession.create(
                    runtime.duplex, runtime.eartts, capacity=8192
                )

                async def ensure_started():
                    nonlocal started
                    if not started:
                        await model_session.start(
                            runtime.prompt_ids(prompt),
                            runtime.speaker,
                            runtime.config.pad_token_id,
                        )
                        started = True

                async def perception_worker():
                    while True:
                        kind, payload, timing = await queues[0].get()
                        try:
                            if kind != "audio":
                                await queues[1].put((kind, payload, timing))
                                if kind == "stop":
                                    return
                                continue
                            timing["perception_started"] = time.perf_counter()
                            embedding = await asyncio.to_thread(
                                runtime.sidecar.encode,
                                audio_session,
                                payload,
                            )
                            timing["perception_done"] = time.perf_counter()
                            await queues[1].put((kind, embedding, timing))
                        finally:
                            queues[0].task_done()

                async def duplex_worker():
                    while True:
                        kind, payload, timing = await queues[1].get()
                        try:
                            if kind != "audio":
                                await queues[2].put((kind, payload, timing))
                                if kind == "stop":
                                    return
                                continue
                            text_token, function_token, duplex_ms = (
                                await model_session.duplex_step(payload)
                            )
                            timing["duplex"] = duplex_ms
                            await queues[2].put(
                                (
                                    kind,
                                    {
                                        "text_token": text_token,
                                        "function_token": function_token,
                                    },
                                    timing,
                                )
                            )
                        finally:
                            queues[1].task_done()

                async def eartts_worker():
                    while True:
                        kind, payload, timing = await queues[2].get()
                        try:
                            if kind != "audio":
                                await queues[3].put((kind, payload, timing))
                                if kind == "stop":
                                    return
                                continue
                            codes, eartts_ms = await model_session.eartts_step(
                                payload["text_token"]
                            )
                            timing["eartts"] = eartts_ms
                            payload["audio_codes"] = codes
                            await queues[3].put((kind, payload, timing))
                        finally:
                            queues[2].task_done()

                async def codec_worker():
                    last_output_at = None
                    while True:
                        kind, payload, timing = await queues[3].get()
                        try:
                            if kind == "stop":
                                return
                            if kind == "commit":
                                count = len(frame_timings)

                                def summarize(name):
                                    values = sorted(
                                        item[name]
                                        for item in frame_timings
                                        if name in item
                                    )
                                    if not values:
                                        return {"mean": 0, "p95": 0}
                                    return {
                                        "mean": sum(values) / len(values),
                                        "p95": values[
                                            min(
                                                len(values) - 1,
                                                int(len(values) * 0.95),
                                            )
                                        ],
                                    }

                                total = summarize("total")
                                await send(
                                    {
                                        "type": "input_audio_buffer.committed",
                                        "timing_ms": {
                                            "frames": count,
                                            **total,
                                            "stages": {
                                                name: summarize(name)
                                                for name in (
                                                    "queue",
                                                    "perception",
                                                    "duplex",
                                                    "eartts",
                                                    "codec",
                                                    "total",
                                                    "output_interval",
                                                )
                                            },
                                        },
                                    }
                                )
                                continue
                            codec_started = time.perf_counter()
                            decoded = await asyncio.to_thread(
                                runtime.sidecar.decode,
                                audio_session,
                                payload["audio_codes"],
                            )
                            codec_done = time.perf_counter()
                            result_timing = {
                                "queue": (
                                    timing["perception_started"] - timing["enqueued"]
                                )
                                * 1000,
                                "perception": (
                                    timing["perception_done"]
                                    - timing["perception_started"]
                                )
                                * 1000,
                                "duplex": timing["duplex"],
                                "eartts": timing["eartts"],
                                "models": timing["duplex"] + timing["eartts"],
                                "codec": (codec_done - codec_started) * 1000,
                                "total": (codec_done - timing["perception_started"])
                                * 1000,
                            }
                            if last_output_at is not None:
                                result_timing["output_interval"] = (
                                    codec_done - last_output_at
                                ) * 1000
                            last_output_at = codec_done
                            frame_timings.append(result_timing)
                            await send(
                                {
                                    "type": "response.output_audio.delta",
                                    "delta": decoded["pcm16"],
                                    "sample_rate": decoded["sample_rate"],
                                    "samples": decoded["samples"],
                                    "text_token": payload["text_token"],
                                    "function_token": payload["function_token"],
                                    "timing_ms": result_timing,
                                }
                            )
                        finally:
                            queues[3].task_done()

                async def run_worker(worker):
                    try:
                        await worker()
                    except Exception as error:
                        try:
                            await send(
                                {
                                    "type": "error",
                                    "error": {"message": str(error)},
                                }
                            )
                            await websocket.close(code=1011)
                        except Exception:
                            pass
                        raise

                worker_tasks = [
                    asyncio.create_task(run_worker(worker))
                    for worker in (
                        perception_worker,
                        duplex_worker,
                        eartts_worker,
                        codec_worker,
                    )
                ]

                async def raise_worker_failure():
                    for task in worker_tasks:
                        if task.done():
                            await task

                while True:
                    message = await websocket.receive_json()
                    await raise_worker_failure()
                    kind = message.get("type")
                    if kind == "session.update":
                        if started or audio_enqueued:
                            raise ValueError(
                                "session.update must precede the first audio frame."
                            )
                        session_options = message.get("session", {})
                        prompt = session_options.get("instructions") or prompt
                        await ensure_started()
                        await send(
                            {
                                "type": "session.updated",
                                "session": {
                                    "instructions": prompt,
                                    "input_sample_rate": INPUT_SAMPLE_RATE,
                                    "output_sample_rate": OUTPUT_SAMPLE_RATE,
                                },
                            }
                        )
                    elif kind == "input_audio_buffer.append":
                        encoded = base64.b64decode(
                            message.get("audio", ""), validate=True
                        )
                        if len(encoded) != FRAME_SAMPLES * 2:
                            raise ValueError(
                                f"Each frame must contain {FRAME_SAMPLES} mono "
                                f"PCM16 samples ({FRAME_SAMPLES * 2} bytes)."
                            )
                        await ensure_started()
                        audio_enqueued = True
                        try:
                            queues[0].put_nowait(
                                (
                                    "audio",
                                    message["audio"],
                                    {"enqueued": time.perf_counter()},
                                )
                            )
                        except asyncio.QueueFull as error:
                            raise RuntimeError(
                                "Audio backlog exceeded "
                                f"{runtime.max_audio_queue_frames} frames."
                            ) from error
                    elif kind == "input_audio_buffer.commit":
                        try:
                            queues[0].put_nowait(
                                ("commit", None, {"enqueued": time.perf_counter()})
                            )
                        except asyncio.QueueFull as error:
                            raise RuntimeError(
                                "Audio backlog is full; commit was not queued."
                            ) from error
                    elif kind == "session.close":
                        await queues[0].put(
                            ("stop", None, {"enqueued": time.perf_counter()})
                        )
                        await asyncio.gather(*worker_tasks)
                        await websocket.close()
                        break
                    else:
                        raise ValueError(f"Unsupported event type: {kind!r}")
            except WebSocketDisconnect:
                pass
            except Exception as error:
                try:
                    await websocket.send_json(
                        {"type": "error", "error": {"message": str(error)}}
                    )
                    await websocket.close(code=1008)
                except Exception:
                    pass
            finally:
                for task in worker_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*worker_tasks, return_exceptions=True)
                try:
                    if model_session is not None:
                        await model_session.close()
                finally:
                    if audio_session is not None:
                        await asyncio.to_thread(runtime.sidecar.close, audio_session)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duplex-model", required=True)
    parser.add_argument("--eartts-model", required=True)
    parser.add_argument("--speaker-latent")
    parser.add_argument("--audio-sidecar", default="http://127.0.0.1:18081")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--duplex-memory-fraction", type=float, default=0.45)
    parser.add_argument("--eartts-memory-fraction", type=float, default=0.20)
    parser.add_argument("--duplex-base-gpu-id", type=int, default=0)
    parser.add_argument("--eartts-base-gpu-id", type=int, default=0)
    parser.add_argument("--eartts-attention-backend", default="torch_native")
    parser.add_argument("--max-audio-queue-frames", type=int, default=256)
    parser.add_argument("--warmup-frames", type=int, default=2)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()
    if args.warmup_frames < 1:
        parser.error("--warmup-frames must be at least 1")
    runtime = VoiceChatRuntime(args)
    uvicorn.run(create_app(runtime), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
