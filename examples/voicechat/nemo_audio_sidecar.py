#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo perception/codec sidecar for SGLang VoiceChat online inference.

Run this file in NVIDIA's VoiceChat runtime image. It loads only the trained
perception encoder and audio codec from the unified checkpoint; Duplex and
EarTTS remain resident in SGLang. The service is intentionally single-session,
matching the current one-H100 deployment capacity.
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from nemo.collections.speechlm2.inference.model_wrappers.perception_cache import (
    PerceptionCacheManager,
)
from nemo.collections.speechlm2.models.duplex_ear_tts import (
    replace_control_speech_codes,
)
from nemo.collections.speechlm2.modules.ear_tts_vae_codec import (
    CausalConv1dCache,
    RVQVAEModel,
)
from nemo.collections.speechlm2.parts.pretrained import setup_speech_encoder
from omegaconf import DictConfig
from pydantic import BaseModel
from safetensors import safe_open
from torch import nn

INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 22_050
FRAME_SAMPLES = 1_280


class AudioRequest(BaseModel):
    pcm16: str


class CodesRequest(BaseModel):
    codes: list[int]


class _PerceptionHolder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = DictConfig(cfg)


def _copy_checkpoint_prefix(module, checkpoint: Path, prefix: str) -> int:
    targets = module.state_dict()
    copied = 0
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        for outer_name in handle.keys():
            if not outer_name.startswith(prefix):
                continue
            name = outer_name[len(prefix) :]
            target = targets.get(name)
            if target is None:
                continue
            source = handle.get_tensor(outer_name)
            if source.shape != target.shape:
                raise ValueError(
                    f"Shape mismatch for {outer_name}: checkpoint "
                    f"{tuple(source.shape)}, module {tuple(target.shape)}"
                )
            target.copy_(source)
            copied += 1
    if copied == 0:
        raise ValueError(f"No checkpoint tensors matched {prefix!r}")
    return copied


class NeMoAudioRuntime:
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = "cuda",
        use_perception_cudagraph: bool = True,
    ):
        root = Path(checkpoint_dir)
        weights = root / "model.safetensors"
        config = json.loads((root / "config.json").read_text())
        self.device = torch.device(device)

        stt_cfg = config["model"]["stt"]["model"]
        holder = _PerceptionHolder(stt_cfg)
        setup_speech_encoder(holder, pretrained_weights=False)
        perception_count = _copy_checkpoint_prefix(
            holder.perception, weights, "stt_model.perception."
        )
        self.perception = holder.perception.to(self.device, torch.float32).eval()

        codec_cfg = config["model"]["speech_generation"]["model"]["codec_config"]
        self.codec = RVQVAEModel(DictConfig(codec_cfg))
        codec_count = _copy_checkpoint_prefix(
            self.codec, weights, "tts_model.audio_codec."
        )
        self.codec = self.codec.to(self.device, torch.float32).eval()
        for parameter in self.perception.parameters():
            parameter.requires_grad = False
        for parameter in self.codec.parameters():
            parameter.requires_grad = False

        with safe_open(weights, framework="pt", device="cpu") as handle:
            self.control_codes = handle.get_tensor("tts_model._control_codes").to(
                self.device
            )
            self.silence_codes = handle.get_tensor("tts_model.codec_silence_tokens").to(
                self.device
            )

        model_view = SimpleNamespace(
            stt_model=SimpleNamespace(perception=self.perception)
        )
        self.perception_cache_manager = PerceptionCacheManager(
            model_view,
            device=self.device,
            dtype=torch.float32,
            use_cudagraph=use_perception_cudagraph,
        )
        if not self.perception_cache_manager.setup():
            raise RuntimeError("The VoiceChat perception encoder is not streamable.")

        self.lock = threading.Lock()
        self.session_id: str | None = None
        self.audio_buffer: torch.Tensor | None = None
        self.frame_index = 0
        self.perception_cache = None
        self.codec_cache = None
        self.loaded_tensors = {
            "perception": perception_count,
            "codec": codec_count,
        }
        self.use_perception_cudagraph = use_perception_cudagraph

    def start(self) -> str:
        with self.lock:
            if self.session_id is not None:
                raise RuntimeError("An audio session is already active.")
            self.session_id = str(uuid.uuid4())
            self.audio_buffer = torch.empty(
                (1, 0), dtype=torch.float32, device=self.device
            )
            self.frame_index = 0
            self.perception_cache = self.perception_cache_manager.get_initial_state(
                batch_size=1
            )
            self.codec_cache = CausalConv1dCache()
            return self.session_id

    def _require_session(self, session_id: str):
        if session_id != self.session_id:
            raise KeyError("Audio session not found.")

    @torch.inference_mode()
    def encode(self, session_id: str, pcm16_base64: str) -> np.ndarray:
        with self.lock:
            self._require_session(session_id)
            raw = base64.b64decode(pcm16_base64, validate=True)
            pcm = np.frombuffer(raw, dtype="<i2")
            if pcm.size != FRAME_SAMPLES:
                raise ValueError(
                    f"Expected {FRAME_SAMPLES} PCM16 samples, got {pcm.size}."
                )
            frame = torch.from_numpy(pcm.astype(np.float32) / 32768.0).to(self.device)
            self.audio_buffer = torch.cat(
                (self.audio_buffer, frame.unsqueeze(0)), dim=1
            )
            encoded, self.perception_cache, _ = self.perception_cache_manager.step(
                audio_input=self.audio_buffer,
                frame_idx=self.frame_index,
                num_frames_per_chunk=1,
                perception_cache=self.perception_cache,
            )
            self.frame_index += 1
            if encoded.shape[1] != 1:
                raise RuntimeError(
                    f"Expected one encoded frame, got {tuple(encoded.shape)}"
                )
            return encoded[:, 0, :].float().cpu().numpy()

    @torch.inference_mode()
    def decode(self, session_id: str, codes: list[int]) -> bytes:
        with self.lock:
            self._require_session(session_id)
            if len(codes) != int(self.silence_codes.numel()):
                raise ValueError(
                    f"Expected {self.silence_codes.numel()} codec IDs, "
                    f"got {len(codes)}."
                )
            tensor = torch.tensor(codes, dtype=torch.long, device=self.device).reshape(
                1, 1, -1
            )
            tensor = replace_control_speech_codes(
                tensor, self.control_codes, self.silence_codes
            )
            lengths = torch.ones(1, dtype=torch.long, device=self.device)
            audio, _ = self.codec.decode(tensor, lengths, cache=self.codec_cache)
            samples = audio.reshape(-1).float().cpu().numpy()
            pcm = np.clip(samples, -1.0, 1.0)
            return (pcm * 32767.0).astype("<i2").tobytes()

    def close(self, session_id: str):
        with self.lock:
            self._require_session(session_id)
            self.session_id = None
            self.audio_buffer = None
            self.perception_cache = None
            self.codec_cache = None
            self.frame_index = 0


def create_app(runtime: NeMoAudioRuntime) -> FastAPI:
    app = FastAPI(title="NeMo audio sidecar for SGLang VoiceChat")

    @app.get("/health")
    def health():
        return {
            "ready": True,
            "input_sample_rate": INPUT_SAMPLE_RATE,
            "output_sample_rate": OUTPUT_SAMPLE_RATE,
            "frame_samples": FRAME_SAMPLES,
            "active_session": runtime.session_id,
            "loaded_tensors": runtime.loaded_tensors,
            "perception_cudagraph": runtime.use_perception_cudagraph,
        }

    @app.post("/session")
    def start():
        try:
            return {"session_id": runtime.start()}
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/session/{session_id}/encode")
    def encode(session_id: str, request: AudioRequest):
        try:
            embedding = runtime.encode(session_id, request.pcm16)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except (ValueError, RuntimeError) as error:
            raise HTTPException(400, str(error)) from error
        return {
            "shape": list(embedding.shape),
            "dtype": "float32",
            "embedding": base64.b64encode(embedding.tobytes()).decode(),
        }

    @app.post("/session/{session_id}/decode")
    def decode(session_id: str, request: CodesRequest):
        try:
            pcm = runtime.decode(session_id, request.codes)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        return {
            "sample_rate": OUTPUT_SAMPLE_RATE,
            "pcm16": base64.b64encode(pcm).decode(),
            "samples": len(pcm) // 2,
        }

    @app.delete("/session/{session_id}")
    def close(session_id: str):
        try:
            runtime.close(session_id)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        return {"closed": True}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-perception-cudagraph", action="store_true")
    args = parser.parse_args()
    runtime = NeMoAudioRuntime(
        args.checkpoint,
        args.device,
        use_perception_cudagraph=not args.disable_perception_cudagraph,
    )
    uvicorn.run(create_app(runtime), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
