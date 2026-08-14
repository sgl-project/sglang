# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-stage SGLang session used by a VoiceChat WebSocket/audio adapter."""

import time
from dataclasses import dataclass
from typing import Any


def _latest(result: dict[str, Any], key: str):
    values = result.get("meta_info", {}).get(key)
    if not values:
        raise RuntimeError(f"VoiceChat stage did not return {key!r}: {result}")
    return values[-1]


@dataclass
class VoiceChatStep:
    text_token: int
    function_token: int
    audio_codes: list[int]
    duplex_ms: float = 0.0
    eartts_ms: float = 0.0


class SGLangVoiceChatSession:
    """Persistent Duplex/EarTTS KV sessions for one full-duplex stream.

    NeMo remains responsible for converting an 80 ms microphone frame to the
    4480-wide ``acoustic_embedding`` and decoding ``audio_codes`` to 22.05 kHz
    audio. Keeping those operations outside the serving core mirrors NVIDIA's
    published vLLM-Omni integration.
    """

    def __init__(self, duplex_engine, eartts_engine, capacity=8192):
        self.duplex_engine = duplex_engine
        self.eartts_engine = eartts_engine
        self.duplex_session = duplex_engine.open_session(capacity, streaming=True)
        self.eartts_session = eartts_engine.open_session(capacity, streaming=True)
        self.function_token = None
        self.system_prompt_ids = None
        self.pad_token_id = None
        self.previous_audio_codes = None
        self._started = False

    @staticmethod
    def _params(session_id):
        return {"id": session_id, "rid": None}

    def start(
        self, system_prompt_ids: list[int], speaker_latent, pad_token_id: int
    ) -> None:
        if self._started:
            raise RuntimeError("VoiceChat session has already been started.")
        if not system_prompt_ids:
            raise ValueError("VoiceChat requires a non-empty system prompt.")
        self.system_prompt_ids = list(system_prompt_ids)
        self.pad_token_id = int(pad_token_id)
        latent_len = int(speaker_latent.shape[0])
        self.eartts_engine.generate(
            input_ids=[0] * latent_len,
            # EarTTS emits a deterministic placeholder token (id 0). Keeping
            # that token as the session tail makes the first synthesis tick
            # identical to every later tick and preserves the normal
            # one-token autoregressive KV handoff.
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
            session_params=self._params(self.eartts_session),
            custom_inputs={
                "is_speaker_prefill": True,
                "speaker_latent": speaker_latent,
            },
        )
        self._started = True

    def step(self, acoustic_embedding) -> VoiceChatStep:
        if not self._started:
            raise RuntimeError("Call start() before sending acoustic frames.")
        sampling = {"max_new_tokens": 1, "temperature": 0.0}
        duplex_started = time.perf_counter()
        if self.function_token is None:
            input_ids = [*self.system_prompt_ids, self.pad_token_id]
            custom_inputs = {
                "is_initial_prefill": True,
                "prompt_length": len(self.system_prompt_ids),
                "acoustic_embedding": acoustic_embedding,
            }
        else:
            input_ids = []
            custom_inputs = {
                "acoustic_embedding": acoustic_embedding,
                "input_function_ids": [self.function_token],
            }
        duplex = self.duplex_engine.generate(
            input_ids=input_ids,
            sampling_params=sampling,
            session_params=self._params(self.duplex_session),
            custom_inputs=custom_inputs,
        )
        duplex_ms = (time.perf_counter() - duplex_started) * 1000
        text_token = int(duplex["output_ids"][-1])
        self.function_token = int(_latest(duplex, "function_tokens"))

        eartts_started = time.perf_counter()
        eartts = self.eartts_engine.generate(
            input_ids=[],
            sampling_params=sampling,
            session_params=self._params(self.eartts_session),
            custom_inputs={
                "text_token": text_token,
                "previous_audio_codes": self.previous_audio_codes,
            },
        )
        eartts_ms = (time.perf_counter() - eartts_started) * 1000
        codes = [int(code) for code in _latest(eartts, "audio_codes")]
        self.previous_audio_codes = codes
        return VoiceChatStep(
            text_token, self.function_token, codes, duplex_ms, eartts_ms
        )

    def close(self):
        self.duplex_engine.close_session(self.duplex_session)
        self.eartts_engine.close_session(self.eartts_session)
        self._started = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AsyncSGLangVoiceChatSession:
    """Async counterpart for WebSocket servers with an active event loop."""

    def __init__(self, duplex_engine, eartts_engine):
        self.duplex_engine = duplex_engine
        self.eartts_engine = eartts_engine
        self.duplex_session = None
        self.eartts_session = None
        self.function_token = None
        self.system_prompt_ids = None
        self.pad_token_id = None
        self.previous_audio_codes = None
        self._started = False

    @classmethod
    async def create(cls, duplex_engine, eartts_engine, capacity=8192):
        self = cls(duplex_engine, eartts_engine)
        self.duplex_session = await duplex_engine.async_open_session(
            capacity, streaming=True
        )
        try:
            self.eartts_session = await eartts_engine.async_open_session(
                capacity, streaming=True
            )
        except Exception:
            await duplex_engine.async_close_session(self.duplex_session)
            raise
        return self

    async def start(
        self, system_prompt_ids: list[int], speaker_latent, pad_token_id: int
    ) -> None:
        if self._started:
            raise RuntimeError("VoiceChat session has already been started.")
        if not system_prompt_ids:
            raise ValueError("VoiceChat requires a non-empty system prompt.")
        self.system_prompt_ids = list(system_prompt_ids)
        self.pad_token_id = int(pad_token_id)
        sampling = {"max_new_tokens": 1, "temperature": 0.0}
        await self.eartts_engine.async_generate(
            input_ids=[0] * int(speaker_latent.shape[0]),
            sampling_params=sampling,
            session_params=SGLangVoiceChatSession._params(self.eartts_session),
            custom_inputs={
                "is_speaker_prefill": True,
                "speaker_latent": speaker_latent,
            },
        )
        self._started = True

    async def duplex_step(self, acoustic_embedding) -> tuple[int, int, float]:
        if not self._started:
            raise RuntimeError("Call start() before sending acoustic frames.")
        sampling = {"max_new_tokens": 1, "temperature": 0.0}
        duplex_started = time.perf_counter()
        if self.function_token is None:
            input_ids = [*self.system_prompt_ids, self.pad_token_id]
            custom_inputs = {
                "is_initial_prefill": True,
                "prompt_length": len(self.system_prompt_ids),
                "acoustic_embedding": acoustic_embedding,
            }
        else:
            input_ids = []
            custom_inputs = {
                "acoustic_embedding": acoustic_embedding,
                "input_function_ids": [self.function_token],
            }
        duplex = await self.duplex_engine.async_generate(
            input_ids=input_ids,
            sampling_params=sampling,
            session_params=SGLangVoiceChatSession._params(self.duplex_session),
            custom_inputs=custom_inputs,
        )
        duplex_ms = (time.perf_counter() - duplex_started) * 1000
        text_token = int(duplex["output_ids"][-1])
        self.function_token = int(_latest(duplex, "function_tokens"))
        return text_token, self.function_token, duplex_ms

    async def eartts_step(self, text_token: int) -> tuple[list[int], float]:
        if not self._started:
            raise RuntimeError("Call start() before sending acoustic frames.")
        sampling = {"max_new_tokens": 1, "temperature": 0.0}
        eartts_started = time.perf_counter()
        eartts = await self.eartts_engine.async_generate(
            input_ids=[],
            sampling_params=sampling,
            session_params=SGLangVoiceChatSession._params(self.eartts_session),
            custom_inputs={
                "text_token": text_token,
                "previous_audio_codes": self.previous_audio_codes,
            },
        )
        eartts_ms = (time.perf_counter() - eartts_started) * 1000
        codes = [int(code) for code in _latest(eartts, "audio_codes")]
        self.previous_audio_codes = codes
        return codes, eartts_ms

    async def step(self, acoustic_embedding) -> VoiceChatStep:
        text_token, function_token, duplex_ms = await self.duplex_step(
            acoustic_embedding
        )
        codes, eartts_ms = await self.eartts_step(text_token)
        return VoiceChatStep(text_token, function_token, codes, duplex_ms, eartts_ms)

    async def close(self):
        if self.duplex_session is not None:
            await self.duplex_engine.async_close_session(self.duplex_session)
            self.duplex_session = None
        if self.eartts_session is not None:
            await self.eartts_engine.async_close_session(self.eartts_session)
            self.eartts_session = None
        self._started = False
