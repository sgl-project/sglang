# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol

import torch

from sglang.srt.ug.context import UGContextBundle, UGContextHandle
from sglang.srt.ug.runtime import (
    FakeUGModelRunner,
    UGDecodeResult,
    UGLatentDecodeRequest,
    UGLatentPrepareRequest,
    UGLatentPrepareResult,
    UGSessionRuntime,
    UGVelocityRequest,
)


class UGDenoiserBridge(Protocol):
    def build_contexts(
        self, *, prompt: str | list[str] | None, image: Any | None
    ) -> UGContextBundle: ...

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor: ...

    def release_contexts(self, contexts: UGContextBundle) -> None: ...

    def prepare_latents(
        self,
        *,
        contexts: UGContextBundle,
        sampling_params: Any,
        seed: int | None,
    ) -> UGLatentPrepareResult | None: ...

    def append_generated_image(
        self, *, contexts: UGContextBundle, image: Any | None
    ) -> None: ...

    def decode_latents(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        sampling_params: Any,
    ) -> Any | None: ...

    def decode_next_segment(self, *, contexts: UGContextBundle) -> UGDecodeResult: ...


class FakeUGDenoiserBridge:
    def build_contexts(
        self, *, prompt: str | list[str] | None, image: Any | None
    ) -> UGContextBundle:
        prompt_text = " ".join(prompt) if isinstance(prompt, list) else prompt or ""
        text_tokens = len(prompt_text.split())
        image_tokens = 2 if image is not None else 0
        return UGContextBundle(
            full=UGContextHandle("full", text_tokens + image_tokens),
            text_cfg=UGContextHandle("text_cfg", image_tokens),
            image_cfg=UGContextHandle("image_cfg", text_tokens),
        )

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor:
        del latent_position_ids, sampling_params
        scale = 1.0 + contexts.full.token_count * 0.01
        return latent_tokens + scale * timestep.reshape(-1, 1, 1).to(latent_tokens)

    def release_contexts(self, contexts: UGContextBundle) -> None:
        del contexts

    def prepare_latents(
        self,
        *,
        contexts: UGContextBundle,
        sampling_params: Any,
        seed: int | None,
    ) -> UGLatentPrepareResult | None:
        del contexts, sampling_params, seed
        return None

    def append_generated_image(
        self, *, contexts: UGContextBundle, image: Any | None
    ) -> None:
        del contexts, image

    def decode_latents(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        sampling_params: Any,
    ) -> Any | None:
        del contexts, latent_tokens, sampling_params
        return None

    def decode_next_segment(self, *, contexts: UGContextBundle) -> UGDecodeResult:
        del contexts
        return UGDecodeResult(type="done")


class SRTBackedUGDenoiserBridge:
    """Diffusion-side bridge that delegates UG model work to SRT runtime state."""

    def __init__(self, runtime: UGSessionRuntime | None = None) -> None:
        self.runtime = runtime or UGSessionRuntime(model_runner=FakeUGModelRunner())

    def build_contexts(
        self, *, prompt: str | list[str] | None, image: Any | None
    ) -> UGContextBundle:
        messages = self.runtime.normalize_messages(prompt=prompt, image=image)
        session = self.runtime.prefill_interleaved(messages)
        segment = self.runtime.decode_next_segment(session)
        if segment.type != "image_marker":
            raise ValueError(
                "UG denoise bridge expected U decode to request an image segment, "
                f"got {segment.type}"
            )

        prompt_text = " ".join(prompt) if isinstance(prompt, list) else prompt or ""
        text_tokens = len(prompt_text.split())
        image_tokens = 2 if image is not None else 0
        return UGContextBundle(
            full=UGContextHandle(
                session.anchor_request_id,
                session.context_length,
                session=session,
            ),
            text_cfg=UGContextHandle(
                f"{session.anchor_request_id}:text_cfg",
                image_tokens,
                session=session,
            ),
            image_cfg=UGContextHandle(
                f"{session.anchor_request_id}:image_cfg",
                text_tokens,
                session=session,
            ),
        )

    def predict_velocity(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        timestep: torch.Tensor,
        latent_position_ids: torch.Tensor,
        sampling_params: Any,
    ) -> torch.Tensor:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        response = self.runtime.predict_velocity(
            UGVelocityRequest(
                session=contexts.full.session,
                latent_tokens=latent_tokens,
                timestep=timestep,
                latent_position_ids=latent_position_ids,
                sampling_params=sampling_params,
            )
        )
        contexts.full.session = response.session
        contexts.text_cfg.session = response.session
        contexts.image_cfg.session = response.session
        return response.velocity

    def prepare_latents(
        self,
        *,
        contexts: UGContextBundle,
        sampling_params: Any,
        seed: int | None,
    ) -> UGLatentPrepareResult | None:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        return self.runtime.prepare_latents(
            UGLatentPrepareRequest(
                session=contexts.full.session,
                sampling_params=sampling_params,
                seed=seed,
            )
        )

    def release_contexts(self, contexts: UGContextBundle) -> None:
        if contexts.full.session is not None:
            self.runtime.close_session(contexts.full.session)

    def append_generated_image(
        self, *, contexts: UGContextBundle, image: Any | None
    ) -> None:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        session = self.runtime.append_generated_image(
            contexts.full.session, image=image
        )
        contexts.full.request_id = session.anchor_request_id
        contexts.full.token_count = session.context_length
        contexts.full.session = session
        contexts.text_cfg.session = session
        contexts.image_cfg.session = session

    def decode_latents(
        self,
        *,
        contexts: UGContextBundle,
        latent_tokens: torch.Tensor,
        sampling_params: Any,
    ) -> Any | None:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        return self.runtime.decode_latents_to_image(
            UGLatentDecodeRequest(
                session=contexts.full.session,
                latent_tokens=latent_tokens,
                sampling_params=sampling_params,
            )
        )

    def decode_next_segment(self, *, contexts: UGContextBundle) -> UGDecodeResult:
        if contexts.full.session is None:
            raise ValueError("SRT-backed UG contexts require a session handle")
        return self.runtime.decode_next_segment(contexts.full.session)
