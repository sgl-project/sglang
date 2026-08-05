# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.pipelines.minwm_causal_dmd_pipeline import (
    _use_remote_realtime_vae,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.latent_handoff import (
    RealtimeLatentHandoffStage,
)


def _req(*, block_idx=3, image_latent=None):
    return SimpleNamespace(
        latents=torch.randn(1, 48, 2, 30, 52, dtype=torch.float32),
        image_latent=image_latent,
        realtime_session_id="session-1",
        realtime_generation_id="generation-1",
        request_id="request-1",
        block_idx=block_idx,
        realtime_event_id=7,
        realtime_action_version=8,
        realtime_prompt_version=4,
        realtime_output_format="webp",
        realtime_preview_max_width=560,
        trajectory_timesteps=None,
        trajectory_latents=None,
        rollout_trajectory_data=None,
        metrics=object(),
    )


def test_handoff_returns_contiguous_bf16_latents_without_decoding():
    stage = object.__new__(RealtimeLatentHandoffStage)
    req = _req()

    out = stage.forward(req, SimpleNamespace())

    assert out.output is None
    assert out.realtime_latents.dtype == torch.bfloat16
    assert out.realtime_latents.is_contiguous()
    assert out.realtime_handoff == {
        "session_id": "session-1",
        "generation_id": "generation-1",
        "request_id": "request-1",
        "chunk_index": 3,
        "event_id": 7,
        "action_version": 8,
        "prompt_version": 4,
        "has_reference": False,
        "generated_latent_frames": 2,
        "output_format": "webp",
        "preview_max_width": 560,
    }
    assert out.metrics is req.metrics


def test_handoff_prepends_i2v_reference_only_for_first_chunk():
    stage = object.__new__(RealtimeLatentHandoffStage)
    reference = torch.randn(1, 48, 1, 30, 52)
    first = _req(block_idx=0, image_latent=reference)
    later = _req(block_idx=1, image_latent=reference)

    first_out = stage.forward(first, SimpleNamespace())
    later_out = stage.forward(later, SimpleNamespace())

    assert first_out.realtime_latents.shape[2] == 3
    assert first_out.realtime_handoff["has_reference"] is True
    assert later_out.realtime_latents.shape[2] == 2
    assert later_out.realtime_handoff["has_reference"] is True


def test_minwm_pipeline_remote_vae_is_feature_flagged():
    assert not _use_remote_realtime_vae(
        SimpleNamespace(
            realtime_vae_worker_url=None,
            realtime_remote_vae_enabled=False,
        )
    )
    assert _use_remote_realtime_vae(
        SimpleNamespace(
            realtime_vae_worker_url="ws://vae:18081/v1/realtime_vae/decode",
            realtime_remote_vae_enabled=False,
        )
    )
    assert _use_remote_realtime_vae(
        SimpleNamespace(
            realtime_vae_worker_url=None,
            realtime_remote_vae_enabled=True,
        )
    )
