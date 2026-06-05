# SPDX-License-Identifier: Apache-2.0
"""Realtime chain stages — latent-prep plan arithmetic + noise discipline.

The chunk PLAN must reproduce the engine-era tick behavior exactly:
front-loaded segments within a fixed horizon (chunk 0 carries the remainder
and the conditioning frame), uniform chunks past the horizon (seamless
continuation), and the noise buffer sliced bitwise within the horizon.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime import server_args as _sa_mod
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.realtime_chain import (
    SanaWMNoiseState,
    SanaWMRealtimeLatentPrepStage,
    SanaWMSessionInputsState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming import (
    SanaWMStreamCacheState,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

MC = 8


@pytest.fixture
def _global_args():
    prev = _sa_mod._global_server_args
    set_global_server_args(
        SimpleNamespace(
            comfyui_mode=False,
            enable_cfg_parallel=False,
            enable_torch_compile=False,
            attention_backend=None,
        )
    )
    try:
        yield
    finally:
        set_global_server_args(prev)


def _prep_stage():
    return SanaWMRealtimeLatentPrepStage(
        use_refiner=True, transformer=None, vae=None, model_path=""
    )


def _batch(session, block_idx, image_latent):
    return SimpleNamespace(
        session=session,
        block_idx=block_idx,
        image_latent=image_latent,
        seed=7,
        generator=None,
        extra={},
        latents=None,
    )


def _server_args():
    return SimpleNamespace(pipeline_config=SimpleNamespace(dit_precision="fp32"))


def test_latent_prep_plan_and_noise_discipline(_global_args):
    stage = _prep_stage()
    session = RealtimeSession()
    fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float32)

    inputs = session.get_or_create_state(SanaWMSessionInputsState)
    inputs.latent_t = 5  # fixed horizon: segments [0, 3, 5] for nfpb=2
    inputs.num_frame_per_block = 2
    inputs.sink_size = 1
    cache = session.get_or_create_state(SanaWMStreamCacheState)

    # Tick 0: front-loaded chunk 0 (cond + remainder) + full-horizon buffer.
    batch = stage.forward(_batch(session, 0, fl), _server_args())
    noise = session.get_or_create_state(SanaWMNoiseState)
    assert noise.segments == [0, 3, 5]
    assert noise.noise_buffer is not None and noise.noise_buffer.shape[2] == 5
    assert batch.extra["sana_wm_chunk_plan"] == [3]
    assert batch.latents.shape[2] == 3
    assert torch.equal(batch.latents[:, :, :1].cpu(), fl)  # cond frame in front
    assert torch.equal(  # noise sliced from the buffer, bitwise
        batch.latents[:, :, 1:].cpu(), noise.noise_buffer[:, :, 1:3].cpu()
    )

    # Tick 1: stage-1 advanced to frame 3 (simulated) -> next segment [3, 5).
    cache.chunk_indices = [0, 3]
    cache.chunk_idx = 1
    batch = stage.forward(_batch(session, 1, fl), _server_args())
    assert batch.extra["sana_wm_chunk_plan"] == [2]
    assert torch.equal(batch.latents.cpu(), noise.noise_buffer[:, :, 3:5].cpu())

    # Tick 2: horizon exhausted -> seamless continuation, uniform chunk from
    # the seeded fallback generator (no reset, no rollover).
    cache.chunk_indices = [0, 3, 5]
    cache.chunk_idx = 2
    batch = stage.forward(_batch(session, 2, fl), _server_args())
    assert batch.extra["sana_wm_chunk_plan"] == [2]
    assert batch.latents.shape[2] == 2
    assert torch.isfinite(batch.latents).all()


def test_latent_prep_open_ended_uniform_chunk0(_global_args):
    stage = _prep_stage()
    session = RealtimeSession()
    fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float32)

    inputs = session.get_or_create_state(SanaWMSessionInputsState)
    inputs.latent_t = None  # open-ended
    inputs.num_frame_per_block = 3
    inputs.sink_size = 1
    session.get_or_create_state(SanaWMStreamCacheState)

    batch = stage.forward(_batch(session, 0, fl), _server_args())
    noise = session.get_or_create_state(SanaWMNoiseState)
    assert noise.segments is None and noise.noise_buffer is None
    # Uniform grid chunk 0 = cond frame + nfpb new frames.
    assert batch.extra["sana_wm_chunk_plan"] == [4]
    assert batch.latents.shape[2] == 4
    assert torch.equal(batch.latents[:, :, :1].cpu(), fl)
