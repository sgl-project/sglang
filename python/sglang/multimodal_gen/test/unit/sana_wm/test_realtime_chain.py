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
from PIL import Image

from sglang.multimodal_gen.runtime import server_args as _sa_mod
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.realtime_chain import (
    SanaWMCameraCondStage,
    SanaWMNoiseState,
    SanaWMRealtimeLatentPrepStage,
    SanaWMSessionInputsState,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.realtime_stage import (
    SanaWMRealtimeStage,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.runtime.realtime.states import (
    get_realtime_causal_dit_state,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

MC = 8


class _TestRealtimeStage(SanaWMRealtimeStage):
    def forward(self, batch, server_args):
        raise NotImplementedError


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


def _realtime_stage():
    return object.__new__(_TestRealtimeStage)


def _camera_stage():
    return object.__new__(SanaWMCameraCondStage)


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


def test_realtime_intrinsics_default_to_centered_heuristic():
    stage = _realtime_stage()
    state = SimpleNamespace(
        intrinsics_raw=None,
        intrinsics_image=Image.new("RGB", (832, 480)),
    )

    intrinsics = stage._prepare_intrinsics(
        SimpleNamespace(condition_inputs={}),
        state,
        num_frames=3,
        device=torch.device("cpu"),
    )

    assert intrinsics.shape == (3, 4)
    assert intrinsics[0].tolist() == pytest.approx([665.6, 665.6, 416.0, 240.0])
    assert intrinsics[2].tolist() == pytest.approx([665.6, 665.6, 416.0, 240.0])


def test_realtime_first_frame_uses_requested_size():
    stage = _realtime_stage()
    batch = SimpleNamespace(
        condition_image=Image.new("RGB", (640, 360)),
        image_path=None,
        height=480,
        width=832,
    )

    cropped, original, src_size, resized_size, crop_offset = stage._prepare_image(batch)

    assert cropped.size == (832, 480)
    assert original.size == (640, 360)
    assert src_size == (640, 360)
    assert resized_size == (853, 480)
    assert crop_offset == (10, 0)


def test_realtime_camera_conditioning_uses_requested_size():
    stage = _camera_stage()
    inputs = SanaWMSessionInputsState()
    inputs.src_size = (640, 360)
    inputs.resized_size = (853, 480)
    inputs.crop_offset = (10, 0)
    inputs.target_height = 480
    inputs.target_width = 832
    inputs.intrinsics_image = Image.new("RGB", (640, 360))
    inputs.open_ended = True
    batch = SimpleNamespace(
        condition_inputs={},
        extra={},
        height=480,
        width=832,
        num_frames=17,
    )

    camera, plucker = stage._build_camera_windows(
        batch,
        inputs,
        target_latent=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert camera.shape == (1, 3, 20)
    assert plucker.shape == (1, 48, 3, 15, 26)


def test_latent_prep_plan_and_noise_discipline(_global_args):
    stage = _prep_stage()
    session = RealtimeSession()
    fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float32)

    inputs = session.get_or_create_state(SanaWMSessionInputsState)
    inputs.latent_t = 5  # fixed horizon: segments [0, 3, 5] for nfpb=2
    inputs.num_frame_per_block = 2
    inputs.sink_size = 1
    cache = get_realtime_causal_dit_state(session)

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
    get_realtime_causal_dit_state(session)

    batch = stage.forward(_batch(session, 0, fl), _server_args())
    noise = session.get_or_create_state(SanaWMNoiseState)
    assert noise.segments is None and noise.noise_buffer is None
    # Uniform grid chunk 0 = cond frame + nfpb new frames.
    assert batch.extra["sana_wm_chunk_plan"] == [4]
    assert batch.latents.shape[2] == 4
    assert torch.equal(batch.latents[:, :, :1].cpu(), fl)
