# SPDX-License-Identifier: Apache-2.0
"""SANA-WM streaming paths must not sync the host per step, chunk or block."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.sana_wm import (
    SanaWMArchConfig,
    SanaWMConfig,
)
from sglang.multimodal_gen.runtime.models.dits import sana_wm
from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (
    _SLOT_TYPE_FLAG,
    compute_chunk_plucker,
    process_camera_conditions_ucpe,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    streaming_refiner,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming import (
    self_forcing_denoise_chunk,
)
from sglang.multimodal_gen.test.unit.host_sync_guard import SyncOnRead


class _ZeroCross(torch.nn.Module):
    def forward(self, x, y, mask=None):
        return torch.zeros_like(x)


def _tiny_model():
    arch = SanaWMArchConfig(
        in_channels=8,
        out_channels=8,
        num_layers=4,  # block 3 is a softmax (concat-window) block
        num_attention_heads=2,
        attention_head_dim=16,
        linear_head_dim=16,
        num_cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_dim=32,
        caption_channels=32,
        model_max_length=8,
        softmax_every_n=4,
        update_rule="torch_recurrent",
        cam_update_rule="torch_recurrent",
        chunk_size=None,
    )
    model = sana_wm.SanaWMTransformer3DModel(SanaWMConfig(arch_config=arch))
    model = model.double().eval()
    for block in model.blocks:
        block.cross_attn = _ZeroCross()
    return model


def _chunk_inputs():
    torch.manual_seed(0)
    return dict(
        hidden_states=torch.randn(1, 8, 2, 2, 2, dtype=torch.float64),
        encoder_hidden_states=torch.randn(1, 4, 32, dtype=torch.float64),
        timestep=torch.full((1, 1, 2), 500.0, dtype=torch.float64),
        camera_conditions=torch.randn(1, 4, 20, dtype=torch.float64),
        chunk_plucker=torch.randn(1, 48, 4, 2, 2, dtype=torch.float64),
        frame_index=torch.arange(2, 4),
        start_f=2,
        end_f=4,
    )


def test_camera_helpers_run_without_host_reads():
    # meta stands in for CUDA: reading any value back raises there
    plucker = compute_chunk_plucker(torch.randn(1, 9, 20, device="meta"), (2, 3, 4))
    assert plucker.shape == (1, 48, 2, 3, 4)
    raymats = process_camera_conditions_ucpe(
        torch.randn(1, 2, 20, device="meta"), HW=(2, 3, 4), patch_size=(1, 1, 1)
    )
    assert raymats.shape == (1, 2, 3, 4, 4, 4)
    # rays of a frame with an undefined camera still map to the identity
    camera = torch.randn(1, 2, 20, dtype=torch.float64)
    camera[0, 1, :16] = float("nan")
    raymats = process_camera_conditions_ucpe(camera, HW=(2, 3, 4), patch_size=(1, 1, 1))
    assert torch.equal(
        raymats[0, 1], torch.eye(4, dtype=torch.float64).expand(3, 4, 4, 4)
    )
    assert torch.isfinite(raymats[0, 0]).all()


@torch.no_grad()
def test_forward_long_windows_chunk_inputs_once():
    model = _tiny_model()
    inputs = _chunk_inputs()

    def run_steps(reuse):
        outputs, caches = [], []
        for step in range(3):
            if step == 0 or not reuse:
                model._chunk_inputs_cache = None
                model._ucpe_apply_fns_cache = None
                model._plucker_emb_cache = None
            with patch.object(torch, "tensor", wraps=torch.tensor) as make_tensor:
                out, cache = model.forward_long(**inputs, save_kv_cache=True)
            # the cache-type flags are host tensors: uploading them blocks per block
            assert all("device" not in call.kwargs for call in make_tensor.mock_calls)
            outputs.append(out)
            caches.append(cache)
        return outputs, caches

    expected, _ = run_steps(reuse=False)
    with patch.object(
        sana_wm,
        "process_camera_conditions_ucpe",
        wraps=sana_wm.process_camera_conditions_ucpe,
    ) as ucpe:
        actual, caches = run_steps(reuse=True)
    # the chunk's steps share one UCPE build (fresh slices missed the cache)
    assert ucpe.call_count == 1
    for out, reference in zip(actual, expected, strict=True):
        assert torch.equal(out, reference)
    assert {block[_SLOT_TYPE_FLAG].device.type for block in caches[0]} == {"cpu"}


def test_self_forcing_chunk_never_reads_device_timesteps():
    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    set_timesteps = scheduler.set_timesteps

    def set_device_timesteps(*args, **kwargs):
        set_timesteps(*args, **kwargs)
        scheduler.timesteps = scheduler.timesteps.as_subclass(SyncOnRead)

    scheduler.set_timesteps = set_device_timesteps
    contexts = []

    def forward_long(**kwargs):
        latents = kwargs["hidden_states"]
        return latents * 0.5, ["cache"]

    chunk = {"lat": torch.randn(1, 3, 2, 2, 2)}

    def run(forward_ctx):
        chunk["lat"] = torch.ones(1, 3, 2, 2, 2)
        return self_forcing_denoise_chunk(
            transformer=SimpleNamespace(forward_long=forward_long),
            scheduler=scheduler,
            sigmas=[1.0, 0.6, 0.3],
            get_chunk=lambda: chunk["lat"],
            set_chunk=lambda denoised: chunk.update(lat=denoised),
            cond_mask=torch.zeros(1, 3, 2, 2, 2),
            embeds=None,
            mask=None,
            camera_conditions=None,
            chunk_plucker=None,
            chunk_kv=None,
            start_f=0,
            end_f=2,
            frame_index=None,
            do_cfg=False,
            cfg_scale=1.0,
            target_dtype=torch.float32,
            device=torch.device("cpu"),
            forward_ctx=forward_ctx,
        )

    assert run(None) == ["cache"]
    realtime = chunk["lat"].as_subclass(torch.Tensor).clone()

    def forward_ctx(timestep):
        contexts.append(timestep)
        return patch.object(torch, "is_grad_enabled", torch.is_grad_enabled)

    run(forward_ctx)
    # the batch path reads the schedule once per chunk for its forward contexts
    assert contexts == [1000, 600, 300, 0]
    assert torch.equal(chunk["lat"].as_subclass(torch.Tensor), realtime)


def test_refiner_rope_builds_without_uploads_and_once_per_block():
    rope_calls = []

    def rope(pixel_coords, device):
        rope_calls.append(pixel_coords)
        return pixel_coords.sum(-1)

    rope.patch_size_t, rope.patch_size = 1, 2
    rope.scale_factors, rope.causal_offset = (8, 32, 32), 1
    transformer = SimpleNamespace(rope=rope)
    with patch.object(torch, "tensor", wraps=torch.tensor) as upload:
        coords = streaming_refiner.build_rotary_emb_for_absolute_positions(
            transformer=transformer,
            batch_size=2,
            frame_positions=range(3, 6),
            height=4,
            width=6,
            device=torch.device("cpu"),
            fps=24.0,
        )
    assert upload.call_count == 0
    # the per-call upload formulation it replaced
    grid = torch.stack(
        torch.meshgrid(
            torch.tensor([3.0, 4.0, 5.0]),
            torch.arange(0, 4, 2, dtype=torch.float32),
            torch.arange(0, 6, 2, dtype=torch.float32),
            indexing="ij",
        )
    )
    ends = grid + torch.tensor((1, 2, 2), dtype=torch.float32).view(3, 1, 1, 1)
    latent = torch.stack([grid, ends], -1).flatten(1, 3).unsqueeze(0).repeat(2, 1, 1, 1)
    pixel = latent * torch.tensor((8, 32, 32)).view(1, -1, 1, 1)
    pixel[:, 0] = (pixel[:, 0] + 1 - 8).clamp(min=0)
    pixel[:, 0] = pixel[:, 0] / 24.0
    assert torch.equal(rope_calls[0], pixel)
    assert torch.equal(coords, pixel.sum(-1))

    core = streaming_refiner._RefinerCore(transformer, torch.device("cpu"), None)
    block = dict(batch_size=2, height=4, width=6, fps=24.0)
    first = core._block_rotary_emb(frame_positions=range(3, 6), **block)
    # a block's denoise levels and capture pass share the table
    assert core._block_rotary_emb(frame_positions=range(3, 6), **block) is first
    core._block_rotary_emb(frame_positions=range(6, 9), **block)
    assert len(rope_calls) == 3


def test_refiner_block_never_reads_device_sigmas():
    transformer = SimpleNamespace(
        config=SimpleNamespace(patch_size=1, patch_size_t=1),
        transformer_blocks=[None],
    )
    refiner = SimpleNamespace(
        transformer=transformer,
        device=torch.device("cpu"),
        dtype=torch.float32,
        _predict_x0_active_block=lambda *, active, sigma_cur, **_: (
            active * (1.0 - sigma_cur)
        ),
        _capture_block_kv=lambda **_: [(torch.zeros(1, 1), torch.zeros(1, 1))],
    )
    sigmas = torch.tensor([0.9, 0.5, 0.0])
    outputs = []
    for schedule in (sigmas, sigmas.as_subclass(SyncOnRead)):
        runner = streaming_refiner.RefinerChunkRunner(
            refiner,
            prompt_embeds=None,
            prompt_attention_mask=None,
            fps=24.0,
            sigmas=schedule,
            source_sink_frames=0,
            block_size=2,
            kv_max_frames=4,
            seed=0,
            spatial_shape=(2, 2),
        )
        outputs.append(
            runner.refine_block(
                block_idx=0,
                clean_block=torch.ones(1, 3, 2, 2, 2),
                block_start=0,
                block_end=2,
            ).as_subclass(torch.Tensor)
        )
    assert torch.equal(outputs[1], outputs[0])
