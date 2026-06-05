# SPDX-License-Identifier: Apache-2.0
"""Realtime (per-chunk) path of SanaWMStreamingDenoisingStage.

Successor of the retired SanaWMChunkGenerator tests: a tiny CPU-safe model is
ticked through stage.forward with a session attached (one chunk per call,
chunk noise supplied directly — the latent-prep stage is exercised separately).
The stage must carry the per-session 10-slot KV cache + the growing latent
across ticks, evict stale chunk entries, stay deterministic for identical
inputs, and reset on block_idx == 0.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.sana_wm import (
    SanaWMArchConfig,
    SanaWMConfig,
)
from sglang.multimodal_gen.runtime import server_args as _sa_mod
from sglang.multimodal_gen.runtime.models.dits.sana_wm import SanaWMTransformer3DModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.streaming import (
    SanaWMStreamCacheState,
    SanaWMStreamingDenoisingStage,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

MC = 8


class _ZeroCross(torch.nn.Module):
    def forward(self, x, y, mask=None):
        return torch.zeros_like(x)


@pytest.fixture
def _global_args():
    prev = _sa_mod._global_server_args
    set_global_server_args(
        SimpleNamespace(
            comfyui_mode=False,
            enable_cfg_parallel=False,
            enable_torch_compile=False,
            attention_backend=None,
            # DenoisingStage.__init__ reads this for its CFG-parallel plumbing.
            pipeline_config=SimpleNamespace(
                dit_config=SimpleNamespace(hidden_size=32, num_attention_heads=2)
            ),
        )
    )
    try:
        yield
    finally:
        set_global_server_args(prev)


def _tiny_model():
    arch = SanaWMArchConfig(
        in_channels=MC, out_channels=MC, num_layers=2,  # GDN-only -> CPU-safe main path
        num_attention_heads=2, attention_head_dim=16, linear_head_dim=16,
        num_cross_attention_heads=2, cross_attention_head_dim=16, cross_attention_dim=32,
        caption_channels=32, model_max_length=8, softmax_every_n=4,
        update_rule="torch_recurrent", cam_update_rule="torch_recurrent", chunk_size=None,
    )
    m = SanaWMTransformer3DModel(SanaWMConfig(arch_config=arch)).float().eval()
    for b in m.blocks:
        b.cross_attn = _ZeroCross()
    return m


def _stage_and_args(nfpb: int = 2):
    stage = SanaWMStreamingDenoisingStage(transformer=_tiny_model())
    prompt = torch.zeros(1, 4, 32, dtype=torch.float32)
    pcfg = SimpleNamespace(
        dit_precision="fp32",
        num_frame_per_block=nfpb,
        num_cached_blocks=2,
        sink_token=True,
        denoising_step_list=(1000, 700, 0),
        streaming_cfg_scale=1.0,
        get_pos_prompt_embeds=lambda batch: [prompt],
        get_neg_prompt_embeds=lambda batch: [],
    )
    server_args = SimpleNamespace(pipeline_config=pcfg, enable_cfg_parallel=False)
    return stage, server_args


def _tick(session, block_idx: int, chunk_lat: torch.Tensor, plan: list[int]):
    return SimpleNamespace(
        session=session,
        block_idx=block_idx,
        latents=chunk_lat,
        extra={"sana_wm_chunk_plan": plan},
        prompt_attention_mask=None,
        negative_attention_mask=None,
        do_classifier_free_guidance=False,
    )


def _noise(*shape, seed: int):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32, generator=g)


def test_realtime_path_multi_tick_carries_state(_global_args):
    stage, server_args = _stage_and_args(nfpb=2)
    session = RealtimeSession()
    fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float32)

    # Tick 0 (chunk 0): conditioning frame + 2 new frames.
    chunk0 = torch.cat([fl, _noise(1, MC, 2, 2, 2, seed=1)], dim=2)
    out = stage.forward(_tick(session, 0, chunk0, [3]), server_args)
    state = session.get_or_create_state(SanaWMStreamCacheState)
    assert out.latents.shape[2] == 3
    assert state.chunk_idx == 1 and state.chunk_indices == [0, 3]
    assert state.stream_kv_cache[0][0][0] is not None  # GDN state stored
    # The condition frame is held fixed.
    assert torch.allclose(state.latents[:, :, 0].cpu(), fl[:, :, 0])

    # Tick 1: 2 more frames, KV carried.
    out = stage.forward(_tick(session, 1, _noise(1, MC, 2, 2, 2, seed=2), [2]), server_args)
    assert out.latents.shape[2] == 5
    assert state.chunk_idx == 2 and state.chunk_indices == [0, 3, 5]
    assert torch.isfinite(state.latents).all()

    # Boundary-style tick: a TWO-chunk plan in one call.
    out = stage.forward(_tick(session, 2, _noise(1, MC, 4, 2, 2, seed=3), [2, 2]), server_args)
    assert out.latents.shape[2] == 9
    assert state.chunk_idx == 4 and state.chunk_indices == [0, 3, 5, 7, 9]


def test_realtime_path_evicts_stale_kv(_global_args):
    stage, server_args = _stage_and_args(nfpb=2)
    session = RealtimeSession()
    fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float32)
    stage.forward(_tick(session, 0, torch.cat([fl, _noise(1, MC, 2, 2, 2, seed=1)], 2), [3]), server_args)
    for i in range(1, 5):
        stage.forward(_tick(session, i, _noise(1, MC, 2, 2, 2, seed=10 + i), [2]), server_args)
    state = session.get_or_create_state(SanaWMStreamCacheState)
    assert state.chunk_indices == [0, 3, 5, 7, 9, 11]

    def _has_any(entry):
        return any(slot is not None for block in entry for slot in block)

    kept = [i for i, e in enumerate(state.stream_kv_cache) if _has_any(e)]
    # Sink chunk + the last num_cached_blocks chunks (accumulate's read window).
    assert kept == [0, 3, 4]


def test_realtime_path_is_deterministic(_global_args):
    def _run():
        stage, server_args = _stage_and_args(nfpb=2)
        torch.manual_seed(0)  # tiny-model init inside _stage_and_args uses global RNG
        session = RealtimeSession()
        fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float32)
        stage.forward(_tick(session, 0, torch.cat([fl, _noise(1, MC, 2, 2, 2, seed=5)], 2), [3]), server_args)
        stage.forward(_tick(session, 1, _noise(1, MC, 2, 2, 2, seed=6), [2]), server_args)
        return session.get_or_create_state(SanaWMStreamCacheState).latents.cpu()

    torch.manual_seed(1234)
    a = _run()
    torch.manual_seed(1234)
    b = _run()
    assert torch.equal(a, b)


def test_realtime_path_resets_on_block_zero(_global_args):
    stage, server_args = _stage_and_args(nfpb=2)
    session = RealtimeSession()
    fl = torch.ones(1, MC, 1, 2, 2, dtype=torch.float32)
    chunk0 = torch.cat([fl, _noise(1, MC, 2, 2, 2, seed=1)], 2)
    stage.forward(_tick(session, 0, chunk0, [3]), server_args)
    stage.forward(_tick(session, 1, _noise(1, MC, 2, 2, 2, seed=2), [2]), server_args)
    state = session.get_or_create_state(SanaWMStreamCacheState)
    assert state.chunk_idx == 2

    # block_idx == 0 restarts the session in place.
    stage.forward(_tick(session, 0, chunk0, [3]), server_args)
    assert state.chunk_idx == 1 and state.chunk_indices == [0, 3]
    assert state.latents.shape[2] == 3
