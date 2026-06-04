# SPDX-License-Identifier: Apache-2.0
"""S3 test — SanaWMRealtimeSession: incremental, state-carried streaming.

A tiny CPU model is stepped several times (one chunk per step). The session must
carry the per-block KV cache + the growing latent across steps and produce finite
chunks — the engine the interactive WASD/IJKL UI drives.
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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.realtime import (
    SanaWMRealtimeSession,
)
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
    m = SanaWMTransformer3DModel(SanaWMConfig(arch_config=arch)).double().eval()
    for b in m.blocks:
        b.cross_attn = _ZeroCross()
    return m


def test_realtime_session_multi_step_carries_state(_global_args):
    m = _tiny_model()
    session = SanaWMRealtimeSession(
        m,
        denoising_step_list=(1000, 700, 0),
        num_frame_per_block=3,
        cfg_scale=1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    torch.manual_seed(0)
    first_latent = torch.randn(1, MC, 1, 2, 2, dtype=torch.float64)  # VAE-encoded first frame
    prompt = torch.randn(1, 4, 32, dtype=torch.float64)
    session.reset(first_latent, prompt)

    # Step 1 (chunk 0): condition frame + 2 new frames.
    f1 = session.step(n_frames=2)
    assert f1.shape == (1, MC, 2, 2, 2)
    assert torch.isfinite(f1).all()
    assert session.latents.shape[2] == 3  # 1 cond + 2
    assert session.chunk_idx == 1
    assert session.kv_cache[0][0][0] is not None  # GDN state stored

    # Step 2 (chunk 1): 3 new frames, carrying the kv-cache.
    f2 = session.step(n_frames=3)
    assert f2.shape == (1, MC, 3, 2, 2)
    assert torch.isfinite(f2).all()
    assert session.latents.shape[2] == 6
    assert session.chunk_idx == 2
    assert len(session.kv_cache) == 2

    # Step 3: another 3 frames.
    f3 = session.step(n_frames=3)
    assert session.latents.shape[2] == 9
    assert torch.isfinite(f3).all()
    # The condition (first) frame is held fixed across the whole session.
    assert torch.allclose(session.latents[:, :, 0], first_latent[:, :, 0])


def test_realtime_session_reset_clears_state(_global_args):
    m = _tiny_model()
    session = SanaWMRealtimeSession(
        m, denoising_step_list=(1000, 0), num_frame_per_block=2,
        cfg_scale=1.0, device=torch.device("cpu"), dtype=torch.float64,
    )
    fl = torch.randn(1, MC, 1, 2, 2, dtype=torch.float64)
    pe = torch.randn(1, 4, 32, dtype=torch.float64)
    session.reset(fl, pe)
    session.step(n_frames=2)
    assert session.chunk_idx == 1
    session.reset(fl, pe)
    assert session.chunk_idx == 0 and len(session.kv_cache) == 0
    assert session.latents.shape[2] == 1
