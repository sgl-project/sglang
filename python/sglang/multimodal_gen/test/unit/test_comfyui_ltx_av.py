# SPDX-License-Identifier: Apache-2.0
"""LTX-2.x audio-video ComfyUI integrated mode: checkpoint spec and step stage."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.ltx_2 import (
    _build_dit_config,
    _dequantize,
    _regular_hadamard,
    is_comfyui_ltx_dit_key,
    load_comfyui_ltx_connectors,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.comfyui_step import (
    run_ltxav_connectors,
    shard_video_frames_for_sp,
)

PREFIX = "model.diffusion_model."
TINY = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "cross_attention_dim": 32,
    "audio_num_attention_heads": 2,
    "audio_attention_head_dim": 8,
    "audio_cross_attention_dim": 16,
    "caption_channels": 32,
    "in_channels": 16,
    "out_channels": 16,
    "rope_type": "split",
}


def test_dit_config_comes_from_the_file(tmp_path):
    path = tmp_path / "ltx.safetensors"
    keys = (
        "audio_adaln_single.linear.weight",
        "keyframes_abs_pos_embedding",
        "transformer_blocks.0.audio_ff.net.0.proj.bias",
        "transformer_blocks.3.attn1.to_q.weight",
    )
    save_file(
        {f"{PREFIX}{key}": torch.zeros(1) for key in keys},
        str(path),
        metadata={"config": json.dumps({"transformer": TINY})},
    )
    server_args = SimpleNamespace(
        model_path=str(path), pipeline_config=SimpleNamespace(dit_config=None)
    )
    arch = _build_dit_config(server_args).arch_config
    assert (arch.num_layers, arch.hidden_size, arch.audio_hidden_size) == (4, 32, 16)
    # LTX-2.5 has no video FF bias; the metadata does not always say so.
    assert not arch.ff_bias and arch.audio_ff_bias
    assert arch.use_keyframes_abs_pos_embedding
    assert server_args.pipeline_config.dit_config.arch_config is arch


def test_dit_config_rejects_video_only_files(tmp_path):
    path = tmp_path / "ltxv.safetensors"
    save_file({f"{PREFIX}proj_out.weight": torch.zeros(1)}, str(path))
    server_args = SimpleNamespace(model_path=str(path), pipeline_config=None)
    with pytest.raises(ValueError, match="ltxav"):
        _build_dit_config(server_args)


def test_dit_keys_exclude_connectors_and_vae():
    assert is_comfyui_ltx_dit_key(f"{PREFIX}proj_out.weight")
    assert not is_comfyui_ltx_dit_key("vae.decoder.conv.weight")
    assert not is_comfyui_ltx_dit_key(
        f"{PREFIX}video_embeddings_connector.learnable_registers"
    )


def test_convrot_int8_weights_dequantize_to_the_original_basis():
    """Comfy stores W @ H^T per 256-wide input group; loading must undo it."""
    torch.manual_seed(0)
    weight = torch.randn(8, 512)
    hadamard = _regular_hadamard(256, torch.device("cpu"))
    rotated = (weight.view(8, 2, 256) @ hadamard.T).view(8, 512)
    scale = rotated.abs().amax(dim=1, keepdim=True) / 127
    restored = _dequantize((rotated / scale).round().to(torch.int8), scale, 256)
    assert ((restored - weight).norm() / weight.norm()).item() < 0.02


def test_sglang_connectors_reproduce_comfyui_connectors(tmp_path):
    """ComfyUI connector weights loaded into SGLang's connector give its output."""
    pytest.importorskip("comfy.ldm.lightricks.embeddings_connector")
    import comfy.ops
    from comfy.ldm.lightricks.embeddings_connector import Embeddings1DConnector

    torch.manual_seed(0)
    tensors, comfy_modules = {}, []
    for name, heads, head_dim in (
        ("video_embeddings_connector", 2, 16),
        ("audio_embeddings_connector", 2, 8),
    ):
        module = Embeddings1DConnector(
            attention_head_dim=head_dim,
            num_attention_heads=heads,
            num_layers=2,
            split_rope=True,
            double_precision_rope=True,
            apply_gated_attention=True,
            operations=comfy.ops.disable_weight_init,
        ).eval()
        with torch.no_grad():
            for param in module.parameters():
                param.copy_(torch.randn(param.shape) * 0.2)
        comfy_modules.append(module)
        tensors.update(
            {
                f"{PREFIX}{name}.{k}": v.contiguous()
                for k, v in module.state_dict().items()
            }
        )
    path = tmp_path / "ltx.safetensors"
    save_file(
        tensors, str(path), metadata={"config": json.dumps({"transformer": TINY})}
    )
    video, audio = load_comfyui_ltx_connectors(
        str(path), torch.device("cpu"), torch.float32
    )
    raw = torch.randn(1, 11, 48)
    with torch.no_grad():
        expected = torch.cat(
            (comfy_modules[0](raw[..., :32])[0], comfy_modules[1](raw[..., 32:])[0]), -1
        )
        actual = run_ltxav_connectors(video, audio, raw, video_dim=32)
    assert actual.shape == expected.shape == (1, 1024, 48)
    # Only the Q/K RMSNorm eps differs (ComfyUI 1e-5, SGLang 1e-6).
    err = (actual - expected).norm() / expected.norm()
    assert err.item() < 1e-3, err.item()


def test_sp_ranks_take_equal_whole_frame_blocks_of_the_video():
    """Ranks must get contiguous frames: the DiT offsets RoPE time by rank."""
    frames = 4
    tokens = torch.randn(1, frames * 6, 5)
    timestep = torch.arange(frames * 6, dtype=torch.float32)[None]
    shards = [
        shard_video_frames_for_sp(tokens, timestep, frames, rank, 2) for rank in (0, 1)
    ]
    torch.testing.assert_close(torch.cat([s[0] for s in shards], 1), tokens)
    torch.testing.assert_close(torch.cat([s[1] for s in shards], 1), timestep)
    assert [s[2] for s in shards] == [2, 2]
    # Per-sample timesteps ([B] or [B, 1]) are not per token and stay whole.
    for per_sample in (torch.full((1,), 0.5), torch.full((2, 1), 0.5)):
        sharded = shard_video_frames_for_sp(tokens, per_sample, frames, 1, 2)[1]
        assert sharded is per_sample
    with pytest.raises(ValueError, match="3 latent frames"):
        shard_video_frames_for_sp(tokens[:, :18], timestep[:, :18], 3, 0, 2)


@pytest.mark.parametrize(
    "options",
    [
        {"stg_self_attn_blocks": {3}},
        {"a2v_cross_attn": False, "v2a_cross_attn": False},
    ],
)
def test_guidance_passes_the_worker_cannot_apply_raise(options):
    """STG / modality guidance would silently equal the cond prediction."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.ltx_av import (
        LTXAVAdapter,
    )

    x = [torch.zeros(1, 128, 2, 4, 4), torch.zeros(1, 8, 3, 16)]
    with pytest.raises(NotImplementedError, match="guidance|STG"):
        LTXAVAdapter().pack(
            x, torch.ones(1), torch.zeros(1, 4, 32), transformer_options=options
        )
    LTXAVAdapter().pack(x, torch.ones(1), torch.zeros(1, 4, 32), transformer_options={})
