# SPDX-License-Identifier: Apache-2.0
"""LTX-2.x audio-video ComfyUI integrated mode: checkpoint spec and step stage."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import LTX25Config
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.ltx_2 import (
    _build_dit_config,
    _dequantize,
    _regular_hadamard,
    is_comfyui_ltx_dit_key,
    load_comfyui_ltx_connectors,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.comfyui_step import (
    patchify_audio,
    patchify_video,
    run_ltxav_connectors,
    shard_video_frames_for_sp,
    unpatchify_audio,
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


def test_patchify_layouts_round_trip():
    video = torch.randn(2, 3, 4, 5, 6)
    tokens = patchify_video(video)
    # Token order is (frame, row, column), channels last.
    torch.testing.assert_close(tokens[1, 5 * 6 + 6 + 2], video[1, :, 1, 1, 2])
    audio = torch.randn(2, 8, 7, 16)
    tokens = patchify_audio(audio)
    torch.testing.assert_close(tokens[0, 3, 16:32], audio[0, 1, 3])
    torch.testing.assert_close(unpatchify_audio(tokens), audio)


def _write_checkpoint(path, extra=()):
    tensors = {
        f"{PREFIX}audio_adaln_single.linear.weight": torch.zeros(1),
        f"{PREFIX}keyframes_abs_pos_embedding": torch.zeros(1),
        f"{PREFIX}transformer_blocks.0.audio_ff.net.0.proj.bias": torch.zeros(1),
        f"{PREFIX}transformer_blocks.3.attn1.to_q.weight": torch.zeros(1),
        **{key: torch.zeros(1) for key in extra},
    }
    save_file(
        tensors, str(path), metadata={"config": json.dumps({"transformer": TINY})}
    )


def test_dit_config_comes_from_the_file(tmp_path):
    path = tmp_path / "ltx.safetensors"
    _write_checkpoint(path)
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


def test_dit_keys_exclude_connectors_vae_and_markers():
    assert is_comfyui_ltx_dit_key(f"{PREFIX}proj_out.weight")
    assert not is_comfyui_ltx_dit_key("vae.decoder.conv.weight")
    assert not is_comfyui_ltx_dit_key(f"{PREFIX}proj_out.comfy_quant")
    assert not is_comfyui_ltx_dit_key(
        f"{PREFIX}video_embeddings_connector.learnable_registers"
    )
    mapping = get_param_names_mapping(LTX25Config().arch_config.param_names_mapping)
    assert (
        mapping(f"{PREFIX}transformer_blocks.3.audio_ff.net.2.weight")[0]
        == "transformer_blocks.3.audio_ff.proj_out.weight"
    )


def test_convrot_int8_weights_dequantize_to_the_original_basis():
    """Comfy stores W @ H^T per 256-wide input group; loading must undo it."""
    torch.manual_seed(0)
    weight = torch.randn(8, 512)
    hadamard = _regular_hadamard(256, torch.device("cpu"))
    torch.testing.assert_close(hadamard @ hadamard, torch.eye(256), atol=1e-5, rtol=0)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_quantization_ignored_layers_match_ltx2_module_paths(
    single_process_model_parallel,
):
    """--quantization-ignored-layers must reach LTX-2 linears by module path."""
    from sglang.multimodal_gen.runtime.layers.linear import LinearBase
    from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
    from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
        LTX2VideoTransformer3DModel,
    )

    config = LTX25Config()
    for key, value in {**TINY, "num_layers": 1}.items():
        if key != "rope_type":
            setattr(config.arch_config, key, value)
    config.arch_config.__post_init__()
    quant_config = Fp8Config(ignored_layers=["transformer_blocks.0.attn1", "proj_out"])
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel(
            config=config, hf_config={}, quant_config=quant_config
        )
    methods = {
        name: type(module.quant_method).__name__
        for name, module in model.named_modules()
        if isinstance(module, LinearBase)
    }
    for name in ("transformer_blocks.0.attn1.to_q", "proj_out"):
        assert methods[name] == "UnquantizedLinearMethod", name
    for name in ("transformer_blocks.0.attn2.to_q", "patchify_proj"):
        assert methods[name] == "Fp8LinearMethod", name


def test_sp_ranks_take_equal_whole_frame_blocks_of_the_video():
    """Ranks must get contiguous frames: the DiT offsets RoPE time by rank."""
    frames, height, width = 4, 2, 3
    tokens = patchify_video(torch.randn(1, 5, frames, height, width))
    timestep = torch.arange(frames * height * width, dtype=torch.float32)[None]
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
