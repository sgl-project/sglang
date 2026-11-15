# SPDX-License-Identifier: Apache-2.0
"""An audio-free H3 DiT (``audio_enabled: false``) drops only the audio projections."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.models.dits.minimax_h3 import (
    MiniMaxH3DiTArchConfig,
    MiniMaxH3DiTConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionMetadata,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.minimax_h3 import MiniMaxH3DiTModel
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

AUDIO_KEYS = {
    "audio_patch_proj.weight",
    "audio_patch_proj.bias",
    "final_layer.audio_out.weight",
    "final_layer.audio_out.bias",
}


def _runtime():
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _arch(audio_enabled, **overrides):
    return MiniMaxH3DiTArchConfig(
        audio_enabled=audio_enabled,
        num_layers=1,
        token_refiner_num_layers=1,
        hidden_size=12,
        num_attention_heads=1,
        attention_head_dim=12,
        ffn_hidden_size=24,
        latents_dim=2,
        audio_latents_dim=32,
        patch_size=(1, 1, 1),
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=12,
        time_embed_dim=6,
        adaln_out_features=216,
        final_adaln_out_features=24,
        rope_inv_freq_len=2,
        **overrides,
    )


def _model(audio_enabled, **overrides):
    return MiniMaxH3DiTModel(
        MiniMaxH3DiTConfig(arch_config=_arch(audio_enabled, **overrides)), {}
    ).eval()


@pytest.fixture(params=[False, True], ids=["plain", "pdd"])
def models(request, tmp_path):
    """The same weights with and without the audio projections, optionally
    with the same fused PDD video heads."""
    _runtime()
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        video = _model(True)
        image = _model(False)
    generator = torch.Generator().manual_seed(123)
    with torch.no_grad():
        for param in video.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.1)
        image.load_state_dict(
            {k: v for k, v in video.state_dict().items() if "audio" not in k},
            strict=True,
        )
    if request.param:
        heads = _write_heads(
            tmp_path / "video.safetensors", 2, audio=True, generator=generator
        )
        save_file(
            {k: v for k, v in heads.items() if k.startswith("video_out.")},
            str(tmp_path / "image.safetensors"),
        )
        video.final_layer.load_pdd_fused_heads(str(tmp_path / "video.safetensors"))
        image.final_layer.load_pdd_fused_heads(str(tmp_path / "image.safetensors"))
    return video, image


def _inputs(*, positive=False, edit=False):
    text_pos = torch.tensor([3, 4] if positive else [], dtype=torch.long)
    return {
        "x": torch.arange(10, dtype=torch.float32).reshape(1, 5, 2) / 10,
        "audio_x": torch.zeros(1, 5, 32),
        "prompt_embeds": (
            torch.arange(16, dtype=torch.bfloat16).reshape(2, 8) / 16
            if positive
            else torch.empty(0, 8, dtype=torch.bfloat16)
        ),
        "img_position_ids": torch.zeros(1, 5, 3),
        "unique_timesteps": torch.tensor([0.0, 0.999]),
        "inverse_indices": torch.tensor([1, 1, 1, 0, 0]),
        "token_tags": torch.tensor([0, 0, 0, 1, 1] if positive else [0, 0, 0, -1, -1]),
        "update_mask": torch.tensor([not edit, True, True]),
        "img_pos_info": {"position_ids": torch.arange(3)},
        "audio_pos_info": {"position_ids": torch.empty(0, dtype=torch.long)},
        "text_pos_info": {"position_ids": text_pos},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(3)},
        "packed_seq_params": {
            "cu_seqlens_q": torch.tensor(
                [0, 5 if positive else 3, 5], dtype=torch.int32
            ),
            "max_seqlen_q": 5 if positive else 3,
        },
        "refiner_packed_seq_params": {
            "cu_seqlens_q": torch.tensor([0, len(text_pos)], dtype=torch.int32),
            "max_seqlen_q": len(text_pos),
        },
    }


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_audio_flag_is_a_strict_bool(value):
    with pytest.raises(ValueError, match="audio_enabled must be a bool"):
        MiniMaxH3DiTArchConfig(audio_enabled=value)
    with pytest.raises(ValueError, match="audio_enabled must be a bool"):
        MiniMaxH3DiTConfig().update_model_arch({"audio_enabled": value})


def test_transformer_config_turns_audio_off():
    config = MiniMaxH3DiTConfig()
    assert config.arch_config.audio_enabled is True
    config.update_model_arch({"audio_enabled": False})
    assert config.arch_config.audio_enabled is False


def test_only_the_audio_projections_are_removed(models):
    video, image = models
    removed = set(video.state_dict()) - set(image.state_dict())
    assert removed and all(
        name.startswith(("audio_patch_proj.", "final_layer.audio_out."))
        for name in removed
    )
    assert removed == {k for k in video.state_dict() if "audio" in k}
    assert not set(image.state_dict()) - set(video.state_dict())
    assert image.audio_patch_proj is None and image.final_layer.audio_out is None
    image.post_load_weights()
    # A checkpoint with audio tensors does not load into an audio-free model.
    with pytest.raises(RuntimeError, match="Unexpected key"):
        image.load_state_dict(video.state_dict(), strict=True)


@pytest.mark.parametrize("edit", [False, True])
def test_image_prediction_matches_the_audio_model(models, edit):
    video, image = models
    inputs = _inputs(positive=True, edit=edit)
    with torch.inference_mode(), set_forward_context(0, AttentionMetadata(0)):
        expected, expected_audio = video(**inputs)
        with patch.object(image, "materialize_mps_non_layer_weights") as materialize:
            actual, audio = image(**inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()
    assert audio.shape == expected_audio.shape == (0, 32)
    assert audio.dtype == expected_audio.dtype == torch.float32
    assert all(
        "audio_patch_proj" not in call.args for call in materialize.call_args_list
    )


def test_empty_text_is_rejected(models):
    for model in models:
        with (
            torch.inference_mode(),
            set_forward_context(0, AttentionMetadata(0)),
            pytest.raises(ValueError, match="refiner cu_seqlens live text length"),
        ):
            model(**_inputs(positive=False))


def test_audio_rows_are_rejected(models):
    _, image = models
    inputs = _inputs()
    inputs["audio_pos_info"] = {"position_ids": torch.tensor([0])}
    with (
        torch.inference_mode(),
        set_forward_context(0, AttentionMetadata(0)),
        pytest.raises(ValueError, match="audio-free"),
    ):
        image(**inputs)


def _write_heads(path, steps, *, audio, generator):
    heads = {
        "video_out.weight": torch.randn(steps, 2, 12, generator=generator),
        "video_out.bias": torch.randn(steps, 2, generator=generator),
    }
    if audio:
        heads["audio_out.weight"] = torch.randn(steps, 32, 12, generator=generator)
        heads["audio_out.bias"] = torch.randn(steps, 32, generator=generator)
    save_file(heads, str(path))
    return heads


def test_audio_free_pdd_heads_project_step_i_through_head_i(tmp_path):
    _runtime()
    image = _model(False)
    path = tmp_path / "pdd_fused_heads.safetensors"
    heads = _write_heads(
        path, 2, audio=False, generator=torch.Generator().manual_seed(0)
    )
    image.final_layer.load_pdd_fused_heads(str(path))
    h = torch.randn(3, 12)
    for step in range(2):
        with set_forward_context(step, None):
            video, audio = image.final_layer._project(h)
        expected = torch.nn.functional.linear(
            h, heads["video_out.weight"][step], heads["video_out.bias"][step]
        )
        torch.testing.assert_close(video, expected, rtol=0, atol=0)
        assert audio.shape == (3, 0)
    with set_forward_context(2, None), pytest.raises(ValueError, match="2 fused heads"):
        image.final_layer._project(h)


def test_audio_free_model_rejects_audio_pdd_heads(tmp_path):
    _runtime()
    path = tmp_path / "pdd_fused_heads.safetensors"
    _write_heads(path, 2, audio=True, generator=torch.Generator().manual_seed(0))
    with pytest.raises(ValueError, match="must hold exactly"):
        _model(False).final_layer.load_pdd_fused_heads(str(path))


def test_video_pipeline_rejects_an_audio_free_dit():
    pipeline = SimpleNamespace(
        is_video_pipeline=True, pipeline_name="MiniMaxH3Pipeline"
    )
    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=MiniMaxH3DiTConfig(arch_config=_arch(False))
        )
    )
    with pytest.raises(ValueError, match="generates audio"):
        MiniMaxH3Pipeline.create_pipeline_stages(pipeline, server_args)
