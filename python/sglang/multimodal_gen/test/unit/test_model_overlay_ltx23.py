import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

pytest.importorskip("triton.compiler")

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    is_ltx23_native_variant,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.model_overlays.ltx_2_3._overlay.materialize import (
    _build_transformer_config,
    _build_vae_config,
    _rename_connector_key,
    _repack_ltx23_image_encoder_weights,
    _repack_ltx23_video_decoder_weights,
)
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    _resolve_ltx2_two_stage_component_paths,
    build_official_ltx2_sigmas,
    prepare_ltx2_mu,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av import (
    LTX2AVDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    resolve_model_overlay_target,
)


def _make_req(**sampling_kwargs) -> Req:
    return Req(
        sampling_params=SamplingParams(**sampling_kwargs),
        prompt="prompt",
        prompt_embeds=[torch.zeros(1, 1, 1)],
    )


def test_ltx23_builtin_overlay_target_is_hf_repo():
    target = resolve_model_overlay_target("Lightricks/LTX-2.3")
    assert target is not None

    source_model_id, overlay_spec = target
    assert source_model_id == "Lightricks/LTX-2.3"
    assert str(overlay_spec["overlay_repo_id"]) == "MickJ/LTX-2.3-overlay"
    assert str(overlay_spec["overlay_revision"]) == "main"
    assert str(overlay_spec["bundled_overlay_subdir"]) == "ltx_2_3"


def test_ltx23_model_info_resolves_to_native_pipeline_and_sampling_params():
    model_info = get_model_info("Lightricks/LTX-2.3", backend="sglang")

    assert model_info is not None
    assert model_info.pipeline_cls.__name__ == "LTX2Pipeline"
    assert model_info.sampling_param_cls.__name__ == "LTX23SamplingParams"


def test_ltx23_sampling_defaults_use_cuda_generator():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2.3",
        backend="sglang",
    )

    assert sampling_params.generator_device == "cuda"
    assert sampling_params.guidance_scale == 3.0
    assert sampling_params.num_inference_steps == 30


def test_ltx2_sampling_defaults_keep_cpu_generator():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2",
        backend="sglang",
    )

    assert sampling_params.generator_device == "cpu"


def test_ltx23_build_request_extra_sets_stage1_guider_defaults():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2.3",
        backend="sglang",
    )

    assert sampling_params.build_request_extra()["ltx2_stage1_guider_params"] == {
        "video_cfg_scale": 3.0,
        "video_stg_scale": 1.0,
        "video_rescale_scale": 0.7,
        "video_modality_scale": 3.0,
        "video_skip_step": 0,
        "video_stg_blocks": [28],
        "audio_cfg_scale": 7.0,
        "audio_stg_scale": 1.0,
        "audio_rescale_scale": 0.7,
        "audio_modality_scale": 3.0,
        "audio_skip_step": 0,
        "audio_stg_blocks": [28],
    }


def test_sampling_params_apply_request_extra_populates_req_extra():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2.3",
        backend="sglang",
    )
    req = Req(sampling_params=sampling_params, prompt="prompt")

    sampling_params.apply_request_extra(req)

    assert req.extra["ltx2_stage1_guider_params"]["video_cfg_scale"] == 3.0
    assert req.extra["ltx2_stage1_guider_params"]["audio_cfg_scale"] == 7.0


def test_ltx23_uses_official_sigma_schedule():
    sigmas = build_official_ltx2_sigmas(30)

    assert len(sigmas) == 30
    assert sigmas[0] == pytest.approx(1.0)
    assert sigmas[1] == pytest.approx(0.99495703, abs=1e-6)
    assert sigmas[-1] == pytest.approx(0.1, abs=1e-6)


def test_ltx23_native_variant_uses_explicit_marker_only():
    assert is_ltx23_native_variant(SimpleNamespace(ltx_variant="ltx_2_3")) is True
    assert is_ltx23_native_variant(SimpleNamespace(ltx_variant="ltx_2")) is False


def test_prepare_ltx2_mu_respects_variant_marker():
    ltx23_server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(ltx_variant="ltx_2_3")
            )
        )
    )
    legacy_server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(ltx_variant="ltx_2")
            ),
            vae_temporal_compression=8,
            vae_scale_factor=32,
        )
    )

    assert prepare_ltx2_mu(
        _make_req(num_frames=121, height=512, width=768),
        ltx23_server_args,
    ) == ("mu", None)

    key, mu = prepare_ltx2_mu(
        _make_req(num_frames=121, height=512, width=768),
        legacy_server_args,
    )
    assert key == "mu"
    assert isinstance(mu, float)
    assert mu > 0.0


def test_ltx23_ti2v_clean_latent_uses_zero_background():
    latents = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
    image_latent = torch.full((1, 2, 4), 99.0)

    conditioned, denoise_mask, clean_latent = (
        LTX2AVDenoisingStage._prepare_ltx2_ti2v_clean_state(
            latents=latents,
            image_latent=image_latent,
            num_img_tokens=2,
            zero_clean_latent=True,
        )
    )

    assert torch.equal(conditioned[:, :2], image_latent)
    assert torch.equal(clean_latent[:, :2], image_latent)
    assert torch.equal(clean_latent[:, 2:], torch.zeros_like(clean_latent[:, 2:]))
    assert torch.equal(denoise_mask[:, :2], torch.zeros_like(denoise_mask[:, :2]))
    assert torch.equal(denoise_mask[:, 2:], torch.ones_like(denoise_mask[:, 2:]))


def test_ltx2_ti2v_clean_latent_keeps_legacy_background_when_requested():
    latents = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
    image_latent = torch.full((1, 2, 4), 99.0)

    conditioned, _, clean_latent = LTX2AVDenoisingStage._prepare_ltx2_ti2v_clean_state(
        latents=latents,
        image_latent=image_latent,
        num_img_tokens=2,
        zero_clean_latent=False,
    )

    assert torch.equal(conditioned[:, :2], image_latent)
    assert torch.equal(clean_latent[:, :2], image_latent)
    assert torch.equal(clean_latent[:, 2:], latents[:, 2:])


def test_ltx23_velocity_to_x0_supports_tokenwise_sigma():
    sample = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    velocity = torch.tensor([[[0.5, 0.5], [1.0, 1.0]]], dtype=torch.float32)
    sigma = torch.tensor([[0.0, 0.5]], dtype=torch.float32)

    denoised = LTX2AVDenoisingStage._ltx2_velocity_to_x0(sample, velocity, sigma)

    expected = torch.tensor([[[1.0, 2.0], [2.5, 3.5]]], dtype=torch.float32)
    assert torch.allclose(denoised, expected)


def test_ltx23_connector_repack_renames_qk_norm_keys():
    assert (
        _rename_connector_key(
            "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.q_norm.weight"
        )
        == "video_connector.transformer_blocks.0.attn1.norm_q.weight"
    )
    assert (
        _rename_connector_key(
            "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.1.attn1.k_norm.weight"
        )
        == "audio_connector.transformer_blocks.1.attn1.norm_k.weight"
    )


def test_ltx23_transformer_config_forces_sdpa_for_v2a_cross_attention():
    with tempfile.TemporaryDirectory() as tmpdir:
        donor_dir = os.path.join(tmpdir, "donor")
        os.makedirs(os.path.join(donor_dir, "transformer"), exist_ok=True)
        with open(os.path.join(donor_dir, "transformer", "config.json"), "w") as f:
            json.dump({"_class_name": "OldClass", "num_layers": 1}, f)

        config = _build_transformer_config(donor_dir)

    assert config["_class_name"] == "LTX2VideoTransformer3DModel"
    assert config["force_sdpa_v2a_cross_attention"] is True


def test_ltx23_vae_config_adds_required_markers():
    with tempfile.TemporaryDirectory() as tmpdir:
        auxiliary_dir = os.path.join(tmpdir, "aux")
        config_donor_dir = os.path.join(tmpdir, "donor")
        os.makedirs(os.path.join(auxiliary_dir, "vae"), exist_ok=True)
        os.makedirs(os.path.join(config_donor_dir, "vae"), exist_ok=True)

        with open(os.path.join(auxiliary_dir, "vae", "config.json"), "w") as f:
            json.dump(
                {
                    "_class_name": "AutoencoderKLLTX2Video",
                    "scaling_factor": 1.0,
                    "patch_size": 4,
                    "decoder_causal": False,
                    "timestep_conditioning": False,
                    "encoder_spatial_padding_mode": "zeros",
                    "decoder_spatial_padding_mode": "reflect",
                },
                f,
            )
        with open(os.path.join(config_donor_dir, "vae", "config.json"), "w") as f:
            json.dump(
                {
                    "vae": {
                        "decoder_blocks": [["res_x", {"num_layers": 2}]],
                        "decoder_base_channels": 128,
                        "patch_size": 4,
                        "spatial_padding_mode": "zeros",
                    }
                },
                f,
            )

        config = _build_vae_config(auxiliary_dir, config_donor_dir)

    assert config["ltx_variant"] == "ltx_2_3"
    assert config["condition_encoder_subdir"] == "ltx23_image_encoder"
    assert config["video_decoder_variant"] == "ltx_2_3"
    assert config["video_decoder_config"]["decoder_base_channels"] == 128


def test_ltx23_repack_image_encoder_keeps_only_encoder_tensors():
    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = os.path.join(tmpdir, "source.safetensors")
        output_path = os.path.join(tmpdir, "output.safetensors")
        save_file(
            {
                "encoder.conv_in.conv.weight": torch.ones(1),
                "decoder.conv_in.conv.weight": torch.full((1,), 2.0),
                "per_channel_statistics.mean-of-means": torch.full((2,), 3.0),
            },
            source_path,
        )

        _repack_ltx23_image_encoder_weights(source_path, output_path)

        with safe_open(output_path, framework="pt") as f:
            assert sorted(f.keys()) == [
                "conv_in.conv.weight",
                "per_channel_statistics.mean-of-means",
            ]


def test_ltx23_repack_video_decoder_keeps_decoder_and_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        auxiliary_path = os.path.join(tmpdir, "aux.safetensors")
        donor_path = os.path.join(tmpdir, "donor.safetensors")
        output_path = os.path.join(tmpdir, "output.safetensors")
        save_file(
            {
                "encoder.conv_in.conv.weight": torch.full((1,), 5.0),
            },
            auxiliary_path,
        )
        save_file(
            {
                "decoder.conv_in.conv.weight": torch.ones(1),
                "per_channel_statistics.mean-of-means": torch.full((2,), 3.0),
                "per_channel_statistics.std-of-means": torch.full((2,), 4.0),
            },
            donor_path,
        )

        _repack_ltx23_video_decoder_weights(auxiliary_path, donor_path, output_path)

        with safe_open(output_path, framework="pt") as f:
            assert sorted(f.keys()) == [
                "decoder.conv_in.conv.weight",
                "decoder.per_channel_statistics.mean_of_means",
                "decoder.per_channel_statistics.std_of_means",
                "encoder.conv_in.conv.weight",
                "latents_mean",
                "latents_std",
            ]


def test_ltx23_decode_skips_external_denorm():
    ltx23_server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(video_decoder_variant="ltx_2_3")
            )
        )
    )
    legacy_server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(video_decoder_variant="ltx_2")
            )
        )
    )

    assert (
        LTX2AVDecodingStage._ltx2_should_externally_denorm_video_latents(
            ltx23_server_args
        )
        is False
    )
    assert (
        LTX2AVDecodingStage._ltx2_should_externally_denorm_video_latents(
            legacy_server_args
        )
        is True
    )


def test_ltx2_two_stage_component_auto_resolution_preserves_legacy_candidates(tmp_path):
    legacy_spatial = tmp_path / "ltx-2-spatial-upscaler-x2-1.0.safetensors"
    legacy_lora = tmp_path / "ltx-2-19b-distilled-lora-384.safetensors"
    legacy_spatial.touch()
    legacy_lora.touch()

    resolved = _resolve_ltx2_two_stage_component_paths(str(tmp_path), {})

    assert resolved["spatial_upsampler"] == str(legacy_spatial)
    assert resolved["distilled_lora"] == str(legacy_lora)


def test_ltx23_two_stage_component_auto_resolution_prefers_23_assets(tmp_path):
    spatial = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    lora = tmp_path / "ltx-2.3-22b-distilled-lora-384.safetensors"
    spatial.touch()
    lora.touch()

    resolved = _resolve_ltx2_two_stage_component_paths(str(tmp_path), {})

    assert resolved["spatial_upsampler"] == str(spatial)
    assert resolved["distilled_lora"] == str(lora)
