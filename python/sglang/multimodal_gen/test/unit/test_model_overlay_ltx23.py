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
    _resolve_existing_file,
)
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.models.dits.ltx_2 import (
    LTX2VideoTransformer3DModel,
)
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
    LTX2RefinementStage,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    _resolve_bundled_overlay_dir,
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


def test_ltx23_builtin_overlay_prefers_bundled_overlay_dir():
    target = resolve_model_overlay_target("Lightricks/LTX-2.3")
    assert target is not None

    _, overlay_spec = target
    bundled_overlay_dir = _resolve_bundled_overlay_dir(overlay_spec)

    assert bundled_overlay_dir is not None
    assert bundled_overlay_dir.endswith("model_overlays/ltx_2_3")


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


def test_ltx23_stage2_keeps_generator_state_while_ltx2_resets():
    ltx23_server_args = SimpleNamespace(
        model_path="Lightricks/LTX-2.3",
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(ltx_variant="ltx_2_3")
            )
        ),
    )
    legacy_server_args = SimpleNamespace(
        model_path="Lightricks/LTX-2",
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(arch_config=SimpleNamespace(ltx_variant="ltx_2"))
        ),
    )

    assert (
        LTX2RefinementStage._should_reset_stage2_generators(ltx23_server_args) is False
    )
    assert (
        LTX2RefinementStage._should_reset_stage2_generators(legacy_server_args) is True
    )


def test_ltx23_stage2_generator_reset_falls_back_to_model_path():
    server_args = SimpleNamespace(
        model_path="Lightricks/LTX-2.3",
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(arch_config=SimpleNamespace())
        ),
    )

    assert LTX2RefinementStage._should_reset_stage2_generators(server_args) is False


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


def test_ltx23_materializer_prefers_current_official_artifact_names(tmp_path):
    current = tmp_path / "ltx-2.3-20b-dev.safetensors"
    legacy = tmp_path / "ltx-2.3-22b-dev.safetensors"
    current.touch()
    legacy.touch()

    resolved = _resolve_existing_file(
        str(tmp_path),
        (
            "ltx-2.3-20b-dev.safetensors",
            "ltx-2.3-22b-dev.safetensors",
        ),
    )

    assert resolved == str(current)


def test_ltx23_materializer_falls_back_to_legacy_artifact_names(tmp_path):
    legacy = tmp_path / "ltx-2.3-22b-distilled-lora-384.safetensors"
    legacy.touch()

    resolved = _resolve_existing_file(
        str(tmp_path),
        (
            "ltx-2.3-20b-distilled-lora-384.safetensors",
            "ltx-2.3-22b-distilled-lora-384.safetensors",
        ),
    )

    assert resolved == str(legacy)


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


def test_ltx23_two_stage_component_auto_resolution_prefers_current_assets(tmp_path):
    current_spatial = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
    legacy_spatial = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    current_lora = tmp_path / "ltx-2.3-20b-distilled-lora-384.safetensors"
    legacy_lora = tmp_path / "ltx-2.3-22b-distilled-lora-384.safetensors"
    current_spatial.touch()
    legacy_spatial.touch()
    current_lora.touch()
    legacy_lora.touch()

    resolved = _resolve_ltx2_two_stage_component_paths(str(tmp_path), {})

    assert resolved["spatial_upsampler"] == str(current_spatial)
    assert resolved["distilled_lora"] == str(current_lora)


def test_ltx23_two_stage_component_auto_resolution_falls_back_to_legacy_assets(
    tmp_path,
):
    legacy_spatial = tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    legacy_lora = tmp_path / "ltx-2.3-22b-distilled-lora-384.safetensors"
    legacy_spatial.touch()
    legacy_lora.touch()

    resolved = _resolve_ltx2_two_stage_component_paths(str(tmp_path), {})

    assert resolved["spatial_upsampler"] == str(legacy_spatial)
    assert resolved["distilled_lora"] == str(legacy_lora)


def test_ltx23_stage1_guider_params_are_ignored_during_stage2():
    stage = object.__new__(LTX2AVDenoisingStage)
    req = Req(
        sampling_params=SamplingParams(),
        prompt="prompt",
        extra={"ltx2_stage1_guider_params": {"video_cfg_scale": 3.0}},
    )

    assert (
        stage._get_ltx2_stage1_guider_params(req, SimpleNamespace(), "stage1")
        == req.extra["ltx2_stage1_guider_params"]
    )
    assert (
        stage._get_ltx2_stage1_guider_params(req, SimpleNamespace(), "stage2") is None
    )


def test_ltx23_refinement_stage_clears_ti2v_conditioning_and_restores_batch_state(
    monkeypatch,
):
    captured = {}

    def fake_forward(self, batch, server_args):
        captured["phase"] = batch.extra["ltx2_phase"]
        captured["image_latent"] = batch.image_latent
        captured["num_image_tokens"] = batch.ltx2_num_image_tokens
        captured["do_cfg"] = batch.do_classifier_free_guidance
        captured["num_inference_steps"] = batch.num_inference_steps
        captured["timesteps"] = batch.timesteps.clone()
        return batch

    monkeypatch.setattr(LTX2AVDenoisingStage, "forward", fake_forward)

    scheduler = SimpleNamespace(
        sigmas=torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32),
        timesteps=torch.tensor([1000.0, 500.0], dtype=torch.float32),
        _step_index=None,
        _begin_index=None,
    )
    stage = LTX2RefinementStage(
        transformer=None,
        scheduler=scheduler,
        distilled_sigmas=[0.9, 0.4, 0.0],
    )
    original_timesteps = torch.tensor([42.0, 21.0], dtype=torch.float32)
    original_generator = torch.Generator(device="cpu").manual_seed(7)
    req = Req(
        sampling_params=SamplingParams(seed=7),
        prompt="prompt",
        latents=torch.zeros((1, 2, 3), dtype=torch.float32),
        audio_latents=torch.zeros((1, 2, 3), dtype=torch.float32),
        image_latent=torch.ones((1, 1, 3), dtype=torch.float32),
        timesteps=original_timesteps.clone(),
        num_inference_steps=8,
        do_classifier_free_guidance=True,
        generator=[original_generator],
    )
    req.ltx2_num_image_tokens = 1

    result = stage.forward(req, SimpleNamespace())

    assert result is req
    assert captured["phase"] == "stage2"
    assert captured["image_latent"] is None
    assert captured["num_image_tokens"] == 0
    assert captured["do_cfg"] is False
    assert captured["num_inference_steps"] == 2
    assert torch.equal(
        captured["timesteps"], torch.tensor([900.0, 400.0], dtype=torch.float32)
    )
    assert req.generator[0] is original_generator
    assert torch.equal(req.timesteps, original_timesteps)
    assert req.num_inference_steps == 8
    assert req.do_classifier_free_guidance is True


def test_ltx23_av_cross_attn_gate_uses_sigma_scale():
    class IdentityPatchify(torch.nn.Module):
        def forward(self, x):
            return x, None

    class RecordingAdaLN(torch.nn.Module):
        def __init__(self, hidden_size, embedding_coefficient=1):
            super().__init__()
            self.hidden_size = hidden_size
            self.embedding_coefficient = embedding_coefficient
            self.last_timestep = None

        def forward(self, timestep, hidden_dtype=None):
            self.last_timestep = timestep.detach().clone()
            dtype = hidden_dtype or torch.float32
            batch = int(timestep.numel())
            out = torch.zeros(
                batch,
                self.embedding_coefficient * self.hidden_size,
                dtype=dtype,
                device=timestep.device,
            )
            embedded = torch.zeros(
                batch,
                self.hidden_size,
                dtype=dtype,
                device=timestep.device,
            )
            return out, embedded

    class IdentityNorm(torch.nn.Module):
        def forward(self, x):
            return x

    class IdentityProj(torch.nn.Module):
        def forward(self, x):
            return x, None

    class ZeroRope(torch.nn.Module):
        def forward(self, coords, device=None):
            device = device or coords.device
            shape = (coords.shape[0], 1, coords.shape[-1], 2)
            zeros = torch.zeros(shape, device=device, dtype=torch.float32)
            return zeros, zeros

    hidden_size = 8
    model = object.__new__(LTX2VideoTransformer3DModel)
    torch.nn.Module.__init__(model)
    model.patchify_proj = IdentityPatchify()
    model.audio_patchify_proj = IdentityPatchify()
    model.rope = ZeroRope()
    model.audio_rope = ZeroRope()
    model.cross_attn_rope = ZeroRope()
    model.cross_attn_audio_rope = ZeroRope()
    model.adaln_single = RecordingAdaLN(hidden_size, embedding_coefficient=6)
    model.audio_adaln_single = RecordingAdaLN(hidden_size, embedding_coefficient=6)
    model.prompt_adaln_single = RecordingAdaLN(hidden_size, embedding_coefficient=2)
    model.audio_prompt_adaln_single = RecordingAdaLN(
        hidden_size, embedding_coefficient=2
    )
    model.av_ca_video_scale_shift_adaln_single = RecordingAdaLN(
        hidden_size, embedding_coefficient=4
    )
    model.av_ca_a2v_gate_adaln_single = RecordingAdaLN(
        hidden_size, embedding_coefficient=1
    )
    model.av_ca_audio_scale_shift_adaln_single = RecordingAdaLN(
        hidden_size, embedding_coefficient=4
    )
    model.av_ca_v2a_gate_adaln_single = RecordingAdaLN(
        hidden_size, embedding_coefficient=1
    )
    model.caption_projection = None
    model.audio_caption_projection = None
    model.transformer_blocks = torch.nn.ModuleList([])
    model.scale_shift_table = torch.nn.Parameter(torch.zeros(2, hidden_size))
    model.audio_scale_shift_table = torch.nn.Parameter(torch.zeros(2, hidden_size))
    model.norm_out = IdentityNorm()
    model.audio_norm_out = IdentityNorm()
    model.proj_out = IdentityProj()
    model.audio_proj_out = IdentityProj()
    model.patch_size = (1, 1, 1)
    model.out_channels_raw = hidden_size
    model.audio_out_channels = hidden_size
    model.timestep_scale_multiplier = 1000
    model.av_ca_timestep_scale_multiplier = 1

    timestep = torch.tensor([647.0], dtype=torch.float32)
    model.forward(
        hidden_states=torch.zeros((1, 4, hidden_size), dtype=torch.float32),
        audio_hidden_states=torch.zeros((1, 4, hidden_size), dtype=torch.float32),
        encoder_hidden_states=torch.zeros((1, 2, hidden_size), dtype=torch.float32),
        audio_encoder_hidden_states=torch.zeros(
            (1, 2, hidden_size), dtype=torch.float32
        ),
        timestep=timestep,
        audio_timestep=timestep,
        num_frames=1,
        height=2,
        width=2,
        audio_num_frames=4,
        video_coords=torch.zeros((1, 3, 4, 2), dtype=torch.float32),
        audio_coords=torch.zeros((1, 1, 4, 2), dtype=torch.float32),
        return_latents=False,
    )

    expected_gate_timestep = torch.tensor([0.647], dtype=torch.float32)
    assert torch.equal(
        model.av_ca_video_scale_shift_adaln_single.last_timestep, timestep
    )
    assert torch.equal(
        model.av_ca_audio_scale_shift_adaln_single.last_timestep, timestep
    )
    assert torch.allclose(
        model.av_ca_a2v_gate_adaln_single.last_timestep, expected_gate_timestep
    )
    assert torch.allclose(
        model.av_ca_v2a_gate_adaln_single.last_timestep, expected_gate_timestep
    )
