import json
import os
import tempfile
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
    _gemma_postprocess_func,
    pack_text_embeds_v2,
)
from sglang.multimodal_gen.model_overlays.ltx_2_3._overlay.materialize import (
    _build_vae_config,
    _rename_connector_key,
    _repack_ltx23_image_encoder_weights,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising_av import (
    LTX2AVDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation_av import (
    LTX2AVLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
    LTX2SigmaPreparationStage,
    build_official_ltx2_sigmas,
    prepare_ltx2_mu,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    maybe_load_overlay_model_index,
    resolve_model_overlay_target,
)


def test_ltx23_builtin_overlay_target_is_hf_repo():
    target = resolve_model_overlay_target("Lightricks/LTX-2.3")
    assert target is not None

    source_model_id, overlay_spec = target
    assert source_model_id == "Lightricks/LTX-2.3"
    assert str(overlay_spec["overlay_revision"]) == "main"
    assert str(overlay_spec["overlay_repo_id"]) == "MickJ/LTX-2.3-overlay"
    assert str(overlay_spec["bundled_overlay_subdir"]) == "ltx_2_3"


def test_ltx23_overlay_model_index_is_native():
    local_overlay_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "model_overlays",
            "ltx_2_3",
        )
    )
    model_index = maybe_load_overlay_model_index(
        "Lightricks/LTX-2.3",
        snapshot_download_fn=lambda **kwargs: local_overlay_dir,
        hf_hub_download_fn=lambda **kwargs: None,
    )
    assert model_index is not None
    assert model_index["_class_name"] == "LTX2Pipeline"


def test_ltx23_model_info_resolves_to_native_pipeline():
    model_info = get_model_info("Lightricks/LTX-2.3", backend="sglang")
    assert model_info is not None
    assert model_info.pipeline_cls.__name__ == "LTX2Pipeline"


def test_ltx23_model_info_uses_ltx23_sampling_params():
    model_info = get_model_info("Lightricks/LTX-2.3", backend="sglang")
    assert model_info is not None
    assert model_info.sampling_param_cls.__name__ == "LTX23SamplingParams"


def test_ltx23_defaults_to_cuda_generator():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2.3",
        backend="sglang",
    )
    pipeline_config = get_model_info(
        "Lightricks/LTX-2.3", backend="sglang"
    ).pipeline_config_cls()

    assert sampling_params.generator_device == "cuda"
    assert pipeline_config.generator_device == "cuda"


def test_ltx23_uses_official_sigma_schedule():
    sigmas = build_official_ltx2_sigmas(30)

    assert len(sigmas) == 30
    assert sigmas[0] == 1.0
    assert abs(sigmas[1] - 0.99495703) < 1e-6
    assert abs(sigmas[-1] - 0.1) < 1e-6


def test_ltx23_sigma_preparation_uses_official_schedule_only_for_ltx23_marker():
    req = Req(
        sampling_params=SamplingParams(num_inference_steps=30),
        prompt="prompt",
        prompt_embeds=[torch.zeros(1, 1, 1)],
    )
    stage = LTX2SigmaPreparationStage()

    ltx23_server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(use_official_image_encoder=True)
            )
        )
    )
    legacy_server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(use_official_image_encoder=False)
            )
        )
    )

    ltx23_batch = stage.forward(req, ltx23_server_args)
    legacy_batch = stage.forward(
        Req(
            sampling_params=SamplingParams(num_inference_steps=30),
            prompt="prompt",
            prompt_embeds=[torch.zeros(1, 1, 1)],
        ),
        legacy_server_args,
    )

    assert abs(ltx23_batch.sigmas[1] - 0.99495703) < 1e-6
    assert abs(legacy_batch.sigmas[1] - (29.0 / 30.0)) < 1e-6


def test_ltx23_skips_mu_when_using_official_sigma_schedule():
    req = Req(
        sampling_params=SamplingParams(num_frames=121, height=512, width=768),
        prompt="prompt",
        prompt_embeds=[torch.zeros(1, 1, 1)],
    )
    ltx23_server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            vae_config=SimpleNamespace(
                arch_config=SimpleNamespace(use_official_image_encoder=True)
            )
        )
    )

    assert prepare_ltx2_mu(req, ltx23_server_args) == ("mu", None)


def test_ltx23_prepare_request_sets_stage1_guider_defaults():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2.3",
        backend="sglang",
    )
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
    )

    req = prepare_request(server_args, sampling_params)

    assert req.num_inference_steps == 30
    assert req.guidance_scale == 3.0
    assert req.extra["ltx2_stage1_guider_params"] == {
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


def test_ltx23_stage1_guider_params_apply_to_one_stage_path():
    sampling_params = SamplingParams.from_pretrained(
        "Lightricks/LTX-2.3",
        backend="sglang",
    )
    server_args = SimpleNamespace(
        attention_backend_config=SimpleNamespace(VSA_sparsity=0.0),
    )

    req = prepare_request(server_args, sampling_params)
    stage = SimpleNamespace(pipeline=None)

    assert (
        LTX2AVDenoisingStage._get_ltx2_stage1_guider_params(
            stage, req, server_args, "stage1"
        )
        == req.extra["ltx2_stage1_guider_params"]
    )
    assert (
        LTX2AVDenoisingStage._get_ltx2_stage1_guider_params(
            stage, req, server_args, "stage2"
        )
        is None
    )


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


def test_ltx23_post_denoise_injection_overrides_latents(tmp_path):
    inject_path = tmp_path / "official_final.pt"
    torch.save(
        {
            "video_latent_after": torch.full((1, 2, 3), 7.0, dtype=torch.float32),
            "audio_latent_after": torch.full((1, 4, 5), 9.0, dtype=torch.float32),
        },
        inject_path,
    )
    os.environ["SGLANG_DIFFUSION_LTX2_POST_DENOISE_INJECT_PATH"] = str(inject_path)

    try:
        latents, audio_latents = LTX2AVDenoisingStage._ltx2_maybe_load_injected_latents(
            torch.zeros((1, 2, 3), dtype=torch.float32),
            torch.zeros((1, 4, 5), dtype=torch.float32),
        )
    finally:
        os.environ.pop("SGLANG_DIFFUSION_LTX2_POST_DENOISE_INJECT_PATH", None)

    assert torch.equal(latents, torch.full((1, 2, 3), 7.0, dtype=torch.float32))
    assert torch.equal(
        audio_latents, torch.full((1, 4, 5), 9.0, dtype=torch.float32)
    )


def test_pack_text_embeds_v2_masks_padding():
    hidden_states = torch.tensor(
        [
            [
                [[3.0, 4.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        ]
    )
    attention_mask = torch.tensor([[1, 0]])

    packed = pack_text_embeds_v2(hidden_states, attention_mask)

    expected_first = torch.tensor(
        [1.0, 1.0, 0.0, 0.0], dtype=packed.dtype, device=packed.device
    )
    assert torch.allclose(packed[0, 0], expected_first)
    assert torch.equal(packed[0, 1], torch.zeros_like(packed[0, 1]))


def test_ltx23_gemma_postprocess_uses_v2_norm():
    hidden_state = torch.tensor(
        [
            [
                [3.0, 0.0],
                [4.0, 0.0],
            ]
        ]
    )
    outputs = SimpleNamespace(hidden_states=(hidden_state,))
    text_inputs = {"attention_mask": torch.tensor([[1]])}
    pipeline_config = SimpleNamespace(
        dit_config=SimpleNamespace(
            arch_config=SimpleNamespace(caption_proj_before_connector=True)
        )
    )

    packed = _gemma_postprocess_func(outputs, text_inputs, pipeline_config)

    expected = torch.tensor([[[0.6, 0.8]]], dtype=packed.dtype, device=packed.device)
    assert torch.allclose(packed, expected, atol=1e-6)


def test_ltx23_connector_repack_renames_qk_norm_keys():
    assert _rename_connector_key(
        "model.diffusion_model.video_embeddings_connector.transformer_1d_blocks.0.attn1.q_norm.weight"
    ) == "video_connector.transformer_blocks.0.attn1.norm_q.weight"
    assert _rename_connector_key(
        "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.1.attn1.k_norm.weight"
    ) == "audio_connector.transformer_blocks.1.attn1.norm_k.weight"


def test_ltx23_vae_config_adds_official_image_encoder_marker():
    with tempfile.TemporaryDirectory() as tmpdir:
        auxiliary_dir = os.path.join(tmpdir, "aux")
        os.makedirs(os.path.join(auxiliary_dir, "vae"), exist_ok=True)

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
        config = _build_vae_config(auxiliary_dir)

    assert config["encoder_spatial_padding_mode"] == "zeros"
    assert config["decoder_spatial_padding_mode"] == "reflect"
    assert config["use_official_image_encoder"] is True
    assert config["official_image_encoder_subdir"] == "ltx23_image_encoder"


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
            keys = sorted(f.keys())
            assert keys == [
                "conv_in.conv.weight",
                "per_channel_statistics.mean-of-means",
            ]


def test_ltx23_latent_preparation_samples_directly_in_packed_space():
    device = get_local_torch_device()
    generator_device = device.type
    generator = torch.Generator(generator_device).manual_seed(123)
    req = Req(
        sampling_params=SamplingParams(
            num_outputs_per_prompt=1,
            num_frames=121,
            height=512,
            width=768,
        ),
        prompt="prompt",
        prompt_embeds=[torch.zeros(1, 1, 1)],
        generator=[generator],
        generate_audio=True,
    )
    pipeline_config = get_model_info(
        "Lightricks/LTX-2.3", backend="sglang"
    ).pipeline_config_cls()
    server_args = SimpleNamespace(pipeline_config=pipeline_config)
    stage = LTX2AVLatentPreparationStage(
        scheduler=SimpleNamespace(init_noise_sigma=1.0)
    )

    batch = stage.forward(req, server_args)

    video_shape = pipeline_config.prepare_latent_shape(
        batch, batch.batch_size, stage.adjust_video_length(batch, server_args)
    )
    audio_shape = pipeline_config.prepare_audio_latent_shape(
        batch, batch.batch_size, batch.num_frames
    )
    expected_generator = torch.Generator(generator_device).manual_seed(123)
    expected_video = torch.randn(
        LTX2AVLatentPreparationStage._packed_video_latent_shape(
            video_shape, pipeline_config
        ),
        generator=expected_generator,
        device=device,
        dtype=torch.float32,
    )
    expected_audio = torch.randn(
        LTX2AVLatentPreparationStage._packed_audio_latent_shape(audio_shape),
        generator=expected_generator,
        device=device,
        dtype=torch.float32,
    )

    assert torch.allclose(batch.latents, expected_video)
    assert torch.allclose(batch.audio_latents, expected_audio)
