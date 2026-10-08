# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Wan-Animate-2 configs, registry wiring, request handling and the
preprocessing helpers that feed the conditioning tensors.

No models, GPU, or network: the registry detector tests patch
``maybe_download_model_index``; stages run with stand-in encoders.
"""

import logging
import math
import os
from dataclasses import fields, replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.dits.wan_animate_2 import (
    WanAnimate2ArchConfig,
    WanAnimate2Config,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan_Animate_2_14B_Config
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.configs.sample.wan import Wan_Animate_2_14B_SamplingParam
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    get_sampling_request_extra_fields,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackendEnum,
)
from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
    _supported_init_kwargs,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2 import (
    WanAnimate2Transformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_block import (
    WanAnimate2TransformerBlock,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_dpm_solver_multistep import (
    DPMSolverMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.wan_animate_2_pipeline import (
    WanAnimate2Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.before_denoising import (
    WanAnimate2BeforeDenoisingStage,
    WanAnimate2RequestState,
    _WanAnimate2Inputs,
    build_clip_conditioning,
    build_schedule,
    generation_token_grid,
    request_state_from_batch,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.denoising import (
    WanAnimate2DenoisingStage,
    get_sampling_sigmas,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.encoder_adapters import (
    WanAnimate2VaeAdapter,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.preprocess import (
    LetterboxInfo,
    get_frame_indices,
    get_padding_len,
    letterbox_resize,
    make_conditioning_mask,
    read_reference_video_frames,
    resize_by_area,
    write_reference_video,
    zigzag_padding,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.server_warmup import prepare_warmup_image_path
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    SERVER_WARMUP_VIDEO_STEPS,
    build_warmup_reqs,
    lighten_warmup_req,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

# Wan_Animate_2_14B_Config


def test_text_encoder_counts_tokens_before_device_transfer():
    config = Wan_Animate_2_14B_Config()
    tokens = BatchEncoding(
        {
            "input_ids": torch.tensor([[1, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        }
    )
    inputs = config.tokenize_prompt(["prompt"], lambda *args, **kwargs: tokens, {})
    inputs.to("meta")
    assert (
        inputs["input_ids"].device.type
        == inputs["attention_mask"].device.type
        == "meta"
    )
    assert inputs["num_tokens"] == 2
    output = SimpleNamespace(last_hidden_state=torch.empty(1, 4, 8, device="meta"))
    assert config.postprocess_text_funcs[0](output, inputs).shape == (2, 8)
    assert not WanAnimate2BeforeDenoisingStage.has_deduplicated_output_fields()


def test_pipeline_keeps_native_modules_visible_to_memory_managers():
    path = "sglang.multimodal_gen.runtime.pipelines.wan_animate_2_pipeline"
    vae = torch.nn.Identity()
    vae.latents_mean, vae.latents_std = [0.0], [1.0]
    modules = {
        "text_encoder": torch.nn.Identity(),
        "image_encoder": torch.nn.Identity(),
        "vae": vae,
        "tokenizer": object(),
        "image_processor": SimpleNamespace(
            crop_size={"height": 224, "width": 224},
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        ),
        "transformer": object(),
        "scheduler": object(),
    }
    stub = SimpleNamespace(
        get_module=modules.__getitem__,
        add_stage=lambda **kwargs: None,
        add_stage_factory=lambda *args: None,
    )
    with (
        patch(f"{path}.get_local_torch_device", return_value=torch.device("cpu")),
        patch(f"{path}.WanAnimate2BeforeDenoisingStage") as before,
        patch(f"{path}.WanAnimate2DenoisingStage") as denoise,
    ):
        WanAnimate2Pipeline.create_pipeline_stages(
            stub, SimpleNamespace(pipeline_config=Wan_Animate_2_14B_Config())
        )
    args = before.call_args.kwargs
    assert "load_modules" not in vars(WanAnimate2Pipeline)
    assert args["text_encoder"] is modules["text_encoder"]
    assert args["tokenizer"] is modules["tokenizer"]
    assert args["image_encoder"].model is modules["image_encoder"]
    assert args["vae"].vae is vae
    assert denoise.call_args.kwargs["vae"] is args["vae"]


def test_single_expert_has_no_boundary_switching():
    # Single-expert model: no low-noise expert switch via boundary_ratio.
    config = Wan_Animate_2_14B_Config()
    assert config.boundary_ratio is None
    assert config.dit_config.arch_config.boundary_ratio is None


@pytest.mark.parametrize(
    "tp_size, ulysses_degree, enable_cfg_parallel, rejected",
    [
        (2, 2, False, True),  # TP2 x Ulysses2 (4 GPUs)
        (2, 2, True, True),  # TP2 x Ulysses2 auto-derived from --num-gpus 8 with CFG
        (2, 1, False, False),  # TP2 alone
        (1, 2, False, False),  # Ulysses2 alone
        (4, 1, False, False),  # TP4
        (1, 2, True, False),  # CFG-parallel x Ulysses2
    ],
)
def test_validate_server_args_rejects_tp_combined_with_sequence_parallel(
    tp_size, ulysses_degree, enable_cfg_parallel, rejected
):
    """TP combined with Ulysses yields a wrong video on this model (12 dB PSNR vs 1 GPU), so
    the server must refuse the layout at startup instead of after loading 14B weights."""
    server_args = _parallel_server_args(
        tp_size=tp_size,
        ulysses_degree=ulysses_degree,
        enable_cfg_parallel=enable_cfg_parallel,
    )
    config = Wan_Animate_2_14B_Config()
    if rejected:
        with pytest.raises(ValueError, match=r"--tp-size 2 .*--ulysses-degree 2"):
            config.validate_server_args(server_args)
    else:
        config.validate_server_args(server_args)


def _parallel_server_args(
    *,
    tp_size: int = 1,
    ulysses_degree: int = 1,
    ring_degree: int = 1,
    enable_cfg_parallel: bool = False,
    use_fsdp_inference: bool = False,
    attention_backend: str | None = None,
    attention_backend_explicit: bool = False,
    component_attention_backends: dict[str, str] | None = None,
) -> SimpleNamespace:
    """The normalized ServerArgs fields validate_server_args reads (sp_degree = ulysses * ring)."""
    requested = dict(component_attention_backends or {})
    explicit = {"attention_backend"} if attention_backend_explicit else set()
    return SimpleNamespace(
        tp_size=tp_size,
        sp_degree=ulysses_degree * ring_degree,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        enable_cfg_parallel=enable_cfg_parallel,
        use_fsdp_inference=use_fsdp_inference,
        attention_backend=attention_backend,
        is_arg_explicitly_set=lambda name: name in explicit,
        requested_component_attention_backend=lambda name: requested.get(name),
    )


def test_validate_server_args_rejects_fsdp_before_any_weights_load():
    with pytest.raises(NotImplementedError, match="use-fsdp-inference"):
        Wan_Animate_2_14B_Config().validate_server_args(
            _parallel_server_args(use_fsdp_inference=True)
        )


def test_validate_server_args_rejects_ring_sequence_parallelism():
    with pytest.raises(NotImplementedError, match="ring sequence parallelism"):
        Wan_Animate_2_14B_Config().validate_server_args(
            _parallel_server_args(ulysses_degree=1, ring_degree=2)
        )


@pytest.mark.parametrize(
    "tp_size, ulysses_degree, rejected",
    [
        (1, 8, False),  # 40 heads / 8
        (1, 3, True),  # 40 heads not divisible by 3
        (1, 16, True),  # 40 heads not divisible by 16
        (3, 1, True),  # 40 heads not divisible by tp 3
        (8, 1, False),
    ],
)
def test_validate_server_args_checks_head_divisibility(
    tp_size, ulysses_degree, rejected
):
    """Ulysses splits the TP-local heads across ranks; a non-dividing degree must be
    rejected at startup, before any weights load."""
    server_args = _parallel_server_args(tp_size=tp_size, ulysses_degree=ulysses_degree)
    config = Wan_Animate_2_14B_Config()
    if rejected:
        with pytest.raises(ValueError, match="divisible by"):
            config.validate_server_args(server_args)
    else:
        config.validate_server_args(server_args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attention_backend": "fa", "attention_backend_explicit": True},
        {"component_attention_backends": {"transformer": "torch_sdpa"}},
    ],
)
def test_validate_server_args_rejects_a_requested_dit_attention_backend(kwargs):
    # The in-context self-attention is flex_attention only, so a requested backend
    # would be honoured by the cross-attention and silently dropped by the main path.
    with pytest.raises(ValueError, match="flex_attention only"):
        Wan_Animate_2_14B_Config().validate_server_args(_parallel_server_args(**kwargs))


def test_validate_server_args_keeps_a_platform_default_attention_backend():
    # ROCm fills attention_backend in when the user gave none; that is not a request.
    Wan_Animate_2_14B_Config().validate_server_args(
        _parallel_server_args(
            attention_backend="aiter", attention_backend_explicit=False
        )
    )


def test_validate_server_args_rejects_mxfp8_offline_qk_rotation(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION", "true")
    with pytest.raises(ValueError, match="MXFP8"):
        Wan_Animate_2_14B_Config().validate_server_args(_parallel_server_args())


class _RotatedQuantConfig:
    """A quantized checkpoint that ships offline Q/K rotation matrices for one block."""

    def __init__(self, prefix: str) -> None:
        self.quant_description = {
            f"{prefix}.attn1.q_rot": "FLOAT",
            f"{prefix}.attn1.k_rot": "FLOAT",
        }

    def get_quant_method(self, layer, prefix: str) -> UnquantizedLinearMethod:
        return UnquantizedLinearMethod()


def test_block_refuses_a_checkpoint_with_offline_qk_rotation(monkeypatch):
    # The parent block registers q_rot / k_rot and applies them in its own attention
    # path; forward_ref / forward_gen never do, so loading would be silently wrong.
    monkeypatch.setenv("SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION", "true")
    _ensure_single_process_parallel_runtime()
    with torch.device("meta"), pytest.raises(ValueError, match="offline Q/K rotation"):
        WanAnimate2TransformerBlock(
            64,
            128,
            2,
            "rms_norm",
            True,
            1e-6,
            None,
            {AttentionBackendEnum.TORCH_SDPA},
            prefix="blocks.0",
            quant_config=_RotatedQuantConfig("blocks.0"),
        )


# WanAnimate2ArchConfig / WanAnimate2Config


def test_in_out_channels_match_in_context_io():
    # in_channels = 16 noise + 20 conditioning; out_channels = 16 latent.
    arch = WanAnimate2ArchConfig()
    assert arch.in_channels == 36
    assert arch.out_channels == 16


def test_backbone_dims_reuse_wan22_i2v_14b():
    arch = WanAnimate2ArchConfig()
    assert arch.num_layers == 40
    assert arch.image_dim == 1280
    assert arch.added_kv_proj_dim == 5120


def test_post_init_derives_hidden_size_and_latent_channels():
    arch = WanAnimate2ArchConfig()
    assert arch.num_attention_heads == 40
    assert arch.attention_head_dim == 128
    assert arch.hidden_size == 5120
    assert arch.num_channels_latents == 16


# The 1303 tensor names of Wan2.2-Animate-2-14B-Diffusers/transformer (the safetensors
# index's weight_map): 23 top-level tensors plus 32 per block for 40 blocks.
_CHECKPOINT_TOP_LEVEL_KEYS = (
    "head.head.bias",
    "head.head.weight",
    "head.modulation",
    "img_emb.proj.0.bias",
    "img_emb.proj.0.weight",
    "img_emb.proj.1.bias",
    "img_emb.proj.1.weight",
    "img_emb.proj.3.bias",
    "img_emb.proj.3.weight",
    "img_emb.proj.4.bias",
    "img_emb.proj.4.weight",
    "patch_embedding.bias",
    "patch_embedding.weight",
    "text_embedding.0.bias",
    "text_embedding.0.weight",
    "text_embedding.2.bias",
    "text_embedding.2.weight",
    "time_embedding.0.bias",
    "time_embedding.0.weight",
    "time_embedding.2.bias",
    "time_embedding.2.weight",
    "time_projection.1.bias",
    "time_projection.1.weight",
)
_CHECKPOINT_BLOCK_KEYS = (
    "cross_attn.add_k_proj.bias",
    "cross_attn.add_k_proj.weight",
    "cross_attn.add_v_proj.bias",
    "cross_attn.add_v_proj.weight",
    "cross_attn.norm_added_k.weight",
    "cross_attn.norm_k.weight",
    "cross_attn.norm_q.weight",
    "cross_attn.to_k.bias",
    "cross_attn.to_k.weight",
    "cross_attn.to_out.0.bias",
    "cross_attn.to_out.0.weight",
    "cross_attn.to_q.bias",
    "cross_attn.to_q.weight",
    "cross_attn.to_v.bias",
    "cross_attn.to_v.weight",
    "ffn.0.bias",
    "ffn.0.weight",
    "ffn.2.bias",
    "ffn.2.weight",
    "modulation",
    "norm3.bias",
    "norm3.weight",
    "self_attn.norm_k.weight",
    "self_attn.norm_q.weight",
    "self_attn.to_k.bias",
    "self_attn.to_k.weight",
    "self_attn.to_out.0.bias",
    "self_attn.to_out.0.weight",
    "self_attn.to_q.bias",
    "self_attn.to_q.weight",
    "self_attn.to_v.bias",
    "self_attn.to_v.weight",
)


def _checkpoint_keys() -> list[str]:
    keys = list(_CHECKPOINT_TOP_LEVEL_KEYS)
    for block in range(WanAnimate2ArchConfig().num_layers):
        keys.extend(f"blocks.{block}.{key}" for key in _CHECKPOINT_BLOCK_KEYS)
    return keys


def test_every_checkpoint_key_maps_onto_a_distinct_dit_parameter():
    """The DiT loads with the shared Wan rules plus the Wan-Animate-2 extras; a rule that
    stops matching leaves a tensor unmapped (skipped with a warning) or lands two tensors on
    one parameter, both of which load silently wrong weights."""
    keys = _checkpoint_keys()
    assert len(keys) == 1303
    map_fn = get_param_names_mapping(WanAnimate2ArchConfig().param_names_mapping)
    mapped = {key: map_fn(key)[0] for key in keys}
    unmapped = sorted(key for key, name in mapped.items() if name == key)
    assert unmapped == []
    assert len(set(mapped.values())) == len(keys)

    _ensure_single_process_parallel_runtime()
    with torch.device("meta"):
        model = WanAnimate2Transformer3DModel(WanAnimate2Config(), hf_config={})
    assert set(mapped.values()) == set(model.state_dict())


def _ensure_single_process_parallel_runtime() -> None:
    # The DiT's parallel linear layers need the TP group even on a meta device.
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def test_official_checkpoint_config_names_update_the_arch_config():
    """transformer/config.json carries the official names (dim, in_dim, num_heads, ...);
    the loader's update_model_arch must land them on the fields the model reads."""
    config = WanAnimate2Config()
    config.update_model_arch(
        {
            "dim": 5120,
            "in_dim": 36,
            "out_dim": 16,
            "num_heads": 40,
            "patch_size": [1, 2, 2],
            "refer_offset_t": 2,
            "log_scale": -1.3,
            "text_len": 512,
        }
    )
    assert config.num_attention_heads == 40
    assert config.attention_head_dim == 128
    assert config.hidden_size == 5120
    assert config.in_channels == 36
    assert config.out_channels == 16
    assert config.patch_size == (1, 2, 2)
    assert config.refer_offset_t == 2
    assert config.log_scale == -1.3
    assert config.text_len == 512


def test_text_len_defaults_to_the_checkpoint_value():
    """The pipeline pads prompts to ``arch_config.text_len``: the checkpoint's
    transformer/config.json sets it and the default covers a config without the key."""
    assert WanAnimate2ArchConfig().text_len == 512
    assert Wan_Animate_2_14B_Config().dit_config.arch_config.text_len == 512


def test_checkpoint_config_without_image_embedding_is_rejected():
    # The model always builds the image-embedding cross-attention; a checkpoint that
    # declares use_img_emb false would load and run the wrong architecture silently.
    config = WanAnimate2Config()
    with pytest.raises(ValueError, match="use_img_emb"):
        config.update_model_arch({"use_img_emb": False})


# Wan_Animate_2_14B_SamplingParam


@pytest.mark.parametrize(
    "requested, expected",
    [
        (37, 37),  # 4k+1 -> unchanged
        (80, 81),  # 4k -> up one frame
        (42, 41),  # 4k+2 -> down
        (43, 41),  # 4k+3 -> down
        (4, 5),  # smallest accepted clip
    ],
)
def test_clip_len_rounds_to_4k_plus_1(requested, expected, caplog):
    # VAE temporal stride is 4, so a clip must be 4k+1 frames; the rounding is logged
    # once, and only when it changed the value.
    with caplog.at_level(logging.INFO):
        params = Wan_Animate_2_14B_SamplingParam(clip_len=requested)
    assert params.clip_len == expected
    assert params.clip_len % 4 == 1
    rounding_logs = [r for r in caplog.records if "is not 4k+1" in r.getMessage()]
    assert len(rounding_logs) == (0 if requested == expected else 1)
    if rounding_logs:
        assert f"clip_len={requested}" in rounding_logs[0].getMessage()
        assert f"clip_len={expected}" in rounding_logs[0].getMessage()


@pytest.mark.parametrize("requested", [-3, 0, 1, 2, 3])
def test_clip_len_below_one_temporal_stride_is_rejected(requested):
    # A clip shorter than one VAE temporal stride is rejected at admission, not in a stage.
    with pytest.raises(ValueError, match=rf"clip_len={requested} is not supported"):
        Wan_Animate_2_14B_SamplingParam(clip_len=requested)


def test_num_frames_is_not_mirrored_from_clip_len():
    """The output length follows the reference video, so clip_len must not be written
    into num_frames: a mirrored value made `dataclasses.replace(num_frames=...)` snap back,
    and the auto-residency probe (which shrinks a request by replacing num_frames until its
    estimate fits) looped forever on cards where the full probe did not fit."""
    params = Wan_Animate_2_14B_SamplingParam(clip_len=37)
    assert params.num_frames == SamplingParams.num_frames
    assert params.adjust_frames is False

    lighter = replace(Wan_Animate_2_14B_SamplingParam(num_frames=37), num_frames=17)
    assert lighter.num_frames == 17
    assert lighter.clip_len == 37


def test_lighten_warmup_req_shrinks_a_wan_animate_2_probe(tmp_path):
    # Each shrink step must strictly reduce width*height*num_frames, the quantity the
    # auto-residency probe loop terminates on; the ladder bottoms out at the 16-pixel floor.
    server_args = _server_args(str(tmp_path))
    req = Req(
        sampling_params=Wan_Animate_2_14B_SamplingParam(num_frames=37),
        prompt="probe",
        width=640,
        height=800,
    )
    units = []
    for _ in range(64):
        params = req.sampling_params
        units.append(params.width * params.height * params.num_frames)
        req = lighten_warmup_req(server_args, req)
        if req is None:
            break
    assert req is None, "the shrink ladder never bottomed out"
    assert units == sorted(units, reverse=True) and len(set(units)) == len(units)


def test_multi_gpu_frame_alignment_leaves_num_frames_alone(tmp_path, caplog):
    # The stages read clip_len, so the multi-GPU frame alignment must not touch num_frames.
    server_args = _server_args(str(tmp_path))
    server_args.num_gpus = 4
    server_args.comfyui_mode = True
    params = Wan_Animate_2_14B_SamplingParam(
        prompt="p", image_path="ref.png", video_path="drv.mp4", num_frames=37
    )
    with caplog.at_level(logging.INFO):
        params._adjust_visual_fields(server_args, server_args.pipeline_config)
    assert params.num_frames == 37
    messages = [r.getMessage() for r in caplog.records]
    assert not any("based on number of GPUs" in m for m in messages)
    assert any("ignores num_frames=37" in m for m in messages)


@pytest.mark.parametrize("field_name", ["enable_teacache", "enable_spectrum"])
def test_sampling_params_reject_the_shared_dit_cache_heuristics(field_name):
    # Every block runs at every step in the in-context loop; the flags would be accepted
    # and change nothing.
    with pytest.raises(ValueError, match="does not support"):
        Wan_Animate_2_14B_SamplingParam(**{field_name: True})


# Registry resolution (no network)


def test_registered_hf_path_resolves_to_wan_animate_configs():
    info = _get_config_info("Wan-AI/Wan2.2-Animate-2-14B-Diffusers")
    assert info is not None
    assert info.pipeline_config_cls is Wan_Animate_2_14B_Config
    assert info.sampling_param_cls is Wan_Animate_2_14B_SamplingParam


@pytest.mark.parametrize(
    "path", ["some/local/dir/wan_animate_2", "my-wan2.2-animate-2-ckpt"]
)
def test_detector_resolves_wan_animate_style_paths(path):
    with patch(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        return_value={},
    ):
        info = _get_config_info(path)
    assert info is not None
    assert info.pipeline_config_cls is Wan_Animate_2_14B_Config
    assert info.sampling_param_cls is Wan_Animate_2_14B_SamplingParam


def test_model_index_class_name_routes_a_checkpoint_at_any_path():
    # A local checkpoint directory need not carry a model token in its name; its
    # model_index.json _class_name is what identifies it.
    with patch(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        return_value={"_class_name": "WanAnimate2Pipeline"},
    ):
        info = _get_config_info("/models/ckpt-7f3a")
    assert info is not None
    assert info.pipeline_config_cls is Wan_Animate_2_14B_Config


def test_v1_animate_checkpoint_does_not_route_to_wan_animate_2():
    # Wan2.2-Animate-14B (v1) is a different model and pipeline; a detector loosened to
    # "animate" would pass the positive cases above and mis-route it here.
    with patch(
        "sglang.multimodal_gen.registry.maybe_download_model_index",
        return_value={"_class_name": "WanAnimatePipeline"},
    ):
        info = _get_config_info("Wan-AI/Wan2.2-Animate-14B")
    assert info is None or info.pipeline_config_cls is not Wan_Animate_2_14B_Config


# /v1/videos routing: the model-specific knobs are declared extras, the reference image and
# driving video travel in the base request fields.


def test_video_api_routes_the_model_knobs_as_declared_extra_fields():
    extras = get_sampling_request_extra_fields(Wan_Animate_2_14B_SamplingParam, "video")
    assert extras == {"clip_len", "prompt_ref", "enable_audio"}
    base_fields = {field.name for field in fields(SamplingParams)}
    for name in extras:
        # Not a base request field (no silent duplicate), and not on the base params
        # either, so the declaration is what makes the API forward the value.
        assert name not in VideoGenerationsRequest.model_fields, name
        assert name not in base_fields, name
    for name in ("input_reference", "reference_url", "video_path", "video_url"):
        assert name in VideoGenerationsRequest.model_fields, name


# Preprocessing helpers (preprocess.py): their numerics feed the conditioning tensors and have
# no diffusers equivalent, so exact outputs are pinned.

# get_frame_indices(num_frames_source, fps_source, num_frames_target, fps_target):
#   idx = clip(round(arange(num_frames_target) / fps_target * fps_source), 0, num_frames_source - 1)


@pytest.mark.parametrize(
    "args, expected",
    [
        ((100, 16, 5, 16), [0, 1, 2, 3, 4]),  # fps match: identity
        ((100, 30, 5, 15), [0, 2, 4, 6, 8]),  # downsample by the fps ratio
        ((3, 30, 6, 15), [0, 2, 2, 2, 2, 2]),  # clipped to the last available frame
    ],
)
def test_get_frame_indices_resamples_by_fps_ratio(args, expected):
    assert get_frame_indices(*args) == expected


def test_get_frame_indices_uses_banker_rounding():
    # times*video_fps = [0.0, 0.5, 1.0, 1.5]; np.round is half-to-even, so 0.5 -> 0 and
    # 1.5 -> 2. Python round()/int() truncation would shift the sampled frames.
    assert get_frame_indices(10, 1, 4, 2) == [0, 0, 1, 2]


# get_padding_len(num_frames, clip_len, num_frames_conditioning=1):
#   remaining = (num_frames - num_frames_conditioning) % (clip_len - num_frames_conditioning)
#   pad = 28 - remaining if remaining < 28 else 4 - remaining % 4

_PADDING_GOLDENS = {
    1: 29,  # remaining 0  -> pad 28
    28: 29,  # remaining 27 -> pad 1 (boundary, still < 28)
    29: 33,  # remaining 28 -> else branch, pad 4
    32: 33,  # remaining 31 -> pad 1
    50: 53,  # remaining 49 -> pad 3
    81: 109,  # remaining 0  -> pad 28
    100: 109,  # remaining 19 -> pad 9
}


def test_get_padding_len_matches_reference_goldens():
    for input_len, expected in _PADDING_GOLDENS.items():
        assert get_padding_len(input_len, 81) == expected, input_len


@pytest.mark.parametrize("clip_len", [37, 81])
def test_get_padding_len_output_is_4k_plus_1(clip_len):
    # Padded length must land on a 4k+1 boundary (VAE temporal stride 4) for the shipped
    # default clip length and the official one.
    for input_len in range(1, 200):
        assert (get_padding_len(input_len, clip_len) - 1) % 4 == 0, input_len


# build_schedule(reference_video_frames, clip_len, num_frames_conditioning)
#   -> [_PerClipDenoisingMetadata]; frame_end_index is exclusive.


def _frames(num_frames: int, height: int = 1, width: int = 1) -> list[np.ndarray]:
    return [
        np.full((height, width, 3), index % 256, np.uint8)
        for index in range(num_frames)
    ]


def _as_tuples(schedule):
    return [
        (
            clip.clip_index,
            clip.frame_start_index,
            clip.frame_end_index,
            clip.num_frames,
            clip.num_frames_conditioning_for_clip,
        )
        for clip in schedule
    ]


@pytest.mark.parametrize(
    "num_frames, clip_len, num_frames_conditioning, expected",
    [
        # Exact tiling: 37 + 2 * (37 - 1) = 109; every clip after the first reuses one frame.
        (109, 37, 1, [(0, 0, 37, 37, 0), (1, 36, 73, 37, 1), (2, 72, 109, 37, 1)]),
        # Last clip shrinks to the frames left.
        (50, 37, 1, [(0, 0, 37, 37, 0), (1, 36, 50, 14, 1)]),
        # Frame 72 is already covered by clip 1, so no 1-frame clip is appended...
        (73, 37, 1, [(0, 0, 37, 37, 0), (1, 36, 73, 37, 1)]),
        # ...but one extra new frame does produce a clip.
        (74, 37, 1, [(0, 0, 37, 37, 0), (1, 36, 73, 37, 1), (2, 72, 74, 2, 1)]),
        # Overlap follows num_frames_conditioning, not a hardcoded 1.
        (
            14,
            5,
            2,
            [(0, 0, 5, 5, 0), (1, 3, 8, 5, 2), (2, 6, 11, 5, 2), (3, 9, 14, 5, 2)],
        ),
    ],
)
def test_build_schedule_clip_boundaries(
    num_frames, clip_len, num_frames_conditioning, expected
):
    schedule = build_schedule(_frames(num_frames), clip_len, num_frames_conditioning)
    assert _as_tuples(schedule) == expected
    covered = set()
    for clip in schedule:
        covered.update(range(clip.frame_start_index, clip.frame_end_index))
    assert covered == set(range(num_frames))


@pytest.mark.parametrize("num_frames", [5, 37, 50, 100, 236])
def test_build_schedule_after_padding_yields_4k_plus_1_clips(num_frames):
    # Same call chain as WanAnimate2BeforeDenoisingStage: pad first, then schedule.
    clip_len = 37
    padded_len = get_padding_len(num_frames, clip_len)
    schedule = build_schedule(
        zigzag_padding(_frames(num_frames), padded_len), clip_len, 1
    )
    assert schedule[-1].frame_end_index == padded_len
    assert all(clip.num_frames % 4 == 1 for clip in schedule)
    assert all(clip.num_frames == clip_len for clip in schedule[:-1])
    # get_padding_len's reason to exist: the last clip generates at least 28 new frames.
    assert schedule[-1].num_frames - 1 >= 28


def test_build_schedule_rejects_video_too_short_for_a_clip():
    with pytest.raises(ValueError, match="0 clips"):
        build_schedule(_frames(1), 37, 1)


# zigzag_padding(array, target_len): bounce the read index 0->end->0 to pad up to target_len.


@pytest.mark.parametrize(
    "array, target_len, expected",
    [
        ([0, 1, 2], 8, [0, 1, 2, 1, 0, 1, 2, 1]),
        (["a", "b"], 5, ["a", "b", "a", "b", "a"]),  # bounces at both ends every step
    ],
)
def test_zigzag_padding_bounces_forward_then_backward(array, target_len, expected):
    assert zigzag_padding(array, target_len) == expected


def test_zigzag_padding_rejects_shortening():
    with pytest.raises(ValueError):
        zigzag_padding([0, 1, 2], 2)


def test_zigzag_padding_returns_independent_deep_copies():
    # A shallow copy would alias reused frames across the zigzag.
    source = [[0], [1]]
    out = zigzag_padding(source, 4)
    out[0][0] = 99
    assert source == [[0], [1]]


# make_conditioning_mask(latent_t, latent_h, latent_w, num_prefix_conditioning_frames, device)
#   -> [4, latent_t, latent_h, latent_w]; the prefix is counted in pixel frames and the VAE
#   maps 4k+1 pixel frames to k+1 latent frames.


@pytest.mark.parametrize(
    "latent_t, prefix_pixel_frames, expected_frame_flags",
    [
        (1, 1, [1]),  # the reference image: its single frame is given
        (
            2,
            1,
            [1, 0],
        ),  # one fed-back pixel frame conditions the first latent frame only
        (3, 5, [1, 1, 0]),  # 5 = 1 + 4 pixel frames -> two latent frames
        (3, 0, [0, 0, 0]),  # clip 0: nothing fed back from a previous clip
        (
            3,
            9,
            [1, 1, 1],
        ),  # the reference video: every one of its 9 pixel frames is given
    ],
)
def test_make_conditioning_mask_prefix_in_pixel_frames_maps_to_latent_frames(
    latent_t, prefix_pixel_frames, expected_frame_flags
):
    mask = make_conditioning_mask(latent_t, 1, 2, prefix_pixel_frames, device="cpu")
    assert tuple(mask.shape) == (4, latent_t, 1, 2)
    assert mask.dtype == torch.float32 and mask.device.type == "cpu"
    for channel in range(4):
        assert mask[channel, :, 0, 0].tolist() == expected_frame_flags, channel
    # Spatially constant.
    assert torch.equal(mask, mask[:, :, :1, :1].expand_as(mask))


def test_make_conditioning_mask_is_channel_major():
    # Flat golden pins the reshape/transpose exactly: [4, latent_t] channel-major.
    mask = make_conditioning_mask(2, 1, 1, 1, device="cpu")
    assert mask.flatten().tolist() == [1, 0, 1, 0, 1, 0, 1, 0]


# letterbox_resize / resize_by_area / LetterboxInfo.crop: how the request ``size`` is applied
# to the reference image and how the bars come back off every output frame.


@pytest.mark.parametrize(
    "source_hw, pad_axis, content_hw",
    [((40, 80), "height", (32, 64)), ((80, 40), "width", (64, 32))],
)
def test_letterbox_crop_round_trips_the_content_region(source_hw, pad_axis, content_hw):
    image = np.full((*source_hw, 3), 200, np.uint8)
    canvas, info = letterbox_resize(image, height=64, width=64)
    assert canvas.shape == (64, 64, 3) and canvas.dtype == np.uint8
    assert info.pad_axis == pad_axis
    content = info.crop(canvas[None])
    assert content.shape == (1, *content_hw, 3)
    # Exactly the content pixels survive the crop; the bars are the only other pixels.
    assert (content == 200).all()
    assert int(canvas.sum()) == int(content.sum())


@pytest.mark.parametrize(
    "source_hw, expected_wh",
    [
        ((1920, 1080), (528, 944)),  # 9:16 reference under the 640x800 default budget
        ((1080, 1920), (944, 528)),
        ((1000, 1000), (704, 704)),  # 1:1
        ((500, 400), (640, 800)),  # the budget's own 4:5 aspect fills it exactly
    ],
)
def test_resize_by_area_treats_the_request_size_as_a_pixel_area_budget(
    source_hw, expected_wh
):
    # The /v1/videos ``size`` and the cookbook document 640x800 -> 528x944 (9:16) and
    # 704x704 (1:1): the request fixes the pixel area, the reference image fixes the aspect
    # ratio, and both sides are floored to a multiple of 16.
    image = np.zeros((*source_hw, 3), np.uint8)
    out, _ = resize_by_area(image, 640 * 800, divisor=16)
    assert (out.shape[1], out.shape[0]) == expected_wh
    assert out.shape[0] * out.shape[1] <= 640 * 800
    assert out.shape[0] % 16 == 0 and out.shape[1] % 16 == 0


# build_clip_conditioning: per-clip tensors from the request state with stand-in encoders.


def _request_state_for_clip_tests(frames, schedule, clip_len):
    height, width = frames[0].shape[:2]
    latent_h, latent_w = height // 8, width // 8
    inputs = _WanAnimate2Inputs(
        reference_image_path="ref.png",
        reference_video_path="drv.mp4",
        prompt="p",
        prompt_ref="r",
        negative_prompt="",
        width=width,
        height=height,
        clip_len=clip_len,
        num_frames_conditioning=1,
        fps=16,
        seed=0,
        num_inference_steps=2,
        guidance_scale=1.0,
        enable_audio=False,
    )
    return WanAnimate2RequestState(
        device=torch.device("cpu"),
        sp_size=1,
        generator=torch.Generator().manual_seed(0),
        inputs=inputs,
        letterbox_info=LetterboxInfo(pad_axis="width", padding=0, content_len=width),
        reference_image_height=height,
        reference_image_width=width,
        latent_h=latent_h,
        latent_w=latent_w,
        reference_image_condition=torch.ones(
            20, 1, latent_h, latent_w, dtype=torch.bfloat16
        ),
        reference_image_embeddings=torch.zeros(1, 257, 1280, dtype=torch.bfloat16),
        prompt_embeddings=torch.zeros(1, 4096, dtype=torch.bfloat16),
        negative_prompt_embeddings=torch.zeros(1, 4096, dtype=torch.bfloat16),
        prompt_ref_embeddings=torch.zeros(1, 4096, dtype=torch.bfloat16),
        reference_video_frames=frames,
        schedule=schedule,
        num_reference_video_frames=len(frames),
        full_clip_grid_sizes=generation_token_grid(clip_len, latent_h, latent_w),
    )


def test_build_clip_conditioning_shapes_and_prefix_mask_contract():
    """Clip 0 has no fed-back frames and clip k has ``num_frames_conditioning`` of them; the
    mask channels of ``generation_condition`` must say so frame by frame, after the
    reference-image frame, or the DiT is told the wrong pixels are given."""
    clip_len = 9  # 4k+1 -> 3 latent frames of its own, 4 with the reference image
    frames = _frames(17, height=32, width=48)  # two full clips sharing one frame
    schedule = build_schedule(frames, clip_len, 1)
    assert [clip.num_frames for clip in schedule] == [clip_len, clip_len]
    state = _request_state_for_clip_tests(frames, schedule, clip_len)

    def vae_encoder(videos):
        return [
            torch.zeros(16, (v.shape[1] - 1) // 4 + 1, v.shape[2] // 8, v.shape[3] // 8)
            for v in videos
        ]

    def image_embedder(videos):
        return torch.zeros(len(videos), 257, 1280)

    clip_0 = build_clip_conditioning(
        state, schedule[0], None, vae_encoder=vae_encoder, image_embedder=image_embedder
    )
    previous_frames = torch.zeros(3, 1, 32, 48)
    clip_1 = build_clip_conditioning(
        state,
        schedule[1],
        previous_frames,
        vae_encoder=vae_encoder,
        image_embedder=image_embedder,
    )

    for clip in (clip_0, clip_1):
        # [20, latent_t, latent_h, latent_w]: 4 mask + 16 latent channels, latent_t = 1 + 3.
        assert tuple(clip.generation_condition.shape) == (20, 4, 4, 6)
        assert clip.generation_condition.dtype == torch.bfloat16
        assert clip.generation_video_grid_sizes == (4, 2, 3)
        assert tuple(clip.reference_video_condition.shape) == (20, 3, 4, 6)
        assert tuple(clip.reference_video_latents.shape) == (16, 3, 4, 6)
        assert clip.reference_video_grid_sizes == (3, 2, 3)
        assert clip.full_clip_grid_sizes == (4, 2, 3)
        assert tuple(clip.reference_video_frame_0_image_embeddings.shape) == (
            1,
            257,
            1280,
        )
        assert tuple(clip.init_noise.shape) == (16, 4, 4, 6)
        # Every reference-video frame is given.
        assert bool((clip.reference_video_condition[:4] == 1).all())
        # Frame 0 is the reference image, given for every clip.
        assert bool((clip.generation_condition[:4, 0] == 1).all())
    # Clip 0: none of its own frames is given; clip 1: the one fed-back frame is.
    assert clip_0.generation_condition[:4, 1:].abs().sum() == 0
    assert bool((clip_1.generation_condition[:4, 1] == 1).all())
    assert clip_1.generation_condition[:4, 2:].abs().sum() == 0

    # A later clip without the previous clip's frames is a caller bug, not a silent clip 0.
    with pytest.raises(ValueError, match="prev_clip_conditioning_frames"):
        build_clip_conditioning(
            state,
            schedule[1],
            None,
            vae_encoder=vae_encoder,
            image_embedder=image_embedder,
        )


def test_build_clip_conditioning_uses_the_stage_drawn_noise_for_clip_0_only():
    # The before-denoising stage draws clip 0's noise into batch.latents; the clip loop must
    # consume that draw instead of drawing again, and later clips must keep drawing.
    clip_len = 9
    frames = _frames(17, height=32, width=48)
    schedule = build_schedule(frames, clip_len, 1)
    state = _request_state_for_clip_tests(frames, schedule, clip_len)
    stage_drawn = torch.full((16, 4, 4, 6), 7.0)
    state.clip_0_init_noise = stage_drawn

    def vae_encoder(videos):
        return [
            torch.zeros(16, (v.shape[1] - 1) // 4 + 1, v.shape[2] // 8, v.shape[3] // 8)
            for v in videos
        ]

    def image_embedder(videos):
        return torch.zeros(len(videos), 257, 1280)

    clip_0 = build_clip_conditioning(
        state, schedule[0], None, vae_encoder=vae_encoder, image_embedder=image_embedder
    )
    clip_1 = build_clip_conditioning(
        state,
        schedule[1],
        torch.zeros(3, 1, 32, 48),
        vae_encoder=vae_encoder,
        image_embedder=image_embedder,
    )
    assert clip_0.init_noise is stage_drawn
    assert clip_1.init_noise is not stage_drawn
    assert tuple(clip_1.init_noise.shape) == (16, 4, 4, 6)
    assert not torch.equal(clip_1.init_noise, stage_drawn)


# Before-denoising stage: _WanAnimate2Inputs.from_request reads the SamplingParam-backed fields


def test_extract_inputs_reads_request_fields_through_the_sampling_params():
    """prompt_ref must reach the reference branch as its own text, separate from the character
    caption that feeds the generation branch."""
    params = Wan_Animate_2_14B_SamplingParam(
        prompt="p",
        image_path="ref.png",
        video_path="drv.mp4",
        width=640,
        height=800,
        clip_len=45,
        fps=24,
        seed=7,
    )
    inputs = _WanAnimate2Inputs.from_request(Req(sampling_params=params))

    # Non-default values prove the fields are read off the request, not the defaults.
    assert (inputs.clip_len, inputs.fps, inputs.seed) == (45, 24, 7)
    assert (inputs.width, inputs.height) == (640, 800)
    assert inputs.num_frames_conditioning == 1
    assert (inputs.reference_image_path, inputs.reference_video_path) == (
        "ref.png",
        "drv.mp4",
    )
    assert inputs.prompt == "p"
    assert inputs.prompt_ref == params.prompt_ref
    assert inputs.prompt_ref != inputs.prompt
    assert inputs.negative_prompt == params.negative_prompt
    assert inputs.enable_audio is True


# Carrying the audio track is a per-request knob, on by default like the official pipeline


def test_enable_audio_is_on_by_default_and_settable_over_the_video_api():
    assert Wan_Animate_2_14B_SamplingParam().enable_audio is True
    assert "enable_audio" not in VideoGenerationsRequest.model_fields
    assert (
        "enable_audio" in Wan_Animate_2_14B_SamplingParam.video_request_extra_fields()
    )
    assert not hasattr(Wan_Animate_2_14B_Config(), "enable_audio")


def _audio_stage(monkeypatch, calls: list[str], *, no_audio_track: bool = False):
    """A before-denoising stage whose audio extractor records its calls instead of decoding
    the file and reports either a silent reference video or a decode failure."""
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.before_denoising as mod

    def fake_extract(path, *, log_info, log_warning, log_no_audio_track):
        calls.append(path)
        if no_audio_track:
            log_no_audio_track(
                "Wan-Animate-2: reference video %s has no audio track", path
            )
        else:
            log_warning("Wan-Animate-2: cannot decode the audio track of %s", path)
        return None, None

    monkeypatch.setattr(mod, "extract_reference_video_audio", fake_extract)
    stage = WanAnimate2BeforeDenoisingStage.__new__(WanAnimate2BeforeDenoisingStage)
    stage._silent_reference_logged = False
    stage._audio_failure_logged = False
    stage.infos = []
    stage.warnings = []
    stage.debugs = []
    stage.log_info = lambda msg, *args: stage.infos.append(msg % args)
    stage.log_warning = lambda msg, *args: stage.warnings.append(msg % args)
    stage.log_debug = lambda msg, *args: stage.debugs.append(msg % args)
    return stage


def _inputs_with_audio(enable_audio: bool):
    params = Wan_Animate_2_14B_SamplingParam(
        prompt="p",
        image_path="ref.png",
        video_path="drv.mp4",
        width=640,
        height=800,
        enable_audio=enable_audio,
    )
    return _WanAnimate2Inputs.from_request(Req(sampling_params=params))


def test_audio_extraction_is_skipped_when_the_request_turns_it_off(monkeypatch):
    calls: list[str] = []
    stage = _audio_stage(monkeypatch, calls)

    assert stage._extract_audio(_inputs_with_audio(False)) == (None, None)
    assert calls == []

    stage._extract_audio(_inputs_with_audio(True))
    assert calls == ["drv.mp4"]


def test_silent_reference_video_logs_info_once_then_debug(monkeypatch):
    stage = _audio_stage(monkeypatch, [], no_audio_track=True)
    inputs = _inputs_with_audio(True)

    stage._extract_audio(inputs)
    stage._extract_audio(inputs)

    assert stage.warnings == []
    assert len(stage.infos) == 1
    assert "no audio track" in stage.infos[0] and "enable_audio=false" in stage.infos[0]
    assert stage.debugs == ["Wan-Animate-2: reference video drv.mp4 has no audio track"]


def test_audio_failures_warn_once_per_process_then_go_to_debug(monkeypatch):
    stage = _audio_stage(monkeypatch, [])
    inputs = _inputs_with_audio(True)

    stage._extract_audio(inputs)
    stage._extract_audio(inputs)

    assert len(stage.warnings) == 1
    assert "drv.mp4" in stage.warnings[0]
    assert "enable_audio=false" in stage.warnings[0]
    assert "debug level" in stage.warnings[0]
    assert stage.debugs == ["Wan-Animate-2: cannot decode the audio track of drv.mp4"]
    # A silent reference afterwards still gets its own first-time info line.
    assert stage.infos == []


# extract_reference_video_audio: the shared in-process reader decodes the track at its native
# rate; a video without an audio track is reported through the dedicated logger, any reader
# failure through one generic warning carrying the reader's message.


def _extract_with_fake_reader(monkeypatch, *, sample_rate, samples=None, error=None):
    """Run the extractor with the stream probe answering ``sample_rate`` and the shared reader
    returning ``samples`` or raising ``error``; return ``(result, logs, reader_calls)``."""
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.audio as mod

    reader_calls: list[tuple] = []

    def fake_decode(path, *, target_sr, mono):
        reader_calls.append((path, target_sr, mono))
        if error is not None:
            raise error
        return samples

    monkeypatch.setattr(mod, "_audio_track_sample_rate", lambda path: sample_rate)
    monkeypatch.setattr(mod, "decode_audio_container", fake_decode)
    logs = {"info": [], "warning": [], "no_audio_track": []}
    result = mod.extract_reference_video_audio(
        "drv.mp4",
        log_info=lambda msg, *args: logs["info"].append(msg % args),
        log_warning=lambda msg, *args: logs["warning"].append(msg % args),
        log_no_audio_track=lambda msg, *args: logs["no_audio_track"].append(msg % args),
    )
    return result, logs, reader_calls


def test_extracted_audio_is_batched_channels_first_at_the_native_rate(monkeypatch):
    # The reader hands back [L, C] at the requested rate.
    samples = np.linspace(-1.0, 1.0, 2 * 480, dtype=np.float32).reshape(480, 2)
    (audio, sample_rate), logs, reader_calls = _extract_with_fake_reader(
        monkeypatch, sample_rate=48000, samples=samples
    )
    assert reader_calls == [("drv.mp4", 48000, False)]
    assert sample_rate == 48000
    assert audio.shape == (1, 2, 480) and audio.dtype == torch.float32
    assert audio.is_contiguous()
    torch.testing.assert_close(audio[0], torch.from_numpy(samples).T)
    assert logs["warning"] == [] and logs["no_audio_track"] == []
    assert logs["info"] == [
        "Wan-Animate-2: extracted reference-video audio (1, 2, 480) @ 48000 Hz."
    ]


def test_missing_audio_track_goes_to_the_no_audio_track_logger(monkeypatch):
    result, logs, reader_calls = _extract_with_fake_reader(
        monkeypatch, sample_rate=None
    )
    assert result == (None, None)
    assert reader_calls == []
    assert logs["warning"] == [] and logs["info"] == []
    assert logs["no_audio_track"] == [
        "Wan-Animate-2: reference video drv.mp4 has no audio track; the output is silent."
    ]


def test_reader_failures_warn_with_the_reader_message(monkeypatch):
    result, logs, _ = _extract_with_fake_reader(
        monkeypatch,
        sample_rate=44100,
        error=ValueError("no decodable audio stream was found in the media container"),
    )
    assert result == (None, None)
    assert logs["info"] == [] and logs["no_audio_track"] == []
    assert logs["warning"] == [
        "Wan-Animate-2: cannot decode the audio track of reference video drv.mp4 (no "
        "decodable audio stream was found in the media container); the output is silent."
    ]


def test_missing_reader_dependency_warns_instead_of_raising(monkeypatch):
    import sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.audio as mod

    def probe(path):
        raise ImportError("No module named 'av'")

    monkeypatch.setattr(mod, "_audio_track_sample_rate", probe)
    warnings: list[str] = []
    result = mod.extract_reference_video_audio(
        "drv.mp4",
        log_info=lambda msg, *args: pytest.fail("no info line expected"),
        log_warning=lambda msg, *args: warnings.append(msg % args),
    )
    assert result == (None, None)
    assert warnings == [
        "Wan-Animate-2: cannot decode the audio track of reference video drv.mp4 (No module "
        "named 'av'); the output is silent."
    ]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"image_path": ["a.png", "b.png"]}, "exactly one 'image_path'"),
        ({"video_path": []}, "exactly one 'video_path'"),
        ({"seed": [1, 2]}, "exactly one 'seed'"),
    ],
)
def test_extract_inputs_rejects_batched_single_value_fields(overrides, message):
    # One video per request: a batched reference image, driving video or seed list must be
    # refused here, not fail inside the stage with an index error or run the wrong seed.
    params = Wan_Animate_2_14B_SamplingParam(
        prompt="p", image_path="ref.png", video_path="drv.mp4"
    )
    batch = Req(sampling_params=params, **overrides)
    with pytest.raises(ValueError, match=message):
        _WanAnimate2Inputs.from_request(batch)


# Scheduler: loaded from the checkpoint's scheduler/scheduler_config.json through the shared
# SchedulerLoader like the other components. The stages pass the official sigma grid to
# set_timesteps explicitly, so the loaded configuration must denoise exactly like a
# DPMSolverMultistepScheduler built from the flow-matching settings alone.

# Wan2.2-Animate-2-14B-Diffusers/scheduler/scheduler_config.json without _class_name, which
# the loader pops before constructing the class it names.
_CHECKPOINT_SCHEDULER_CONFIG = {
    "_diffusers_version": "0.40.0.dev0",
    "algorithm_type": "dpmsolver++",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "disable_corrector": [],
    "dynamic_thresholding_ratio": 0.995,
    "euler_at_final": False,
    "final_sigmas_type": "zero",
    "flow_shift": 5.0,
    "lambda_min_clipped": -math.inf,
    "lower_order_final": True,
    "num_train_timesteps": 1000,
    "predict_x0": True,
    "prediction_type": "flow_prediction",
    "rescale_betas_zero_snr": False,
    "sample_max_value": 1.0,
    "solver_order": 2,
    "solver_p": None,
    "solver_type": "midpoint",
    "steps_offset": 0,
    "thresholding": False,
    "time_shift_type": "exponential",
    "timestep_spacing": "linspace",
    "trained_betas": None,
    "use_beta_sigmas": False,
    "use_dynamic_shifting": False,
    "use_exponential_sigmas": False,
    "use_flow_sigmas": True,
    "use_karras_sigmas": False,
    "use_lu_lambdas": False,
    "variance_type": None,
}

_FLOW_MATCHING_SCHEDULER_KWARGS = {
    "num_train_timesteps": 1000,
    "prediction_type": "flow_prediction",
    "use_flow_sigmas": True,
    "flow_shift": 5.0,
}


def test_scheduler_is_a_loaded_component():
    assert "scheduler" in WanAnimate2Pipeline._required_config_modules
    assert len(WanAnimate2Pipeline._required_config_modules) == 7
    # No pipeline-side construction: the base class hook is inherited unchanged.
    assert "initialize_pipeline" not in vars(WanAnimate2Pipeline)


def test_loader_keeps_the_checkpoint_scheduler_settings():
    kwargs = _supported_init_kwargs(
        DPMSolverMultistepScheduler, dict(_CHECKPOINT_SCHEDULER_CONFIG)
    )
    assert kwargs["beta_schedule"] == "linear"
    assert kwargs["use_flow_sigmas"] is True
    assert kwargs["prediction_type"] == "flow_prediction"
    assert kwargs["flow_shift"] == 5.0
    scheduler = DPMSolverMultistepScheduler(**kwargs)
    config = scheduler._inner.config
    assert config.algorithm_type == "dpmsolver++"
    assert config.solver_order == 2
    assert config.beta_schedule == "linear"
    assert config.use_flow_sigmas is True
    assert config.prediction_type == "flow_prediction"
    assert config.flow_shift == 5.0


@pytest.mark.parametrize("num_inference_steps", [4, 40])
def test_checkpoint_scheduler_denoises_like_the_flow_matching_settings(
    num_inference_steps,
):
    # beta_schedule is the only setting that differs between the two configurations
    # (linear vs the wrapper's scaled_linear default); it only shapes the beta-derived
    # tables, which the explicit sigma grid path never reads.
    from_checkpoint = DPMSolverMultistepScheduler(**_CHECKPOINT_SCHEDULER_CONFIG)
    from_settings = DPMSolverMultistepScheduler(**_FLOW_MATCHING_SCHEDULER_KWARGS)
    sigmas = get_sampling_sigmas(num_inference_steps, 5.0)
    from_checkpoint.set_timesteps(sigmas=sigmas)
    from_settings.set_timesteps(sigmas=sigmas)
    assert torch.equal(from_checkpoint.timesteps, from_settings.timesteps)
    assert torch.equal(from_checkpoint.sigmas, from_settings.sigmas)
    assert len(from_checkpoint.timesteps) == num_inference_steps

    generator = torch.Generator().manual_seed(0)
    sample = torch.randn(1, 16, 2, 4, 4, generator=generator)
    model_output = torch.randn(1, 16, 2, 4, 4, generator=generator)
    t = from_checkpoint.timesteps[0]
    out_checkpoint = from_checkpoint.step(model_output, t, sample).prev_sample
    out_settings = from_settings.step(model_output, t, sample).prev_sample
    assert torch.equal(out_checkpoint, out_settings)


# Official sampler sigma grid


def test_official_sampler_shift_and_sigma_grid():
    # Both literals come from the official sampler (sample_shift 5.0 and its
    # shift * s / (1 + (shift - 1) * s) grid); the config's shift must feed that grid.
    config = Wan_Animate_2_14B_Config()
    assert config.flow_shift == 5.0
    sigmas = get_sampling_sigmas(40, config.flow_shift)
    assert len(sigmas) == 40
    assert sigmas[0] == 1.0
    np.testing.assert_array_equal(
        sigmas[:3], [1.0, 0.9948979591836734, 0.9895833333333334]
    )


# WanAnimate2VaeAdapter wraps the sglang AutoencoderKLWan, whose encode returns the
# DiagonalGaussianDistribution itself and whose decode returns a plain tensor (no diffusers
# AutoencoderKLOutput / DecoderOutput wrappers).
def test_vae_adapter_uses_the_sglang_vae_return_types():
    class _StubWanVae(torch.nn.Module):
        latents_mean = [0.5, -0.5]
        latents_std = [2.0, 4.0]

        def encode(self, x):
            # mean = x's per-channel mean, logvar = 0, in the [B, 2C, T, H, W] layout.
            mean = x.mean(dim=(2, 3, 4), keepdim=True).expand(-1, -1, 1, 1, 1)
            return DiagonalGaussianDistribution(
                torch.cat([mean, torch.zeros_like(mean)], dim=1)
            )

        def decode(self, z):
            return z * 3.0  # a plain tensor, as the in-tree VAE returns

    adapter = WanAnimate2VaeAdapter(_StubWanVae())

    pixels = torch.zeros(2, 1, 4, 4)
    pixels[0] += 2.5  # channel 0 mean 2.5, channel 1 mean 0.0
    (latent,) = adapter.encode([pixels])
    # (mode - mean) / std per channel: (2.5 - 0.5) / 2 = 1.0, (0.0 + 0.5) / 4 = 0.125
    torch.testing.assert_close(latent.flatten(), torch.tensor([1.0, 0.125]))
    assert latent.dtype == torch.float32

    decoded = adapter.decode([latent])
    # denormalize (z * std + mean) = [2.5, 0.0], then the stub's x3, clamped to [-1, 1]
    torch.testing.assert_close(decoded.flatten(), torch.tensor([1.0, 0.0]))
    assert decoded.shape[0] == 1 and decoded.dtype == torch.float32


# Deployment topology


def test_disaggregated_roles_are_rejected_at_argument_validation():
    """The pipeline's inter-stage state lives in batch.extra as GPU tensors, which the
    disaggregation transport cannot carry; a split-role launch must fail before any weights
    load instead of at the first request."""
    for role in (RoleType.ENCODER, RoleType.DENOISER, RoleType.DECODER):
        args = SimpleNamespace(
            pipeline_config=Wan_Animate_2_14B_Config(), disagg_role=role
        )
        with pytest.raises(ValueError, match="only supports monolithic"):
            ServerArgs._validate_disagg_capability(args)
    ServerArgs._validate_disagg_capability(
        SimpleNamespace(
            pipeline_config=Wan_Animate_2_14B_Config(), disagg_role=RoleType.MONOLITHIC
        )
    )


# Synthetic server warmup. The generic warmup builder supplies only ``image_path``; a
# Wan-Animate-2 warmup request must also carry a decodable driving video that the clip
# schedule accepts as one clip, or ``--warmup-mode server`` (forced on by
# ``--enable-torch-compile``) can never finish. Requests are built exactly as
# ``run_async_client_warmup`` builds them and run through the before-denoising stage with
# stand-in encoders.

_BEFORE_DENOISING = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages"
    ".wan_animate_2.before_denoising"
)


def _server_args(input_save_path: str) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline_class_name="WanAnimate2Pipeline",
        pipeline_config=Wan_Animate_2_14B_Config(),
        component_precisions={},
        enable_layerwise_nvtx_marker=False,
        comfyui_mode=False,
        warmup_steps=1,
        warmup_num_frames=None,
        warmup_resolutions=None,
        enable_torch_compile=False,
        enable_breakable_cuda_graph=False,
        enable_cfg_parallel=False,
        num_gpus=1,
        backend="sglang",
        # anything but "auto": keeps the auto-residency full-shape probe out of the build
        performance_mode="balanced",
        input_save_path=input_save_path,
        is_arg_explicitly_set=lambda name: False,
    )


def _build_server_warmup_reqs(server_args: SimpleNamespace) -> list[Req]:
    warmup_image = prepare_warmup_image_path(server_args)
    with patch(
        "sglang.multimodal_gen.runtime.warmup_request_builder.get_model_sampling_defaults",
        return_value=Wan_Animate_2_14B_SamplingParam(),
    ):
        return build_warmup_reqs(
            server_args,
            warmup_resolutions=None,
            warmup_input_path=warmup_image,
            server_based_warmup=True,
        )


def _single_clip_schedule(inputs: _WanAnimate2Inputs) -> list[int]:
    """Per-clip frame counts the stage would schedule for the request's driving video."""
    frames = read_reference_video_frames(inputs.reference_video_path, inputs.fps)
    assert frames.ndim == 4 and frames.shape[-1] == 3 and len(frames) > 0
    padded = zigzag_padding(list(frames), get_padding_len(len(frames), inputs.clip_len))
    return [
        clip.num_frames
        for clip in build_schedule(
            padded, inputs.clip_len, inputs.num_frames_conditioning
        )
    ]


def test_reference_video_url_uses_native_media_loading(tmp_path):
    path = tmp_path / "reference.mp4"
    write_reference_video(str(path), np.full((5, 16, 16, 3), 128, dtype=np.uint8), 16)
    expected = read_reference_video_frames(str(path), 8)
    with patch(
        "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages."
        "wan_animate_2.preprocess.get_video_bytes",
        return_value=path.read_bytes(),
    ) as download:
        actual = read_reference_video_frames("https://example.com/reference.mp4", 8)
    download.assert_called_once_with("https://example.com/reference.mp4")
    np.testing.assert_array_equal(actual, expected)


class _StubVae:
    """encode: [3, T, H, W] -> [16, (T-1)//4+1, H//8, W//8] fp32 zeros; the VAE module itself
    is only handed to the residency manager."""

    def __init__(self) -> None:
        self.vae = torch.nn.Identity()

    def encode(self, videos: list[torch.Tensor]) -> list[torch.Tensor]:
        return [
            torch.zeros(16, (v.shape[1] - 1) // 4 + 1, v.shape[2] // 8, v.shape[3] // 8)
            for v in videos
        ]


class _StubImageEncoder:
    def __init__(self) -> None:
        self.model = torch.nn.Identity()

    def visual(self, videos: list[torch.Tensor]) -> torch.Tensor:
        return torch.zeros(len(videos), 257, 1280)


def _before_denoising_stage(pipeline_config) -> WanAnimate2BeforeDenoisingStage:
    stage = WanAnimate2BeforeDenoisingStage(
        vae=_StubVae(),
        image_encoder=_StubImageEncoder(),
        text_encoder=torch.nn.Identity(),
        tokenizer=object(),
        pipeline_config=pipeline_config,
        scheduler=DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=pipeline_config.flow_shift,
        ),
    )
    stage.encode_text = lambda text, *args, **kwargs: (
        [torch.full((max(1, len(text)), 4096), 0.5)],
        [],
    )
    return stage


def test_server_warmup_request_denoises_one_clip_of_a_synthetic_driving_video(tmp_path):
    server_args = _server_args(str(tmp_path))
    reqs = _build_server_warmup_reqs(server_args)
    assert len(reqs) == 1

    inputs = _WanAnimate2Inputs.from_request(reqs[0])
    assert os.path.isfile(inputs.reference_image_path)
    assert os.path.isfile(inputs.reference_video_path)
    assert os.path.dirname(inputs.reference_video_path) == str(tmp_path)
    assert inputs.num_inference_steps == SERVER_WARMUP_VIDEO_STEPS
    # Warm up at the default production clip length so the compiled attention is
    # specialized for the geometry real requests use (the builder alone asks for 17 frames).
    assert inputs.clip_len == Wan_Animate_2_14B_SamplingParam().clip_len
    assert _single_clip_schedule(inputs) == [inputs.clip_len]
    # The synthetic driving video is silent, so warmup must not decode audio for it.
    assert inputs.enable_audio is False


def test_warmup_requests_with_different_clip_lengths_keep_separate_driving_videos(
    tmp_path,
):
    # All warmup requests are built before the first one runs, so the bounded request and
    # the full-shape probe must not overwrite each other's video.
    server_args = _server_args(str(tmp_path))
    warmup_image = prepare_warmup_image_path(server_args)
    inputs = []
    for num_frames in (17, 45):
        params = Wan_Animate_2_14B_SamplingParam()
        req = Req(
            sampling_params=params,
            prompt="warmup",
            image_path=[warmup_image],
            width=640,
            height=800,
            num_frames=num_frames,
        )
        params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        inputs.append(_WanAnimate2Inputs.from_request(req))

    default_len, longer = inputs
    assert default_len.reference_video_path != longer.reference_video_path
    assert default_len.clip_len == Wan_Animate_2_14B_SamplingParam().clip_len == 37
    assert longer.clip_len == 45
    assert _single_clip_schedule(default_len) == [default_len.clip_len]
    assert _single_clip_schedule(longer) == [longer.clip_len]


_SHARED_VIDEO_WRITER = (
    "sglang.multimodal_gen.runtime.entrypoints.utils.post_process_sample"
)


def test_write_reference_video_goes_through_the_shared_video_writer(tmp_path):
    # The synthetic warmup video is encoded by the writer request outputs go through, with
    # the frames handed over unchanged, so it decodes wherever request outputs decode.
    frames = np.full((9, 64, 64, 3), 200, dtype=np.uint8)
    path = os.path.join(str(tmp_path), "warmup_driving_video_9f_30fps.mp4")

    with patch(_SHARED_VIDEO_WRITER) as writer:
        write_reference_video(path, frames, fps=30)

    writer.assert_called_once()
    (sample, data_type, fps), kwargs = writer.call_args
    assert sample is frames
    assert data_type == DataType.VIDEO
    assert fps == 30
    assert kwargs == {"save_file_path": path}


def test_write_reference_video_names_the_shared_writer_when_it_fails(tmp_path):
    path = os.path.join(str(tmp_path), "warmup_driving_video_9f_30fps.mp4")
    frames = np.zeros((9, 64, 64, 3), dtype=np.uint8)

    with (
        patch(_SHARED_VIDEO_WRITER, side_effect=OSError("no video backend")),
        pytest.raises(RuntimeError, match="shared video writer") as excinfo,
    ):
        write_reference_video(path, frames, fps=30)

    assert path in str(excinfo.value)
    assert "no video backend" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


@pytest.mark.parametrize("clip_len", [17, 37])
def test_before_denoising_stage_populates_the_standard_denoising_inputs(
    tmp_path, clip_len
):
    """After the before-denoising stage the Req must carry the fields the shared denoising
    contract requires, consistent with what the clip loop will use: clip 0's initial noise
    (the seed's first draw, which the loop consumes instead of drawing again) and the
    official sigma grid."""
    server_args = _server_args(str(tmp_path))
    (req,) = _build_server_warmup_reqs(server_args)
    req.clip_len = clip_len
    set_global_server_args(server_args)
    stage = _before_denoising_stage(server_args.pipeline_config)

    with (
        patch(f"{_BEFORE_DENOISING}.get_sp_world_size", return_value=1),
        patch(
            f"{_BEFORE_DENOISING}.get_local_torch_device",
            return_value=torch.device("cpu"),
        ),
    ):
        batch = stage(req, server_args)

    request_state = request_state_from_batch(batch)
    assert all(
        clip.frame_start_index + clip.num_frames_conditioning_for_clip
        < request_state.num_reference_video_frames
        for clip in request_state.schedule
    )
    assert batch.generator is request_state.generator
    assert batch.prompt_embeds == [request_state.prompt_embeddings]
    assert batch.negative_prompt_embeds == [request_state.negative_prompt_embeddings]
    assert batch.image_embeds == [request_state.reference_image_embeddings]
    assert len(batch.sigmas) == batch.num_inference_steps == len(batch.timesteps)
    assert batch.sigmas[0] == 1.0
    assert torch.equal(
        batch.timesteps, torch.tensor([int(s * 1000) for s in batch.sigmas])
    )

    clip_0 = build_clip_conditioning(
        request_state,
        request_state.schedule[0],
        None,
        vae_encoder=stage.vae_encoder,
        image_embedder=stage.image_embedder,
    )
    assert torch.equal(batch.latents, clip_0.init_noise)
    assert tuple(batch.raw_latent_shape) == tuple(clip_0.init_noise.shape)

    # One-draw-ahead invariant: clip 0 is the seed's first draw of its shape, and the
    # request generator now stands exactly one draw further, so the next clip gets the
    # seed's second draw, as it did when the loop drew clip 0 itself.
    reference = torch.Generator(device=request_state.device).manual_seed(
        request_state.inputs.seed
    )
    shape = tuple(batch.latents.shape)
    first = torch.randn(*shape, device=request_state.device, generator=reference)
    second = torch.randn(*shape, device=request_state.device, generator=reference)
    assert torch.equal(first, batch.latents)
    next_clip_noise = torch.randn(
        *shape, device=request_state.device, generator=request_state.generator
    )
    assert torch.equal(next_clip_noise, second)
    assert not torch.equal(next_clip_noise, batch.latents)

    denoising_verification = WanAnimate2DenoisingStage.verify_input(
        None, batch, server_args
    )
    assert denoising_verification.is_valid(), (
        denoising_verification.get_failure_summary()
    )
    # A request that skipped the before-denoising stage must not pass the same check.
    bare = WanAnimate2DenoisingStage.verify_input(
        None, Req(sampling_params=Wan_Animate_2_14B_SamplingParam()), server_args
    )
    assert not bare.is_valid()
