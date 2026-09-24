# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.flux3 import (
    Flux3ArchConfig,
    Flux3DiTConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
    Flux3ActionPipelineConfig,
    _validate_parallelism,
    flux3_action_variant_subfolder,
    is_flux3_action_package,
    read_flux3_action_config,
)
from sglang.multimodal_gen.configs.sample.flux3_action import Flux3ActionSamplingParams
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_metadata,
    build_action_sampling_params,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.flux3_action import (
    _cfg_parallel_policy,
    cosmos_unipc,
    denormalize,
    normalize,
    pack_action,
    pack_video,
    parse_observation,
    targets_to_actions,
    text_ids,
)

DROID_CONFIG = {
    "action_dim": 8,
    "action_modality": "action_prediction_droid",
    "camera_layout": "droid",
    "camera_keys": ["images.wrist", "images.left", "images.right"],
    "canvas_hw": [544, 736],
    "chunk_size": 32,
    "n_action_steps": 32,
    "fps": 15.0,
    "action_scale": 2.0,
    "gripper_flip_dims": [-1],
    "action_parameterization": "absolute",
    "absolute_action_dims": [],
    "action_normalization": None,
    "state_normalization": None,
    "normalization_clip": 6.0,
    "video_vae_id": "black-forest-labs/flux-3-action-base:video_vae.safetensors@rev",
    "text_encoder_id": "black-forest-labs/flux-3-action-base:text_encoder@rev",
    "dit_config": {},
    "content_streams": ["video", "video_cond"],
    "torch_dtype": "bfloat16",
    "quantization": None,
    "inference_profile": "default",
    "sampler": "cosmos_unipc",
    "num_inference_steps": 4,
    "guidance_scale": 4.0,
    "guidance_scale_action": 1.0,
    "sampler_shift": 5.0,
    "inference_seed": 0,
}


def _droid_config() -> Flux3ActionPipelineConfig:
    config = Flux3ActionPipelineConfig()
    config.load_policy_config(dict(DROID_CONFIG))
    return config


def _write_package(root) -> None:
    (root / "config.native.json").write_text(json.dumps(DROID_CONFIG))
    (root / "manifest.json").write_text(json.dumps({"kind": "policy_export"}))


# ---------------------------------------------------------------- config / registry
def test_policy_config_adopts_droid_recipe():
    config = _droid_config()
    assert config.image_keys == ("wrist", "left", "right")
    assert config.latent_hw == (17, 20)
    assert config.default_num_inference_steps == 4
    assert (config.guidance_scale, config.guidance_scale_action) == (4.0, 1.0)
    arch = config.dit_config.arch_config
    assert arch.in_channels == {
        "video": 96,
        "video_cond": 96,
        "action_prediction_droid": 8,
        "action_prediction_droid_cond": 8,
    }
    assert list(arch.sequence) == [
        "x_video",
        "x_video_cond",
        "x_action_prediction_droid",
        "x_action_prediction_droid_cond",
    ]


def test_policy_config_rejects_unsupported_profiles():
    config = Flux3ActionPipelineConfig()
    with pytest.raises(NotImplementedError):
        config.load_policy_config({**DROID_CONFIG, "inference_profile": "history"})


def _parallel_args(**overrides):
    args = dict(
        num_gpus=2,
        enable_cfg_parallel=True,
        cfg_parallel_degree=2,
        tp_size=1,
        sp_degree=1,
    )
    return SimpleNamespace(**{**args, **overrides})


def test_multi_gpu_serving_requires_two_way_cfg_parallel():
    """Layouts the DiT does not shard (TP, SP) must be refused, not run replicated."""
    _validate_parallelism(_parallel_args(num_gpus=1, enable_cfg_parallel=False))
    _validate_parallelism(_parallel_args())
    for overrides in (
        dict(enable_cfg_parallel=False, cfg_parallel_degree=1, sp_degree=2),
        dict(enable_cfg_parallel=False, cfg_parallel_degree=1, tp_size=2),
        dict(num_gpus=4, sp_degree=2),
        dict(num_gpus=4, tp_size=2),
    ):
        with pytest.raises(NotImplementedError):
            _validate_parallelism(_parallel_args(**overrides))


def test_cfg_parallel_branches_are_conditional_then_unconditional():
    """The combine reads ``cond, uncond = preds``; a swapped order inverts guidance."""
    cond, uncond = object(), object()
    policy = _cfg_parallel_policy([cond, uncond], _parallel_args())
    assert [b.kwargs["context"] for b in policy.branches] == [cond, uncond]
    assert [b.is_conditional for b in policy.branches] == [True, False]
    assert policy.parallel_uses_serial_arithmetic
    assert _cfg_parallel_policy([cond], _parallel_args()) is None
    assert (
        _cfg_parallel_policy([cond, uncond], _parallel_args(enable_cfg_parallel=False))
        is None
    )


def test_variant_subfolders():
    assert flux3_action_variant_subfolder(None) == ""
    assert flux3_action_variant_subfolder("base") == ""
    assert flux3_action_variant_subfolder("gd-fp8r") == "variants/gd-fp8r"
    with pytest.raises(ValueError):
        flux3_action_variant_subfolder("nope")


def test_local_package_detection_and_registry(tmp_path):
    from sglang.multimodal_gen.registry import (
        get_model_info,
        get_non_diffusers_pipeline_name,
    )

    assert not is_flux3_action_package(str(tmp_path))
    _write_package(tmp_path)
    assert is_flux3_action_package(str(tmp_path))
    assert (
        read_flux3_action_config(tmp_path)["action_modality"]
        == "action_prediction_droid"
    )
    assert get_non_diffusers_pipeline_name(str(tmp_path)) == "Flux3ActionPipeline"
    for path in (str(tmp_path), "black-forest-labs/flux-3-action-droid"):
        get_model_info.cache_clear()
        info = get_model_info(path)
        assert info.pipeline_cls.__name__ == "Flux3ActionPipeline"
        assert info.sampling_param_cls is Flux3ActionSamplingParams
    get_model_info.cache_clear()


# ---------------------------------------------------------------- protocol
def _server_args(config: Flux3ActionPipelineConfig) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=None,
        model_path="black-forest-labs/flux-3-action-droid",
        served_model_name="flux3-action",
        output_path=None,
        comfyui_mode=False,
        backend=None,
        pipeline_class_name=None,
        pipeline_config=config,
    )


def test_action_request_builds_flux3_sampling_params():
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    payload = {
        "input": {
            "task": "pour the cup",
            "observation": {
                "images": {"wrist": image, "left": image, "right": image},
                "state": [0.0] * 8,
            },
        },
        "parameters": {"guidance_scale": 2.0, "seed": 7},
    }
    params = build_action_sampling_params(payload, _server_args(_droid_config()))
    assert isinstance(params, Flux3ActionSamplingParams)
    assert params.prompt == "pour the cup"
    assert params.num_inference_steps == 4
    assert params.guidance_scale == 2.0 and params.guidance_scale_action is None
    assert params.seed == 7
    extra = params.build_request_extra()["vla"]
    assert set(extra["observation"]["images"]) == {"wrist", "left", "right"}
    assert extra["options"]["guidance_scale"] == 2.0


def test_action_metadata_reports_policy_recipe():
    metadata = action_metadata(_server_args(_droid_config()))
    assert metadata["policy_family"] == "flux3_action"
    assert metadata["input"]["image_keys"] == ["wrist", "left", "right"]
    assert metadata["output"]["action_horizon"] == 32
    assert metadata["defaults"]["guidance_scale"] == 4.0


def test_sampling_params_reject_unsupported_requests():
    """Unsupported request fields must fail loudly instead of being ignored."""
    for kwargs in (
        {"guidance_scale": -1.0},
        {"guidance_scale": "4"},
        {"seed": [1, 2]},
        {"num_outputs_per_prompt": 2},
        {"action_horizon": 0},
    ):
        with pytest.raises(ValueError):
            Flux3ActionSamplingParams(**kwargs)


def test_guidance_resolution_follows_video_scale_when_action_is_unset():
    """Reference rule: an unset action scale follows the (possibly overridden) video scale."""
    config = Flux3ActionPipelineConfig()
    config.load_policy_config({**DROID_CONFIG, "guidance_scale_action": None})
    assert config.resolve_guidance(None, None) == {
        "video": 4.0,
        "action_prediction_droid": 4.0,
    }
    assert config.resolve_guidance(2.0, None) == {
        "video": 2.0,
        "action_prediction_droid": 2.0,
    }
    droid = _droid_config()  # explicit action scale in the package
    assert droid.resolve_guidance(2.0, None) == {
        "video": 2.0,
        "action_prediction_droid": 1.0,
    }


def test_reference_dit_config_fields_are_translated():
    """Exports carry the reference JointSingleSeqParams names and the full trunk's streams."""
    config = Flux3ActionPipelineConfig()
    config.load_policy_config(
        {
            **DROID_CONFIG,
            "dit_config": {
                "num_heads": 24,
                "depth": 5,
                "attn_mode": "torch",
                "in_channels": {"video": 96, "video_cond": 96, "image": 128},
            },
        }
    )
    arch = config.dit_config.arch_config
    assert arch.num_attention_heads == 24
    assert "image" not in arch.in_channels
    with pytest.raises(NotImplementedError):
        config.load_policy_config(
            {**DROID_CONFIG, "dit_config": {"depth_late_blocks": 2}}
        )


def test_manifest_checksums_are_verified(tmp_path):
    import hashlib

    from sglang.multimodal_gen.configs.pipeline_configs.flux3_action import (
        verify_flux3_action_manifest,
    )

    config_bytes = json.dumps(DROID_CONFIG).encode()
    (tmp_path / "config.native.json").write_bytes(config_bytes)
    manifest = {
        "kind": "policy_export",
        "sha256": {"config.native.json": hashlib.sha256(config_bytes).hexdigest()},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verify_flux3_action_manifest(tmp_path, include_weights=False)
    (tmp_path / "config.native.json").write_bytes(config_bytes + b" ")
    with pytest.raises(ValueError, match="checksum"):
        verify_flux3_action_manifest(tmp_path, include_weights=False)


# ---------------------------------------------------------------- observation / action space
def _views(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
        for name in ("wrist", "left", "right")
    }


def test_droid_canvas_from_cameras():
    config = _droid_config()
    views = _views()
    obs = parse_observation(
        {
            "images": {f"images.{k}": v for k, v in views.items()},
            "state": np.zeros(8),
            "prompt": "x",
        },
        config,
    )
    assert tuple(obs.canvas.shape) == (3, 544, 736)
    assert obs.canvas.min() >= -1.0 and obs.canvas.max() <= 1.0
    wrist = torch.from_numpy(views["wrist"]).permute(2, 0, 1).float() / 255 * 2 - 1
    torch.testing.assert_close(obs.canvas[:, :360, :640], wrist)
    # reflect padding to the right of the 640-wide composite
    torch.testing.assert_close(obs.canvas[:, :, 640], obs.canvas[:, :, 638])


def test_droid_composite_matches_three_camera_canvas():
    """A client-built 540x640 composite and the three cameras give the same canvas."""
    config = _droid_config()
    views = _views(1)
    from_cameras = parse_observation({"images": views, "state": np.zeros(8)}, config)
    composite = (from_cameras.canvas[:, :540, :640] + 1) / 2
    from_composite = parse_observation(
        {"images": {"composite": composite}, "state": np.zeros(8)}, config
    )
    torch.testing.assert_close(from_composite.canvas, from_cameras.canvas)


def test_lerobot_camera_names_and_float_images():
    config = _droid_config()
    views = _views()
    lerobot = {
        "observation.images.wrist_image_left": views["wrist"],
        "observation.images.exterior_image_1_left": views["left"],
        "observation.images.exterior_image_2_left": views["right"],
    }
    by_alias = parse_observation({**lerobot, "observation.state": np.zeros(8)}, config)
    floats = {k: v.astype(np.float32) / 255 for k, v in views.items()}
    by_float = parse_observation({"images": floats, "state": np.zeros(8)}, config)
    torch.testing.assert_close(by_alias.canvas, by_float.canvas)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        pixels = {k: v.astype(np.float32) for k, v in views.items()}
        parse_observation({"images": pixels, "state": np.zeros(8)}, config)


def test_json_decoded_pixel_lists_are_accepted():
    """Cookbook JSON requests send image.tolist(); the lists decode to int64 arrays."""
    config = _droid_config()
    views = _views(2)
    as_json = {k: np.asarray(v.tolist()) for k, v in views.items()}
    assert next(iter(as_json.values())).dtype == np.int64
    from_json = parse_observation({"images": as_json, "state": [0.0] * 8}, config)
    from_uint8 = parse_observation({"images": views, "state": np.zeros(8)}, config)
    assert torch.equal(from_json.canvas, from_uint8.canvas)
    with pytest.raises(ValueError, match="0, 255"):
        parse_observation(
            {
                "images": {**as_json, "wrist": as_json["wrist"] + 256},
                "state": [0.0] * 8,
            },
            config,
        )


def test_missing_camera_is_reported():
    views = _views()
    del views["left"]
    with pytest.raises(KeyError, match="left"):
        parse_observation({"images": views, "state": np.zeros(8)}, _droid_config())


def test_action_space_roundtrip_and_joint_delta():
    stats = {"q01": [-1.0, 0.0, 2.0], "q99": [1.0, 4.0, 2.0]}
    x = torch.tensor([[0.5, 1.0, 2.0]])
    roundtrip = denormalize(normalize(x, stats=stats, clip=6.0), stats=stats)
    torch.testing.assert_close(roundtrip, x)

    config = _droid_config()
    config.action_parameterization = "joint_delta"
    config.gripper_flip_dims = (-1,)
    config.absolute_action_dims = (2,)
    state = torch.tensor([1.0, 2.0, 0.25])
    deltas = torch.tensor([[0.1, 0.0, 0.2], [0.1, -1.0, 0.4]])
    actions = targets_to_actions(deltas, state=state, config=config)
    expected = torch.tensor([[1.1, 2.0, 0.8], [1.2, 1.0, 0.6]])
    torch.testing.assert_close(actions, expected)


def test_position_ids_follow_the_10ms_clock():
    _, ids = pack_video(torch.zeros(1, 96, 2, 2, 3), first_frame=1, fps=15.0)
    assert ids.shape == (1, 12, 4)
    assert ids[0, 0].tolist() == [26, 0, 0, 0]  # frame 1 at 4 / 15 s
    assert ids[0, -1].tolist() == [53, 1, 2, 0]
    seconds = (torch.arange(3).float() + 1) / 15.0
    tokens, ids = pack_action(torch.zeros(1, 8, 3), seconds=seconds)
    assert tokens.shape == (1, 3, 8)
    assert ids[0, :, 0].tolist() == [6, 13, 20]
    assert text_ids(3)[0, :, 3].tolist() == [0, 1, 2]


# ---------------------------------------------------------------- samplers
def _unipc_scheduler() -> FlowUniPCMultistepScheduler:
    # Same configuration as Flux3ActionPipeline.load_modules.
    return FlowUniPCMultistepScheduler(
        solver_order=2,
        solver_type="bh2",
        predict_x0=True,
        lower_order_final=True,
        final_sigmas_type="zero",
        shift=1.0,
    )


def test_cosmos_unipc_feeds_reference_ticks_and_reuses_the_scheduler():
    """DROID recipe (4 steps, shift 5): the model sees ticks 999, 937, 833, 624,
    and the pipeline's scheduler module is not consumed by a request."""
    scheduler = _unipc_scheduler()
    for _ in range(2):
        seen = []
        cosmos_unipc(
            {"a": torch.zeros(1, 3, 2)},
            lambda samples, t: (
                seen.append(round(t * 1000)),
                {"a": torch.zeros(1, 3, 2)},
            )[1],
            scheduler=scheduler,
            n_steps=4,
            shift=5.0,
        )
        assert seen == [999, 937, 833, 624]


def test_cosmos_unipc_recovers_x0_on_a_straight_flow():
    """Exact velocities on a straight flow must land on x0 (solver bookkeeping is consistent)."""
    generator = torch.Generator().manual_seed(0)
    x0 = {
        "a": torch.randn(1, 2, 5, generator=generator),
        "b": torch.randn(1, 3, 8, generator=generator),
    }
    eps = {k: torch.randn(v.shape, generator=generator) for k, v in x0.items()}
    # UniPC starts at the shifted 0.999: 5 * 0.999 / (1 + 4 * 0.999)
    sigma_max = 5 * 0.999 / (1 + 4 * 0.999)
    start = {k: sigma_max * eps[k] + (1 - sigma_max) * x0[k] for k in x0}
    result = cosmos_unipc(
        start,
        lambda samples, t: {k: eps[k] - x0[k] for k in samples},
        scheduler=_unipc_scheduler(),
        n_steps=4,
        shift=5.0,
    )
    for k in x0:
        torch.testing.assert_close(result[k], x0[k], atol=1e-4, rtol=0)


# ---------------------------------------------------------------- DiT
def _init_single_process_parallel() -> None:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        ensure_distributed_env_defaults,
    )

    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _tiny_arch() -> Flux3ArchConfig:
    return Flux3ArchConfig(
        hidden_size=64,
        num_attention_heads=4,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(4, 4, 4, 4),
        context_in_dim=32,
        vec_in_dim=16,
    ).with_streams({"act": 3, "act_cond": 3})


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs the CUDA parallel runtime"
)
def test_dit_loads_released_checkpoint_names():
    """Released checkpoints load by name; a renamed module or mapping breaks them."""
    from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
    from sglang.multimodal_gen.runtime.models.dits.flux3 import Flux3Transformer

    _init_single_process_parallel()
    with torch.device("meta"):
        model = Flux3Transformer(Flux3DiTConfig(arch_config=_tiny_arch()))
    params = set(model.state_dict())
    assert all("bias" not in n for n in params)
    # Layerwise offload streams exactly these block lists; a stale name is skipped silently.
    modules = dict(model.named_modules())
    blocks = {name: modules.get(name) for name in model.layer_names}
    assert all(isinstance(m, torch.nn.ModuleList) and len(m) for m in blocks.values())
    assert set(blocks) == {
        "txt_mode_blocks",
        "single_blocks",
        *(
            f"content_mode_blocks.{m}"
            for m in ("video", "video_cond", "act", "act_cond")
        ),
    }
    mapping = get_param_names_mapping(model.param_names_mapping)
    fused = {}
    for name in (
        "dit.emb_in.act.weight",
        "dit.txt_in.weight",
        "dit.time_in.in_layer.weight",
        "dit.vector_in.out_layer.weight",
        "dit.early_stream_modulations.txt.lin.weight",
        "dit.single_stream_modulations.act_cond.lin.weight",
        "dit.content_mode_blocks.act.0.norm.query_norm.scale",
        "dit.txt_mode_blocks.1.attn_out.weight",
        "dit.single_blocks.1.mlp_out.weight",
        "dit.final_layer.act.adaLN_modulation.1.weight",
        "dit.final_layer.video.linear.weight",
        "dit.content_mode_blocks.video.1.q_proj.weight",
        "dit.content_mode_blocks.video.1.k_proj.weight",
        "dit.content_mode_blocks.video.1.v_proj.weight",
        "dit.content_mode_blocks.video.1.mlp_in.weight",
    ):
        target, index, count = mapping(name)
        assert target in params, (name, target)
        if index is not None:
            fused.setdefault(target, set()).add((index, count))
    # q, k, v, mlp_in concatenate in this order into the fused projection
    assert fused == {
        "content_mode_blocks.video.1.qkv_mlp.weight": {(0, 4), (1, 4), (2, 4), (3, 4)}
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA attention")
def test_dit_cached_streams_match_full_forward():
    """The pipeline's per-caption / per-request caching must not change predictions."""
    from sglang.multimodal_gen.runtime.managers.forward_context import (
        set_forward_context,
    )
    from sglang.multimodal_gen.runtime.models.dits.flux3 import Flux3Transformer

    _init_single_process_parallel()
    torch.manual_seed(0)
    model = Flux3Transformer(Flux3DiTConfig(arch_config=_tiny_arch())).cuda().bfloat16()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.05)

    video, video_ids = pack_video(torch.randn(1, 96, 2, 2, 3), first_frame=1, fps=15.0)
    cond, cond_ids = pack_video(torch.randn(1, 96, 1, 2, 3), first_frame=0, fps=15.0)
    act, act_ids = pack_action(
        torch.randn(1, 3, 4), seconds=(torch.arange(4).float() + 1) / 15
    )
    state, state_ids = pack_action(torch.randn(1, 3, 1), seconds=torch.zeros(1))
    ctx = torch.randn(1, 5, 32).cuda()
    t = 0.7
    streams = {
        "x_video": (video, video_ids, t),
        "x_video_cond": (cond, cond_ids, 0.0),
        "x_act": (act, act_ids, t),
        "x_act_cond": (state, state_ids, 0.0),
    }
    kwargs = {}
    for key, (x, ids, ts) in streams.items():
        kwargs[key] = x.cuda()
        kwargs[f"{key}_ids"] = ids.cuda()
        kwargs[f"{key}_timesteps"] = torch.full(x.shape[:2], ts).cuda()
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        full = model(ctx=ctx, ctx_ids=text_ids(5).cuda(), **kwargs)
        context = model.encode_context(ctx, text_ids(5).cuda())
        encoded = [
            model.encode_stream(
                name=key[2:],
                x=x.cuda(),
                ids=ids.cuda(),
                timesteps=torch.full((1,), ts).cuda(),
            )
            for key, (x, ids, ts) in streams.items()
        ]
        cached = model.denoise(
            context=context, streams=encoded, targets=["video", "act"]
        )
    # Same kernels in the same order: the cached path is bit-identical.
    assert torch.equal(cached["video"], full["x_video"])
    assert torch.equal(cached["act"], full["x_act"])
    assert full["x_act_cond"].shape == (1, 1, 3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="needs FP8 scaled_mm",
)
def test_fp8r_checkpoint_loads_fused_rowwise_linears():
    """Native FP8r payloads (E4M3 + per-row scales) must fuse and dequantize consistently."""
    from sglang.multimodal_gen.runtime.models.dits.flux3 import (
        Flux3Fp8RowwiseLinear,
        Flux3Transformer,
        load_fp8r_checkpoint,
        quantize_fp8_rowwise,
    )

    _init_single_process_parallel()
    with torch.device("meta"):
        model = Flux3Transformer(Flux3DiTConfig(arch_config=_tiny_arch()))
    reference = {}
    state = {}
    generator = torch.Generator().manual_seed(0)
    for name, tensor in model.state_dict().items():
        for part in (
            ("q_proj", "k_proj", "v_proj", "mlp_in") if ".qkv_mlp." in name else (None,)
        ):
            source = name if part is None else name.replace("qkv_mlp", part)
            rows = tensor.shape[0] if part is None else {"mlp_in": 384}.get(part, 64)
            value = torch.randn(rows, *tensor.shape[1:], generator=generator) * 0.05
            key = f"dit.{source}"
            # the action boundary layers stay BF16, as in the released packages
            if value.ndim == 2 and ".act" not in source:
                q, scale = quantize_fp8_rowwise(value)
                state[key], state[f"{key}_scale"] = q.cuda(), scale.cuda()
                reference[source] = q.float() * scale[:, None]
            else:
                state[key] = value.to(torch.bfloat16).cuda()
    load_fp8r_checkpoint(model, state)

    block = model.single_blocks[0]
    assert isinstance(block.qkv_mlp, Flux3Fp8RowwiseLinear)
    assert isinstance(model.early_stream_modulations["txt"].lin, Flux3Fp8RowwiseLinear)
    assert isinstance(model.emb_in["act"], torch.nn.Linear)
    assert model.dtype == torch.bfloat16
    x = torch.randn(5, 64, generator=generator).cuda().to(torch.bfloat16)
    expected = torch.cat(
        [
            reference[f"single_blocks.0.{p}.weight"]
            for p in ("q_proj", "k_proj", "v_proj", "mlp_in")
        ]
    ).cuda()
    out, _ = block.qkv_mlp(x)
    torch.testing.assert_close(
        out.float(), x.float() @ expected.T, atol=5e-2, rtol=5e-2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA fused kernel")
def test_fused_qknorm_rope_matches_the_eager_rotation():
    """The fused kernel must use the same [cos | sin] cache layout and interleaved pairs."""
    from sglang.kernels.ops.diffusion import fused_inplace_qknorm_rope
    from sglang.multimodal_gen.runtime.models.dits.flux3 import (
        Flux3QKNorm,
        apply_rope,
        rope_cos_sin,
    )

    torch.manual_seed(0)
    norm = Flux3QKNorm(128).cuda().bfloat16()
    torch.nn.init.normal_(norm.query_norm.scale, mean=1.0, std=0.1)
    torch.nn.init.normal_(norm.key_norm.scale, mean=1.0, std=0.1)
    ids = torch.randint(0, 50, (1, 7, 4)).cuda()
    rope = rope_cos_sin(ids, (32, 32, 32, 32), 10000)
    qkv = torch.randn(1, 7, 3, 2, 128, device="cuda", dtype=torch.bfloat16)
    q, k, v = qkv.unbind(2)
    expected_q, expected_k = apply_rope(*norm(q, k, v), rope)
    fused_inplace_qknorm_rope(
        q=q.view(-1, 2, 128),
        k=k.view(-1, 2, 128),
        q_weight=norm.query_norm.scale,
        k_weight=norm.key_norm.scale,
        cos_sin_cache=rope.reshape(-1, 128),
        positions=torch.arange(7, device="cuda"),
        is_neox=False,
        eps=1e-6,
    )
    torch.testing.assert_close(q, expected_q, atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(k, expected_k, atol=3e-2, rtol=2e-2)
