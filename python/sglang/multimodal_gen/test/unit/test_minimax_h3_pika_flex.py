# SPDX-License-Identifier: Apache-2.0
"""Pika H3 Flex (MiniMax-H3 + six-step LoRA adapters) release contracts."""

from __future__ import annotations

from types import SimpleNamespace

import msgspec
import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    MiniMaxH3PipelineConfig,
    PikaH3FlexPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.minimax_h3 import PikaH3FlexSamplingParams
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    prequeue,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MiniMaxH3DenoiseBranch,
    _minimax_h3_renoise_target_rows_,
    minimax_h3_denoise_loop,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_sequence import (
    minimax_h3_packed_sequence,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    PIKA_H3_FLEX_POLICY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    is_model_overlay,
    resolve_model_overlay,
)

PIKA_KNOTS = (1000, 750, 500, 250, 100, 40)


def test_registry_overlay_and_six_step_contract() -> None:
    info = get_model_info("Pika-Labs/Pika-H3-Flex")
    assert info.sampling_param_cls is PikaH3FlexSamplingParams
    assert info.pipeline_config_cls is PikaH3FlexPipelineConfig
    overlay = resolve_model_overlay("Pika-Labs/Pika-H3-Flex")
    assert overlay["overlay_repo_id"] == "Niehen6174/Pika-H3-Flex-overlay"
    assert is_model_overlay("Pika-Labs/Pika-H3-Flex")
    assert PikaH3FlexSamplingParams(prompt="p").num_inference_steps == 6
    with pytest.raises(ValueError, match="fixed six-step schedule"):
        PikaH3FlexSamplingParams(prompt="p", num_inference_steps=8)
    PikaH3FlexSamplingParams(prompt="p", flow_shift=12.0, audio_flow_shift=3.0)
    for shifts in ({"flow_shift": 10.0}, {"audio_flow_shift": 6.0}):
        with pytest.raises(ValueError, match="trained at"):
            PikaH3FlexSamplingParams(prompt="p", **shifts)


def test_release_metadata_fields() -> None:
    raw = {
        "schema_version": 1,
        "partition": "fl2va",
        "tasks": ["t2va", "fl2va"],
        "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        "dmd_denoising_steps": list(PIKA_KNOTS),
        "stochastic_sampling": True,
        "lora_adapter": "pika/adapter.safetensors",
    }
    metadata = MiniMaxH3ReleaseMetadata.from_model_index({"_minimax_h3": raw})
    assert metadata.stochastic_sampling
    assert metadata.lora_adapter == "pika/adapter.safetensors"
    for overrides, match in (
        ({"dmd_denoising_steps": None}, "requires dmd_denoising_steps"),
        ({"lora_adapter": "../adapter.safetensors"}, "relative .safetensors"),
    ):
        with pytest.raises(ValueError, match=match):
            MiniMaxH3ReleaseMetadata.from_model_index(
                {"_minimax_h3": {**raw, **overrides}}
            )


@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_knot_schedule(shift: float) -> None:
    # diffusers MiniMaxH3Scheduler(shift_provided_sigmas=True) on the same knots
    base = torch.tensor([k / 1000.0 for k in PIKA_KNOTS] + [0.0], dtype=torch.float64)
    expected = (shift * base / (1 + (shift - 1) * base)).tolist()
    sigmas = minimax_h3_time_shift_sigmas(
        num_steps=6, shift_scale=shift, dmd_steps=PIKA_KNOTS
    )
    assert sigmas == pytest.approx(expected, abs=1e-6)


def test_renoise_update() -> None:
    """The released FP32 equations round multiply and add separately."""
    generator = torch.Generator().manual_seed(42)
    state, velocity = [torch.randn(4, 8, generator=generator) for _ in range(2)]
    denoised = state + 0.8 * velocity
    out = state.clone()
    _minimax_h3_renoise_target_rows_(
        out,
        velocity,
        sigma=0.8,
        sigma_next=0.5,
        generator=torch.Generator().manual_seed(7),
    )
    noise = torch.randn(4, 8, generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(out, 0.5 * denoised + 0.5 * noise, rtol=0, atol=0)
    last = state.clone()
    _minimax_h3_renoise_target_rows_(
        last,
        velocity,
        sigma=0.8,
        sigma_next=0.0,
        generator=torch.Generator(),
    )
    torch.testing.assert_close(last, denoised, rtol=0, atol=0)


def _renoise_run(seed: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    packed = minimax_h3_packed_sequence(
        text_len=3,
        latent_t=2,
        latent_h=4,
        latent_w=4,
        audio_t=3,
        include_keyframe_cond=False,
    )
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(3, 5120),
        token_tags=packed["token_tags"],
        device=torch.device("cpu"),
    )
    video_rows = int(branch.update_mask.sum())
    audio_rows = branch.audio_pos.numel()

    def model_forward(_model, kwargs, step):
        return (
            torch.full((video_rows, 96), 0.1 * (step + 1)),
            torch.full((audio_rows, 32), -0.1 * (step + 1)),
        )

    sigmas = minimax_h3_time_shift_sigmas(
        num_steps=6, shift_scale=12.0, dmd_steps=PIKA_KNOTS
    )
    return minimax_h3_denoise_loop(
        model=SimpleNamespace(prepare_adaln_plans=lambda _: None),
        model_forward=model_forward,
        positive=branch,
        initial_video_rows=torch.ones(branch.img_pos.numel(), 96),
        initial_audio_rows=torch.ones(audio_rows, 32),
        keyframe_cond_rows=None,
        sigmas_video=sigmas,
        sigmas_audio=sigmas,
        device=torch.device("cpu"),
        renoise_generator=(
            None if seed is None else torch.Generator().manual_seed(seed)
        ),
    )


def test_renoise_loop_is_seed_deterministic() -> None:
    video, audio = _renoise_run(7)
    again_video, again_audio = _renoise_run(7)
    assert torch.equal(video, again_video) and torch.equal(audio, again_audio)
    other_video, _ = _renoise_run(8)
    assert not torch.equal(video, other_video)
    euler_video, _ = _renoise_run(None)
    assert not torch.equal(video, euler_video)


def test_release_adapter_defaults_to_dynamic_lora(monkeypatch) -> None:
    monkeypatch.setattr(
        MiniMaxH3PipelineConfig, "validate_server_args", lambda self, args: None
    )

    def server_args(explicit: bool, mode: str) -> SimpleNamespace:
        return SimpleNamespace(
            model_variant="FL2VA",
            lora_alpha=64,
            lora_scale=0.5,
            lora_merge_mode=mode,
            is_arg_explicitly_set=lambda name: explicit and name == "lora_merge_mode",
        )

    implicit = server_args(False, "auto")
    PikaH3FlexPipelineConfig().validate_server_args(implicit)
    assert implicit.lora_merge_mode == "dynamic"
    explicit = server_args(True, "merge")
    PikaH3FlexPipelineConfig().validate_server_args(explicit)
    assert explicit.lora_merge_mode == "merge"

    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    with pytest.raises(ValueError, match="disable SGLANG_CACHE_DIT_ENABLED"):
        PikaH3FlexPipelineConfig().validate_server_args(implicit)


def _pika_request(task="t2va", conditions=(), **target):
    return PikaH3FlexSamplingParams(
        prompt="a waterfall",
        task=task,
        conditions=list(conditions),
        target={
            "short_edge": 480,
            "aspect_ratio": "16:9",
            "duration_seconds": 5,
            **target,
        },
    )


def _plan(params):
    return minimax_h3_resolve_plan(
        params.build_request_extra()["minimax_h3_canonical_request"]
    )


@pytest.mark.parametrize(
    "tier,aspect,duration,frames,size",
    [
        (480, "21:9", 5, None, (960, 416)),
        (768, "9:21", 6, 158, (672, 1536)),
        (512, "7:4", 5.5, 132, (896, 512)),
        (480, "16:9", 15, 362, (832, 480)),
    ],
)
def test_geometry_and_warmup_round_trip(tier, aspect, duration, frames, size):
    """Pixel caps and 17n+5 alignment must survive inverse warmup resolution."""
    plan = _plan(
        _pika_request(short_edge=tier, aspect_ratio=aspect, duration_seconds=duration)
    )
    assert (plan.shape["width"], plan.shape["height"]) == size
    target = PikaH3FlexSamplingParams._synthetic_warmup_target(
        SimpleNamespace(width=size[0], height=size[1]),
        SimpleNamespace(
            warmup_resolutions=[f"{size[0]}x{size[1]}"], warmup_num_frames=frames
        ),
    )
    assert target["duration_seconds"] == duration
    assert _plan(_pika_request(**target)).shape == plan.shape


@pytest.mark.parametrize("task,index", [("i2va", 0), ("l2va", -1)])
def test_keyframe_alias_keeps_policy_through_admission(monkeypatch, task, index):
    """Media probing and scheduler serialization must retain Pika's policy and public task."""
    params = _pika_request(
        task,
        [dict(type="image", role="keyframe", uri="frame.png", frame_index=index)],
        aspect_ratio="auto",
    )
    request = VideoGenerationsRequest(
        prompt=params.prompt,
        task=task,
        conditions=params.conditions,
        target=params.target,
    )
    params = PikaH3FlexSamplingParams(
        **PikaH3FlexSamplingParams.lower_video_request_kwargs(
            request, {"prompt": params.prompt}
        )
    )
    batch = SimpleNamespace(
        sampling_params=params,
        extra=params.build_request_extra(),
        num_inference_steps=6,
        is_warmup=False,
    )
    monkeypatch.setattr(
        prequeue,
        "minimax_h3_probe_material",
        lambda *a, **kw: dict(display_width=832, display_height=480),
    )
    plan = prequeue.minimax_h3_prepare_for_queue(batch)
    plan = msgspec.msgpack.decode(msgspec.msgpack.encode(plan), type=type(plan))
    assert plan.policy == PIKA_H3_FLEX_POLICY
    assert plan.condition_mask["semantic_frame_indices"] == [index]
    assert (plan.shape["width"], plan.shape["height"], plan.shape["keyframe_fit"]) == (
        832,
        480,
        "source_crop",
    )
    assert params.task == task
    batch.extra[prequeue.MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = plan
    metadata = MiniMaxH3ReleaseMetadata(
        schema_version=1,
        partition="fl2va",
        tasks=("t2va", "fl2va"),
        task_aliases={},
        video_sigma_shift=12.0,
        audio_sigma_shift=3.0,
    )
    MiniMaxH3PartitionAdmissionStage(metadata).forward(
        batch, SimpleNamespace(minimax_h3_adaln_online=False)
    )


@pytest.mark.parametrize(
    "tier,image_size,video_size",
    [(480, (1536, 1024), (704, 480)), (768, (3072, 2048), (1152, 768))],
)
def test_prequeue_reference_sizes_and_duration_budget(
    monkeypatch, tier, image_size, video_size
):
    """Reference sizes follow the output tier; two individually valid clips can exceed the budget."""
    params = _pika_request(
        "ref2va",
        [
            dict(type=kind, role="reference", uri=str(i))
            for i, kind in enumerate(("image", "video", "video"))
        ],
        short_edge=tier,
    )
    batch = SimpleNamespace(sampling_params=params, extra=params.build_request_extra())
    facts = dict(display_width=600, display_height=400, video_duration_seconds=7.5)
    monkeypatch.setattr(prequeue, "minimax_h3_probe_material", lambda *a, **kw: facts)
    prequeue.minimax_h3_prepare_for_queue(batch)
    shapes = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY]
    assert (shapes[0]["width"], shapes[0]["height"]) == image_size
    assert (shapes[1]["width"], shapes[1]["height"]) == video_size
    for duration in (1.99, 8):
        facts["video_duration_seconds"] = duration
        with pytest.raises(ValueError, match="Pika H3 Flex"):
            prequeue.minimax_h3_prepare_for_queue(batch)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_condition_noise_preserves_clean_cache_and_release_precision(task, monkeypatch):
    """Per-condition draws preserve cache rows and each adapter's conditioning precision."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monkeypatch.setattr(
        denoising.current_platform, "get_local_torch_device", lambda: device
    )
    frames = (1, 1 if task == "fl2va" else 3)
    clean = torch.linspace(-2.003, 7.997, sum(frames) * 2 * 96).reshape(-1, 96)
    original = clean.clone()
    seed = None if task == "fl2va" else 7
    ctx = SimpleNamespace(
        cond_rows=clean,
        audio_ref_rows=torch.ones(4, 32),
        plan=SimpleNamespace(policy=PIKA_H3_FLEX_POLICY, task=task, seed=seed),
        latent_t=8,
        is_ref2va=task == "ref2va",
        keyframe={"keyframes": [dict(latent_h=2, latent_w=4)] * 2}
        if task == "fl2va"
        else None,
        ref2va_positive_blocks=[
            dict(kind=kind, latent_t=t, latent_h=2, latent_w=4, ref_audio_t=2)
            for kind, t in zip(("image", "video"), frames)
        ]
        if task == "ref2va"
        else None,
    )
    audio_aug = 0.0 if task == "ref2va" else 1.0
    sampling = PikaH3FlexSamplingParams(
        prompt="p", audio_cond_noise_aug_for_inference=audio_aug
    )
    sampling.seed = seed  # FL2VA also exercises the internal seed fallback.
    denoising._apply_condition_noise_aug(
        ctx, sampling=sampling, imgvid_noise_aug=0.999, audio_noise_aug=audio_aug
    )
    generator = torch.Generator(device=device).manual_seed(
        (42 if seed is None else seed) + 7919
    )
    noise = torch.cat(
        [
            minimax_h3_patchify_video_latent(
                torch.randn(1, t, 24, 2, 4, generator=generator, device=device).permute(
                    0, 2, 1, 3, 4
                ),
                patch_size=[1, 2, 2],
            )
            for t in frames
        ]
    )
    rounded = (clean.bfloat16().float() if task == "fl2va" else clean).to(device)
    torch.testing.assert_close(
        ctx.cond_rows, 0.999 * rounded + (1.0 - 0.999) * noise, rtol=0, atol=0
    )
    torch.testing.assert_close(clean, original, rtol=0, atol=0)
    expected_audio = (
        torch.randn(4, 32, generator=torch.Generator().manual_seed(8))
        if task == "ref2va"
        else torch.ones(4, 32)
    )
    torch.testing.assert_close(ctx.audio_ref_rows, expected_audio, rtol=0, atol=0)


def test_text_limit_counts_text_only_and_leaves_base_h3_untouched():
    """The release's 2048-token cap excludes media; base H3 must not scan tags."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages.text_encoding import (
        _validate_text_length,
    )

    plan = _plan(_pika_request())
    tags = torch.cat(
        (torch.ones(2048, dtype=torch.int), torch.zeros(4096, dtype=torch.int))
    )
    _validate_text_length(plan, tags, tagged=True)
    tags[-1] = 1
    with pytest.raises(ValueError, match="2049 text tokens"):
        _validate_text_length(plan, tags, tagged=True)
    _validate_text_length(SimpleNamespace(policy=None), object(), tagged=True)
