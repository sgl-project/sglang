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


@pytest.mark.parametrize("variant", ["FL2VA", "Ref2VA"])
def test_release_adapter_defaults_to_dynamic_lora(monkeypatch, variant) -> None:
    monkeypatch.setattr(
        MiniMaxH3PipelineConfig, "validate_server_args", lambda self, args: None
    )

    def server_args(explicit: bool, mode: str) -> SimpleNamespace:
        return SimpleNamespace(
            model_variant=variant,
            lora_alpha=None,
            lora_scale=1.0,
            lora_merge_mode=mode,
            is_arg_explicitly_set=lambda name: explicit and name == "lora_merge_mode",
        )

    implicit = server_args(False, "auto")
    PikaH3FlexPipelineConfig().validate_server_args(implicit)
    assert implicit.lora_merge_mode == "dynamic"
    explicit = server_args(True, "merge")
    PikaH3FlexPipelineConfig().validate_server_args(explicit)
    assert explicit.lora_merge_mode == "merge"


def _pika_request(task="t2va", conditions=None, **target):
    return PikaH3FlexSamplingParams(
        prompt="a waterfall",
        task=task,
        conditions=conditions or [],
        seed=42,
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
    "tier,aspect,expected",
    [
        (480, "21:9", (960, 416)),
        (480, "16:9", (832, 480)),
        (480, "3:2", (704, 480)),
        (480, "4:3", (640, 480)),
        (480, "5:4", (608, 480)),
        (480, "1:1", (480, 480)),
        (480, "4:5", (480, 608)),
        (480, "3:4", (480, 640)),
        (480, "2:3", (480, 704)),
        (480, "9:16", (480, 832)),
        (480, "9:21", (416, 960)),
        (768, "21:9", (1536, 672)),
        (768, "16:9", (1344, 768)),
        (768, "3:2", (1152, 768)),
        (768, "4:3", (1024, 768)),
        (768, "5:4", (960, 768)),
        (768, "1:1", (768, 768)),
        (768, "4:5", (768, 960)),
        (768, "3:4", (768, 1024)),
        (768, "2:3", (768, 1152)),
        (768, "9:16", (768, 1344)),
        (768, "9:21", (672, 1536)),
    ],
)
def test_released_buckets_round_trip_through_warmup(tier, aspect, expected):
    """Released canvases must survive inverse warmup geometry without bucket drift."""
    plan = _plan(_pika_request(short_edge=tier, aspect_ratio=aspect))
    assert (plan.shape["width"], plan.shape["height"]) == expected
    warmup_target = PikaH3FlexSamplingParams._synthetic_warmup_target(
        SimpleNamespace(width=expected[0], height=expected[1]),
        SimpleNamespace(
            warmup_resolutions=[f"{expected[0]}x{expected[1]}"], warmup_num_frames=None
        ),
    )
    assert warmup_target == {
        "short_edge": tier,
        "aspect_ratio": aspect,
        "duration_seconds": 5.0,
    }
    assert _plan(_pika_request(**warmup_target)).shape == plan.shape


@pytest.mark.parametrize(
    "task,indices", [("i2va", [0]), ("l2va", [-1]), ("fl2va", [0, -1])]
)
def test_keyframe_tasks_survive_http_lowering_and_partition_admission(
    monkeypatch, task, indices
):
    """Canonical FL2VA admission must preserve each public keyframe task."""
    conditions = [
        dict(type="image", role="keyframe", uri=f"frame{i}.png", frame_index=i)
        for i in indices
    ]
    params = _pika_request(task, conditions)
    request = VideoGenerationsRequest(
        prompt=params.prompt, task=task, conditions=conditions, target=params.target
    )
    kwargs = PikaH3FlexSamplingParams.lower_video_request_kwargs(
        request, {"prompt": params.prompt}
    )
    http_plan = _plan(PikaH3FlexSamplingParams(**kwargs))
    plan = _plan(params)
    assert http_plan.shape == plan.shape
    assert plan.task == "fl2va"
    assert plan.condition_mask["semantic_frame_indices"] == indices
    assert plan.shape["keyframe_fit"] == "source_crop"
    batch = SimpleNamespace(
        sampling_params=params,
        extra=params.build_request_extra(),
        num_inference_steps=6,
        is_warmup=False,
    )
    monkeypatch.setattr(
        prequeue,
        "minimax_h3_probe_material",
        lambda *a, **kw: {"display_width": 832, "display_height": 480},
    )
    prepared = prequeue.minimax_h3_prepare_for_queue(batch)
    # The scheduler handoff must retain the release policy after media probing.
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(prepared), type=type(plan))
    assert decoded == prepared
    batch.extra[prequeue.MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY] = decoded
    assert params.task == task
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
    "target",
    [
        {"short_edge": 512},
        {"aspect_ratio": "auto"},
        {"duration_seconds": 4},
        {"duration_seconds": 5.5},
    ],
)
def test_pika_rejects_unreleased_targets(target):
    """Pika restricts base H3 geometry and duration to the released CLI contract."""
    with pytest.raises(ValueError, match="Pika H3 Flex target"):
        _plan(_pika_request(**target))


@pytest.mark.parametrize(
    "counts,match",
    [
        ((10, 0, 0), "reference limits"),
        ((1, 4, 0), "reference limits"),
        ((1, 0, 4), "reference limits"),
        ((9, 3, 1), "reference limits"),
        ((0, 0, 1), "at least one image or video"),
    ],
)
def test_reference_limits(counts, match):
    """Per-kind counts, total count and required visual input are independent limits."""
    refs = [
        dict(type=kind, role="reference", uri=f"{kind}{i}")
        for kind, count in zip(("image", "video", "audio"), counts)
        for i in range(count)
    ]
    with pytest.raises(ValueError, match=match):
        _plan(_pika_request("ref2va", refs))


@pytest.mark.parametrize(
    "durations,accepted",
    [([2, 13], True), ([1.99], False), ([15.01], False), ([8, 8], False)],
)
def test_reference_duration_limits_before_queue(monkeypatch, durations, accepted):
    refs = [
        dict(type="video", role="reference", uri=f"ref{i}.mp4")
        for i in range(len(durations))
    ]
    params = _pika_request("ref2va", refs)
    batch = SimpleNamespace(sampling_params=params, extra=params.build_request_extra())

    def probe(batch, uri, *, condition_type, condition_index):
        return {
            "display_width": 1920,
            "display_height": 1080,
            "video_duration_seconds": durations[condition_index],
        }

    monkeypatch.setattr(prequeue, "minimax_h3_probe_material", probe)
    if accepted:
        prequeue.minimax_h3_prepare_for_queue(batch)
    else:
        with pytest.raises(ValueError, match="Pika H3 Flex"):
            prequeue.minimax_h3_prepare_for_queue(batch)


@pytest.mark.parametrize("tier,expected", [(480, (1536, 1024)), (768, (3072, 2048))])
def test_reference_image_uses_output_tier(monkeypatch, tier, expected):
    params = _pika_request(
        "ref2va", [dict(type="image", role="reference", uri="ref.png")], short_edge=tier
    )
    batch = SimpleNamespace(sampling_params=params, extra=params.build_request_extra())
    monkeypatch.setattr(
        prequeue,
        "minimax_h3_probe_material",
        lambda *a, **kw: {"display_width": 600, "display_height": 400},
    )
    prequeue.minimax_h3_prepare_for_queue(batch)
    shape = batch.extra[prequeue.MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY][0]
    assert (shape["width"], shape["height"]) == expected


@pytest.mark.parametrize(
    "task,seed,device",
    [
        ("fl2va", 42, "cpu"),
        ("ref2va", 42, "cpu"),
        ("fl2va", None, "cpu"),
        pytest.param(
            "ref2va",
            42,
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA condition noise"
            ),
        ),
    ],
)
def test_condition_noise_preserves_clean_cache_and_release_precision(
    task, seed, device, monkeypatch
):
    """Each condition draws in external latent order; an omitted seed uses 42."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        denoising,
    )

    shapes = [(1, 4, 6), (1 if task == "fl2va" else 3, 4, 4)]
    monkeypatch.setattr(
        denoising.current_platform,
        "get_local_torch_device",
        lambda: torch.device(device),
    )
    clean = torch.linspace(
        -2.003, 7.997, sum(t * h * w // 4 for t, h, w in shapes) * 96
    ).reshape(-1, 96)
    original = clean.clone()
    audio = torch.tensor([[0.123456]], dtype=torch.float32)
    sampling = SimpleNamespace(seed=seed)
    plan = SimpleNamespace(policy=PIKA_H3_FLEX_POLICY, task=task, seed=seed)
    contexts = [
        SimpleNamespace(
            cond_rows=clean,
            audio_ref_rows=audio,
            plan=plan,
            latent_t=8,
            is_ref2va=task == "ref2va",
            keyframe={"keyframes": [dict(latent_h=h, latent_w=w) for _, h, w in shapes]}
            if task == "fl2va"
            else None,
            ref2va_positive_blocks=[
                dict(kind=kind, latent_t=t, latent_h=h, latent_w=w)
                for kind, (t, h, w) in zip(("image", "video"), shapes)
            ]
            if task == "ref2va"
            else None,
        )
        for _ in range(2)
    ]
    for context in contexts:
        denoising._apply_condition_noise_aug(
            context, sampling=sampling, imgvid_noise_aug=0.999, audio_noise_aug=1.0
        )
    # Reference wrapper draws [B,F,C,H,W] per condition before packing.
    generator = torch.Generator(device=device).manual_seed(42 + 7919)
    noise = torch.cat(
        [
            minimax_h3_patchify_video_latent(
                torch.randn(1, t, 24, h, w, generator=generator, device=device).permute(
                    0, 2, 1, 3, 4
                ),
                patch_size=[1, 2, 2],
            )
            for t, h, w in shapes
        ]
    )
    rounded = (clean.bfloat16().float() if task == "fl2va" else clean).to(device)
    torch.testing.assert_close(
        contexts[0].cond_rows, 0.999 * rounded + (1.0 - 0.999) * noise, rtol=0, atol=0
    )
    assert torch.equal(contexts[0].cond_rows, contexts[1].cond_rows)
    assert torch.equal(clean, original)
    assert contexts[0].audio_ref_rows is audio


def test_server_cache_default_cannot_override_the_six_step_contract(monkeypatch):
    """A global cache default must not bypass the release's full six evaluations."""
    monkeypatch.setenv("SGLANG_CACHE_DIT_ENABLED", "true")
    monkeypatch.setattr(
        MiniMaxH3PipelineConfig, "validate_server_args", lambda *_: None
    )
    args = SimpleNamespace(
        model_variant="fl2va",
        lora_path=None,
        lora_alpha=None,
        lora_scale=1.0,
        is_arg_explicitly_set=lambda _: False,
    )
    with pytest.raises(ValueError, match="disable SGLANG_CACHE_DIT_ENABLED"):
        PikaH3FlexPipelineConfig().validate_server_args(args)


@pytest.mark.parametrize(
    "duration,frames,aligned",
    [(5, 120, 124), (5, 124, 124), (6, 158, 158), (15, 362, 362)],
)
def test_warmup_frames_recover_requested_duration(duration, frames, aligned):
    """17n+5 alignment is not inverted by dividing aligned frames by 24."""
    target = PikaH3FlexSamplingParams._synthetic_warmup_target(
        SimpleNamespace(width=832, height=480),
        SimpleNamespace(warmup_resolutions=["832x480"], warmup_num_frames=frames),
    )
    assert target["duration_seconds"] == duration
    assert _plan(_pika_request(**target)).shape["frame_count"] == aligned


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
