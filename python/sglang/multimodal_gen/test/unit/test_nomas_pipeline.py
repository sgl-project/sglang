# SPDX-License-Identifier: Apache-2.0
"""Pipeline contracts and optional fixed-checkpoint/reference parity.

NOMAS_RANDOM_PIPELINE points to test/scripts/create_nomas_fixture.py's output;
NOMAS_REFERENCE_ROOT points to the pinned upstream repository.
"""

import os
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.nomas import (
    NomasVAEArchConfig,
    NomasVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.nomas import (
    NomasPipelineConfig,
    nomas_postprocess_text,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.nomas import (
    GenExpertArch,
    NomasPixelUMMDecoder,
    window_starts,
)


@pytest.fixture(autouse=True, scope="module")
def _model_parallel(single_process_model_parallel):
    """The framework's parallel linears resolve the TP group at construction."""
    yield


def test_exact_upstream_schedule_and_shapes():
    config = NomasPipelineConfig()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    for steps in (1, 2, 8, 50):
        scheduler.set_timesteps(sigmas=config.prepare_sigmas(None, steps))
        s = torch.linspace(1, 0, steps + 1)
        expected = 5 * s / (1 + 4 * s)
        torch.testing.assert_close(scheduler.sigmas.cpu(), expected, atol=0, rtol=0)
    request = SimpleNamespace(height=64, width=96, num_frames=48, guidance_scale=1)
    assert config.prepare_latent_shape(request, 1, 12) == (1, 32, 12, 2, 3)
    request.num_frames = 49
    assert config.prepare_latent_shape(request, 1, 13) == (1, 32, 13, 2, 3)
    request.num_frames = 48
    with pytest.raises(ValueError, match="batch size"):
        config.prepare_latent_shape(request, 2, 12)
    assert not config.supports_dynamic_batching()


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"generator_device": None},
        {"generator_device": "cuda"},
        {"generator_device": "cpu"},
    ],
)
def test_http_video_builder_uses_nomas_cpu_noise_policy(monkeypatch, overrides):
    from sglang.multimodal_gen.configs.sample.nomas import NomasSamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        VideoGenerationsRequest,
    )

    request = VideoGenerationsRequest(
        prompt="road",
        seed=42,
        num_frames=17,
        segments=[{"frames": 17, "caption": "road"}],
        prompt_format="json",
        **overrides,
    )
    monkeypatch.setattr(video_api, "get_global_server_args", lambda: SimpleNamespace())
    monkeypatch.setattr(
        video_api, "resolve_sampling_params_cls", lambda _: NomasSamplingParams
    )
    monkeypatch.setattr(
        video_api, "build_sampling_params", lambda request_id, **kwargs: kwargs
    )
    kwargs = video_api._build_video_sampling_params("nomas-http", request)
    assert kwargs["generator_device"] == "cpu"
    assert kwargs["seed"] == 42
    assert kwargs["segments"] == [{"frames": 17, "caption": "road"}]
    assert kwargs["prompt_format"] == "json"


def test_nomas_http_extensions_stay_model_specific():
    from sglang.multimodal_gen.configs.sample.nomas import NomasSamplingParams
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        VideoGenerationsRequest,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        request_model_kwargs,
    )

    values = {
        "continuation_prompts": ["turn left"],
        "prompt_format": "plain",
        "norm_extrinsics": torch.eye(4).repeat(17, 1, 1).tolist(),
    }
    request = VideoGenerationsRequest(prompt="road", extra_body=values)
    assert request_model_kwargs(request, NomasSamplingParams, "video") == values
    assert request_model_kwargs(request, SamplingParams, "video") == {}


def test_text_mask_and_skip_layer_contract():
    mask = torch.tensor([[1, 1, 1, 1, 0]])
    hidden = tuple(torch.full((1, 5, 8), i, dtype=torch.float32) for i in range(5))
    emb, out_mask = nomas_postprocess_text(
        SimpleNamespace(hidden_states=hidden),
        {"attention_mask": mask, "nomas_crop_start": torch.tensor(2)},
    )
    assert emb.shape == (1, 3, 8)
    assert (emb == 2).all()
    assert out_mask.tolist() == [[1, 1, 0]]


def test_segment_text_dedup_includes_continuations_and_copies_conditions():
    from sglang.multimodal_gen.configs.sample.nomas import NomasSamplingParams
    from sglang.multimodal_gen.runtime.pipelines.nomas_pipeline import (
        NomasTextEncodingStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

    stage = object.__new__(NomasTextEncodingStage)
    args = SimpleNamespace(pipeline_config=NomasPipelineConfig())
    a = Req(sampling_params=NomasSamplingParams(prompt="road", num_frames=48))
    b = Req(
        sampling_params=NomasSamplingParams(
            prompt="road", num_frames=48, continuation_prompts=["turn left"]
        )
    )
    assert stage.build_dedup_fingerprint(a, args) != stage.build_dedup_fingerprint(
        b, args
    )
    a.extra["nomas_conditions"] = ["sentinel"]
    stage.copy_deduplicated_outputs(a, b)
    assert b.extra["nomas_conditions"] == ["sentinel"]
    assert b.extra["nomas_conditions"] is not a.extra["nomas_conditions"]


def test_pixelumm_window_boundaries_and_config_rejection():
    assert window_starts(8, 10, 5) == [0]
    assert window_starts(12, 10, 5) == [0, 2]
    assert window_starts(21, 10, 5) == [0, 5, 10, 11]
    for kwargs in (
        {"temporal_overlap_latents": 10},
        {"num_steps": 0},
        {"denoising_step_list": (0.9, 0.95), "num_steps": 2},
    ):
        with pytest.raises(ValueError):
            NomasVAEArchConfig(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA fusion")
@pytest.mark.parametrize(
    "tokens,heads,batch", [(17, 8, 1), (1025, 32, 1), (2049, 8, 2)]
)
@torch.inference_mode()
def test_pixelumm_fused_rope_preserves_chunk_rounding(tokens, heads, batch):
    from sglang.multimodal_gen.runtime.models.vaes.nomas import (
        _QK_CHUNK_INPUT_BYTES,
        GenAttention,
        apply_mrope,
    )

    torch.manual_seed(831)
    attention = GenAttention(GenExpertArch(hidden_size=128)).cuda().bfloat16()
    attention.q_norm_t.weight.normal_()
    attention.q_norm_hw.weight.normal_()
    x = torch.randn(batch, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(1, tokens, 1, 128, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    chunk = max(1, _QK_CHUNK_INPUT_BYTES // (batch * heads * 128 * 4))
    expected = torch.empty_like(x)
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        normalized = attention._split_norm(
            x[:, start:end].float(), attention.q_norm_t, attention.q_norm_hw
        )
        expected[:, start:end] = apply_mrope(
            normalized, cos[:, start:end], sin[:, start:end], (64, 32, 32)
        ).to(x.dtype)
    actual = attention._prepare_qk(
        x, cos, sin, attention.q_norm_t, attention.q_norm_hw, x.dtype
    )
    assert torch.equal(actual, expected)


@pytest.mark.skipif(
    not os.getenv("NOMAS_RANDOM_PIPELINE"), reason="requires random pipeline fixture"
)
def test_tokenizer_template_and_crop():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path(os.environ["NOMAS_RANDOM_PIPELINE"]) / "tokenizer"
    )
    config = NomasPipelineConfig()
    for prompt in ("A red car drives forward.", "", "A blue boat turns left."):
        inputs = config.tokenize_prompt([prompt], tokenizer, {"max_length": 64})
        crop = inputs["nomas_crop_start"].item()
        assert inputs["input_ids"].shape[1] == crop + 64
        assert inputs["attention_mask"][0, crop] == 1
        text = tokenizer.decode(
            inputs["input_ids"][0, crop:], skip_special_tokens=False
        )
        assert text.startswith(prompt or " ")
        assert "<|im_end|>" in text


@pytest.mark.skipif(
    not os.getenv("NOMAS_REFERENCE_ROOT") or not torch.cuda.is_available(),
    reason="requires CUDA and upstream checkout",
)
@pytest.mark.parametrize(
    "steps,length,tile", [(1, 8, 8), (1, 12, 8), (4, 12, 8), (1, 3, 1)]
)
@torch.inference_mode()
def test_pixelumm_reference_parity(steps, length, tile):
    sys.path.insert(0, os.environ["NOMAS_REFERENCE_ROOT"])
    from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder
    from hymm.models.vae.pixelumm_decoder.model import (
        DCAEVPixelUMMDecoder as ReferenceDecoder,
    )
    from hymm.models.vae.pixelumm_decoder.model import GenExpertArch as ReferenceArch

    torch.manual_seed(73)
    arch = NomasVAEArchConfig(
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=64,
        num_steps=steps,
        local_attention_tile=tile,
    )
    native = (
        NomasPixelUMMDecoder(NomasVAEConfig(arch_config=arch)).eval().cuda().bfloat16()
    )
    ref_model = ReferenceDecoder(
        ReferenceArch(**{f.name: getattr(arch, f.name) for f in fields(GenExpertArch)}),
        local_attention_tile=tile,
    )
    ref = (
        PixelUMMDCAELatentDecoder(
            ref_model,
            num_steps=steps,
            decoder_tile_pixels=0,
            temporal_window_latents=10,
            temporal_overlap_latents=5,
        )
        .eval()
        .cuda()
        .bfloat16()
    )
    ref.load_state_dict(native.state_dict(), strict=True)
    latent = torch.randn(1, 32, length, 2, 2, device="cuda", dtype=torch.bfloat16)
    expected = ref.decode(latent).sample
    actual = native.decode(latent).sample
    assert torch.isfinite(actual).all()
    assert actual.shape == (1, 3, length * 4, 64, 64)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(actual, native.decode(latent).sample, atol=0, rtol=0)
    assert not torch.equal(actual, native.decode(latent + 0.25).sample)


@pytest.mark.skipif(
    not os.getenv("NOMAS_REFERENCE_ROOT") or not torch.cuda.is_available(),
    reason="requires CUDA and upstream checkout",
)
@pytest.mark.parametrize("frames", [(17, 32, 9), (40, 40, 40), (13, 33, 9, 1, 61)])
@torch.inference_mode()
def test_pixelumm_streaming_matches_upstream_stable_prefix(frames):
    """Per-event output of the realtime timeline equals upstream push/finish."""
    sys.path.insert(0, os.environ["NOMAS_REFERENCE_ROOT"])
    from hymm.models.vae.pixelumm_decoder.autoencoder import PixelUMMDCAELatentDecoder
    from hymm.models.vae.pixelumm_decoder.model import (
        DCAEVPixelUMMDecoder as ReferenceDecoder,
    )
    from hymm.models.vae.pixelumm_decoder.model import GenExpertArch as ReferenceArch

    from sglang.multimodal_gen.test.unit.test_nomas_realtime import stream_events

    torch.manual_seed(73)
    arch = NomasVAEArchConfig(
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=64,
    )
    native = (
        NomasPixelUMMDecoder(NomasVAEConfig(arch_config=arch)).eval().cuda().bfloat16()
    )
    ref_model = ReferenceDecoder(
        ReferenceArch(**{f.name: getattr(arch, f.name) for f in fields(GenExpertArch)}),
        local_attention_tile=arch.local_attention_tile,
    )
    ref = (
        PixelUMMDCAELatentDecoder(
            ref_model,
            num_steps=1,
            decoder_tile_pixels=0,
            temporal_window_latents=10,
            temporal_overlap_latents=5,
        )
        .eval()
        .cuda()
        .bfloat16()
    )
    ref.load_state_dict(native.state_dict(), strict=True)
    gen = torch.Generator(device="cuda").manual_seed(5)
    segments = [
        torch.randn(
            1, 32, (f - 1) // 4 + 1, 2, 2, device="cuda", generator=gen
        ).bfloat16()
        for f in frames
    ]
    session = ref.create_streaming_decode_session(to_cpu=False, retain_history=False)
    expected = [session.push(s, pixel_frames=f) for s, f in zip(segments, frames)]
    expected.append(session.finish())
    _, actual = stream_events(native, segments, frames)
    assert [0 if a is None else a.shape[2] for a in actual] == [
        e.shape[2] if e.numel() else 0 for e in expected
    ]
    for a, e in zip(actual, expected):
        if a is not None:
            torch.testing.assert_close(a, e, atol=0, rtol=0)


def test_scheduler_loader_uses_effective_shift(tmp_path):
    import json

    from sglang.multimodal_gen.runtime.loader.component_loaders.scheduler_loader import (
        SchedulerLoader,
    )
    from sglang.multimodal_gen.runtime.pipelines.nomas_pipeline import NomasPipeline

    (tmp_path / "scheduler_config.json").write_text(
        json.dumps(
            {
                "_class_name": "FlowMatchEulerDiscreteScheduler",
                "shift": 1.0,
                "num_train_timesteps": 1000,
            }
        )
    )
    config = NomasPipelineConfig()
    args = SimpleNamespace(pipeline_config=config)
    scheduler = SchedulerLoader().load_customized(str(tmp_path), args)
    assert scheduler.config.shift == 1 and scheduler.shift == 5
    encoder = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=3584))
    )
    pipeline = SimpleNamespace(
        get_module={"scheduler": scheduler, "text_encoder": encoder}.__getitem__
    )
    NomasPipeline.initialize_pipeline(pipeline, args)
    scheduler.set_timesteps(sigmas=config.prepare_sigmas(None, 8))
    sigma = torch.linspace(1, 0, 9)
    torch.testing.assert_close(
        scheduler.sigmas, 5 * sigma / (1 + 4 * sigma), atol=0, rtol=0
    )
    scheduler.set_shift(1)
    with pytest.raises(ValueError, match="effective shift"):
        NomasPipeline.initialize_pipeline(pipeline, args)


def test_event_plan_and_budget():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.request_validation import (
        check_cache_budget,
        plan_segments,
    )

    events = [{"frames": 17, "caption": "first"}, {"frames": 32, "caption": "second"}]
    plan = plan_segments(49, "unused", events, None, 10)
    assert [(s.start, s.end, s.caption) for s in plan] == [
        (0, 4, "first"),
        (4, 13, "second"),
    ]
    with pytest.raises(ValueError, match="sum of segment"):
        plan_segments(48, "unused", events, None, 10)
    with pytest.raises(ValueError, match="empty latent"):
        plan_segments(2, "unused", [{"frames": 1, "caption": "a"}] * 2, None, 10)
    config = NomasPipelineConfig()
    assert check_cache_budget(config, 10080) == 4459069440
    with pytest.raises(ValueError, match="KV requires"):
        check_cache_budget(config, 10080 * 20)
    with pytest.raises(ValueError, match="KV requires"):
        check_cache_budget(config, 10080, free_bytes=9 * 2**30)


def test_action_windows_and_camera_gate_follow_upstream():
    """Malformed action timing must degrade to full attention, not reject the
    request, and a camera-on segment must not carry a gate mask."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
        NomasSegmentCondition,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.request_validation import (
        action_video_windows,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.stages import (
        segment_camera_condition,
    )

    drive = {"start_time": 0, "end_time": "00:00.5"}
    turn = {"start_time": 0.5, "end_time": 1}
    assert action_video_windows([drive, turn], 6, 24) == ((0, 3, 0), (3, 6, 1))
    # A null timestamp is 0 seconds upstream; here it empties the second span.
    assert action_video_windows(
        [drive, {"start_time": 0.5, "end_time": None}], 6, 24
    ) == (
        (0, 3, 0),
        (3, 6, -1),
    )
    assert action_video_windows([drive], 6, 24) == ((0, 3, 0), (3, 6, -1))
    assert action_video_windows([{"start_time": 0, "end_time": 9}], 6, 24) is None
    assert (
        action_video_windows([drive, {"start_time": 0.25, "end_time": 1}], 6, 24)
        is None
    )
    assert action_video_windows([drive, {"action": "no timing"}], 6, 24) is None
    assert action_video_windows([{"start_time": 1, "end_time": 0.5}], 6, 24) is None

    condition = NomasSegmentCondition(
        torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)
    )
    assert segment_camera_condition(condition, None, 0, 2) is condition
    sliced = segment_camera_condition(condition, torch.arange(5.0).view(1, 5, 1), 1, 3)
    assert sliced.camera_embedding.flatten().tolist() == [1.0, 2.0]
    assert sliced.camera_has_mask is None


@pytest.mark.skipif(
    not os.getenv("NOMAS_RANDOM_PIPELINE") or not os.getenv("NOMAS_REFERENCE_ROOT"),
    reason="requires tokenizer and upstream",
)
@torch.inference_mode()
def test_upstream_prompt_encoding_and_action_metadata():
    import json

    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen2_5_VLConfig, Qwen2_5_VLTextModel

    from sglang.multimodal_gen.configs.pipeline_configs.nomas import NOMAS_SYSTEM_PROMPT
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.request_validation import (
        action_metadata,
        canonical_caption,
    )

    sys.path.insert(0, os.environ["NOMAS_REFERENCE_ROOT"])
    from hymm.models.text_encoder.encoder import TextEncoder
    from hymm.pipeline.base import BasePipeline

    path = Path(os.environ["NOMAS_RANDOM_PIPELINE"])
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    model = Qwen2_5_VLTextModel(
        Qwen2_5_VLConfig.from_pretrained(path / "text_encoder").text_config
    ).eval()
    weights = load_file(str(path / "text_encoder/model.safetensors"))
    # Fixture uses the SGLang/HF-v4 component namespace, not HF-v5's multimodal wrapper.
    model.load_state_dict(
        {
            k.removeprefix("model."): v
            for k, v in weights.items()
            if k.startswith("model.")
        },
        strict=True,
    )
    encoder = TextEncoder.__new__(TextEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.model, encoder.tokenizer = model, tokenizer
    encoder.text_encoder_type = "qwen2.5"
    encoder.use_template = encoder.use_video_template = encoder.use_attention_mask = (
        True
    )
    encoder.hidden_state_skip_layer = 2
    encoder.apply_final_norm = False
    encoder.reproduce = True
    config = NomasPipelineConfig()
    crop = int(
        config.tokenize_prompt(["x"], tokenizer, {"max_length": 512})[
            "nomas_crop_start"
        ]
    )
    encoder.prompt_template_video = {
        "crop_start": crop,
        "template": [
            {"role": "system", "content": NOMAS_SYSTEM_PROMPT},
            {"role": "user", "content": "{}"},
        ],
    }
    reference = SimpleNamespace(
        text_encoder=encoder,
        args=SimpleNamespace(text_len=512, video_fps=24),
        device=torch.device("cpu"),
        target_dtype=torch.float32,
        t_ratio=4,
    )
    caption = canonical_caption(
        json.dumps(
            {
                "segments": [
                    {
                        "description": "a road",
                        "contained_actions": [
                            {"start_time": 0, "end_time": 0.5, "action": "drive"},
                            {"start_time": 0.5, "end_time": 1, "action": "turn"},
                        ],
                    }
                ]
            }
        ),
        "json",
    )
    for prompt in ("A car.", caption, "long " * 1000):
        inputs = config.tokenize_prompt([prompt], tokenizer, {"max_length": 512})
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        actual, mask = nomas_postprocess_text(outputs, inputs)
        expected, expected_mask = BasePipeline._encode_single_prompt(reference, prompt)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert torch.equal(mask.bool(), expected_mask.bool())
    inputs = config.tokenize_prompt([caption], tokenizer, {"max_length": 512})
    mask = inputs["attention_mask"][:, crop:].bool()
    start, offsets, windows = action_metadata(caption, config, tokenizer, mask, 6, 24)
    assert start == BasePipeline._compute_action_start(reference, caption)
    ref_offsets = BasePipeline._compute_action_text_offsets(reference, caption)
    assert list(offsets[:-1]) == ref_offsets[:-1]
    assert offsets[-1] == int(mask.sum())  # upstream helper counts pre-crop tokens here
    assert [list(w) for w in windows] == BasePipeline._compute_action_video_windows(
        reference, caption, 0, 6
    )
    overlapping = canonical_caption(
        json.dumps(
            {
                "segments": [
                    {
                        "description": "a road",
                        "contained_actions": [
                            {"start_time": 0, "end_time": 0.5, "action": "drive"},
                            {"start_time": 0.25, "end_time": 1, "action": "turn"},
                        ],
                    }
                ]
            }
        ),
        "json",
    )
    inputs = config.tokenize_prompt([overlapping], tokenizer, {"max_length": 512})
    mask = inputs["attention_mask"][:, crop:].bool()
    with pytest.raises(ValueError):
        BasePipeline._compute_action_video_windows(reference, overlapping, 0, 6)
    assert action_metadata(overlapping, config, tokenizer, mask, 6, 24) == (
        BasePipeline._compute_action_start(reference, overlapping),
        None,
        None,
    )
    with pytest.raises(ValueError, match="segment JSON"):
        canonical_caption("not json", "json")


@pytest.mark.parametrize("online", [False, True])
def test_latent_preparation_preserves_offline_noise_and_skips_online_draw(online):
    from sglang.multimodal_gen.configs.sample.nomas import NomasSamplingParams
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
        LatentPreparationStage,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.stages import (
        NomasLatentPreparationStage,
    )

    config = NomasPipelineConfig()
    args = SimpleNamespace(pipeline_config=config)
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)

    def request():
        batch = Req(
            sampling_params=NomasSamplingParams(
                prompt="road",
                num_frames=17,
                height=64,
                width=64,
            )
        )
        batch.generator = torch.Generator().manual_seed(42)
        batch.prompt_embeds = [torch.zeros(1, 8, 128)]
        return batch

    batch = request()
    before = batch.generator.get_state()
    if online:
        batch.session = SimpleNamespace()
    result = NomasLatentPreparationStage(scheduler, None).forward(batch, args)
    assert result.latents.shape == (1, 32, 5, 2, 2)
    if online:
        assert torch.equal(result.generator.get_state(), before)
        assert (
            result.latents.untyped_storage().nbytes() == result.latents.element_size()
        )
    else:
        expected = LatentPreparationStage(scheduler, None).forward(request(), args)
        torch.testing.assert_close(result.latents, expected.latents, atol=0, rtol=0)
        assert torch.equal(result.generator.get_state(), expected.generator.get_state())
