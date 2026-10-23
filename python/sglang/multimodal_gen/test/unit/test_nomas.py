# SPDX-License-Identifier: Apache-2.0
"""Weightless Nomas tests: DiT segment session, pipeline stages and request contracts."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.nomas import NomasArchConfig, NomasConfig
from sglang.multimodal_gen.configs.models.vaes.nomas import NomasVAEArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.nomas import (
    NomasPipelineConfig,
    nomas_postprocess_text,
)
from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel
from sglang.multimodal_gen.runtime.models.dits.nomas_kv_cache import (
    NomasAttentionPlan,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.nomas import GenExpertArch, window_starts
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
    NomasLatentSession,
    NomasSegmentCondition,
)
from sglang.srt.utils.network import get_free_port_below_ephemeral


@pytest.fixture(autouse=True, scope="module")
def _model_parallel(single_process_model_parallel):
    """The framework's parallel linears resolve the TP group at construction."""
    yield


def tiny_model(device="cpu", backend="sdpa"):
    torch.manual_seed(123)
    model = (
        NomasTransformer3DModel(
            NomasConfig(
                arch_config=NomasArchConfig(
                    hidden_size=256,
                    num_attention_heads=2,
                    num_layers=2,
                )
            ),
            attention_backend=backend,
        )
        .eval()
        .requires_grad_(False)
    )
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and "norm" in name and name.endswith("weight"):
                p.fill_(1)
            else:
                p.normal_(0, 0.02)
    return model.to(device)


def condition(device="cpu", dtype=torch.float32, padded=True):
    gen = torch.Generator().manual_seed(456)
    valid = 6 if padded else 8
    return NomasSegmentCondition(
        text_states=torch.randn(1, 8, 3584, generator=gen).to(
            device=device, dtype=dtype
        ),
        text_mask=(torch.arange(8, device=device)[None] < valid),
        camera_embedding=torch.randn(1, 2, 64, generator=gen).to(
            device=device, dtype=dtype
        ),
        camera_has_mask=torch.ones(1, device=device, dtype=torch.bool),
        text_offsets=(2, 4, valid),
        video_windows=((0, 1, 0), (1, 2, 1)),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA fusion")
@pytest.mark.parametrize("tokens", [1280, 26400])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_shared_modulation_and_residual_preserve_rounding(tokens, dtype):
    from sglang.multimodal_gen.runtime.models.dits.nomas_layers import (
        gated_residual,
        modulate,
    )

    gen = torch.Generator(device="cuda").manual_seed(831)
    x = torch.randn(1, tokens, 2048, device="cuda", dtype=dtype, generator=gen)
    update = torch.randn(x.shape, device="cuda", dtype=dtype, generator=gen)
    shift, scale, gate = torch.randn(
        1, 3 * 2048, device="cuda", dtype=torch.bfloat16, generator=gen
    ).chunk(3, -1)
    # BF16 exercises the fused path; FP32 covers camera-promoted residuals.
    expected = x * (1 + scale[:, None]) + shift[:, None]
    assert torch.equal(modulate(x, shift, scale), expected)
    expected = x + update * gate[:, None]
    assert torch.equal(gated_residual(x, update, gate), expected)
    # Non-contiguous sequence views must retain the same fallback semantics.
    assert torch.equal(
        gated_residual(x[:, ::2], update[:, ::2], gate), expected[:, ::2]
    )


def test_segment_mask_against_independent_dense_oracle():
    torch.manual_seed(77)
    q, k, v = (torch.randn(1, n, 2, 16) for n in (16, 19, 19))
    plan = NomasAttentionPlan(
        torch.tensor([[True] * 6 + [False] * 2]),
        torch.tensor([[True, False, True]]),
        2,
        4,
        (2, 4, 6),
        ((0, 1, 0), (1, 2, 1)),
        "sdpa",
    )
    expected = torch.zeros(16, 19, dtype=torch.bool)
    expected[:, [0, 2]] = True
    expected[:, 3:13] = True
    expected[:4, 13:15] = True
    expected[4:8, 15:17] = True
    expected[10:12, 13:15] = True
    expected[12:14, 15:17] = True
    result = (
        torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=expected
        )
        .transpose(1, 2)
        .flatten(2)
    )
    result[:, 14:] = 0
    torch.testing.assert_close(plan(q, k, v).flatten(2), result, atol=0, rtol=0)


@torch.inference_mode()
def test_session_commit_reset_isolation_capacity_and_retry():
    model = tiny_model()
    a = NomasLatentSession(model, cache_capacity=32, seed=42, steps=2)
    b = NomasLatentSession(model, cache_capacity=32, seed=42, steps=2)
    cond = condition()
    first = a.generate_segment((2, 2, 2), cond)
    assert (
        first.abs().max() > 0
        and a.cache.valid_length == 16
        and a.cache.frame_offset == 2
    )
    assert a.cache.visible[0, :14].all() and not a.cache.visible[0, 14:16].any()
    assert b.cache.valid_length == 0
    torch.testing.assert_close(
        first, b.generate_segment((2, 2, 2), cond), atol=0, rtol=0
    )
    state = a.generator.get_state().clone()
    before = [{key: x.clone() for key, x in layer.items()} for layer in a.cache.layers]
    with pytest.raises(ValueError, match="shape"):
        a.generate_segment((2, 2, 2), cond, initial_latents=torch.zeros(1))
    assert torch.equal(state, a.generator.get_state()) and a.cache.valid_length == 16
    for old, new in zip(before, a.cache.layers):
        assert all(torch.equal(old[key], new[key]) for key in old)
    second = a.generate_segment((2, 2, 2), cond)
    assert a.cache.valid_length == 32 and a.cache.frame_offset == 4
    assert not torch.equal(first, second)
    history = a.cache.history_mask(True)
    assert not history[:, 10:16].any() and not history[:, 26:32].any()
    assert a.cache.visible[:, 10:14].all()  # reads do not mutate stored visibility
    with pytest.raises(ValueError, match="capacity"):
        a.generate_segment((2, 2, 2), cond)
    a.reset()
    torch.testing.assert_close(
        first, a.generate_segment((2, 2, 2), cond), atol=0, rtol=0
    )


@torch.inference_mode()
def test_invalid_conditions_and_failed_compute_do_not_publish(monkeypatch):
    model = tiny_model()
    session = NomasLatentSession(model, cache_capacity=32, seed=42, steps=2)
    state = session.generator.get_state().clone()
    cond = condition()
    old_forward = model.double_blocks[1].forward
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Two successful Euler passes, then fail halfway through cache collection.
        if calls == 3:
            raise RuntimeError("injected layer failure")
        return old_forward(*args, **kwargs)

    monkeypatch.setattr(model.double_blocks[1], "forward", fail)
    with pytest.raises(RuntimeError, match="injected"):
        session.generate_segment((2, 2, 2), cond)
    assert session.cache.valid_length == 0 and not session.cache.visible.any()
    assert torch.equal(state, session.generator.get_state())
    monkeypatch.setattr(model.double_blocks[1], "forward", old_forward)
    assert torch.isfinite(session.generate_segment((2, 2, 2), cond)).all()


@torch.inference_mode()
def test_conditions_are_effective_and_meta_loading_restores_buffers():
    model = tiny_model()
    cond = condition()
    latent = torch.randn(1, 32, 2, 2, 2)
    cache = model.allocate_cache(64)
    kwargs = dict(
        hidden_states=latent,
        timestep=torch.tensor([600.0]),
        encoder_hidden_states=cond.text_states,
        text_mask=cond.text_mask,
        camera_embedding=cond.camera_embedding,
        camera_has_mask=cond.camera_has_mask,
        text_offsets=cond.text_offsets,
        video_windows=cond.video_windows,
        cache=cache,
    )
    baseline, entries = model.predict(**kwargs, collect_cache=True)
    assert baseline.abs().max() > 0.01
    for name in ("hidden_states", "encoder_hidden_states", "camera_embedding"):
        changed, _ = model.predict(**(kwargs | {name: kwargs[name] + 0.5}))
        assert not torch.equal(baseline, changed), name
    no_camera, _ = model.predict(**(kwargs | {"camera_has_mask": torch.zeros(1)}))
    assert not torch.equal(baseline, no_camera)
    cache.commit(entries, cond.text_mask, 8, 2, 2)
    history, _ = model.predict(**kwargs)
    assert not torch.equal(baseline, history)
    cache.reset()
    with torch.device("meta"):
        restored = NomasTransformer3DModel(
            NomasConfig(arch_config=model.config), attention_backend="sdpa"
        )
    restored.load_state_dict(model.state_dict(), strict=True, assign=True)
    restored.post_load_weights()
    restored.eval().requires_grad_(False)
    result, _ = restored.predict(**kwargs)
    torch.testing.assert_close(result, baseline, atol=0, rtol=0)


@pytest.mark.parametrize(
    "mask,offsets,windows",
    [
        ([[False] * 8], None, None),
        ([[True, False] * 4], None, None),
        ([[True] * 8], (2, 5, 7), ((0, 2, 0),)),
        ([[True] * 8], (2, 5, 8), ((0, 1, 0),)),
        ([[True] * 8], (2, 5, 8), ((0, 2, 5),)),
    ],
)
def test_invalid_attention_layout_rejected(mask, offsets, windows):
    with pytest.raises(ValueError):
        NomasAttentionPlan(
            torch.tensor(mask),
            torch.zeros(1, 0, dtype=torch.bool),
            2,
            4,
            offsets,
            windows,
            "sdpa",
        )


def test_model_registry_and_full_parameter_count():
    from sglang.multimodal_gen.runtime.models.registry import (
        _discover_and_register_models,
    )

    assert "NomasTransformer3DModel" in _discover_and_register_models()
    with torch.device("meta"):
        model = NomasTransformer3DModel(NomasConfig())
    assert sum(p.numel() for p in model.parameters()) == 8_553_434_592


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FlexAttention")
@torch.inference_mode()
def test_growing_history_across_twelve_segments():
    # Exercise more distinct history lengths than Dynamo's default recompile
    # limit; an eager fallback would materialize a dense mask on long rollouts.
    model = tiny_model("cuda", "flex").bfloat16()
    session = NomasLatentSession(model, cache_capacity=192, seed=42, steps=1)
    cond = condition("cuda", torch.bfloat16)
    for _ in range(12):
        assert torch.isfinite(session.generate_segment((2, 2, 2), cond)).all()
    assert session.cache.valid_length == 192 and session.cache.frame_offset == 24


@torch.inference_mode()
def test_final_segment_skips_cache_forward_and_noise(monkeypatch):
    model = tiny_model()
    calls = []
    predict = model.predict

    def record(*args, **kwargs):
        calls.append(kwargs.get("collect_cache", False))
        return predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", record)
    cached = NomasLatentSession(model, cache_capacity=16, seed=9, steps=2)
    final = NomasLatentSession(model, cache_capacity=0, seed=9, steps=2)
    initial = torch.randn(1, 32, 2, 2, 2)
    before = final.generator.get_state().clone()
    actual = final.generate_segment(
        (2, 2, 2), condition(), initial_latents=initial, cache_segment=False
    )
    assert calls == [False, False]
    assert final.cache.valid_length == 0 and final.cache.frame_offset == 0
    assert torch.equal(before, final.generator.get_state())
    expected = cached.generate_segment((2, 2, 2), condition(), initial_latents=initial)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert calls[-3:] == [False, False, True]
    with pytest.raises(RuntimeError, match="ended the session"):
        final.generate_segment((2, 2, 2), condition(), cache_segment=False)
    final.reset()
    torch.testing.assert_close(
        final.generate_segment(
            (2, 2, 2), condition(), initial_latents=initial, cache_segment=False
        ),
        actual,
        atol=0,
        rtol=0,
    )


@torch.inference_mode()
def test_action_head_only_without_intra_segment_windows():
    from msgspec.structs import replace

    model = tiny_model()
    session = NomasLatentSession(model, cache_capacity=16, seed=3, steps=1)
    cond = replace(condition(), text_offsets=None, video_windows=None, action_start=2)
    session.generate_segment((2, 2, 2), cond)
    history = session.cache.history_mask(True)
    assert history[0, :10].all()
    assert not history[0, 10:].any()
    assert session.cache.history_mask(False)[0, 10:14].all()


@pytest.mark.parametrize("evict", [False, True])
@torch.inference_mode()
def test_segment_context_reuse_matches_rebuilding_every_prediction(monkeypatch, evict):
    model = tiny_model()
    cond = condition()
    options = dict(
        cache_capacity=32,
        seed=7,
        steps=3,
        max_cache_segments=1 if evict else None,
    )
    reference = NomasLatentSession(model, **options)
    optimized = NomasLatentSession(model, **options)
    for session in (reference, optimized):
        session.generate_segment((2, 2, 2), cond, evict=evict)
    predict = model.predict

    def rebuild(*args, **kwargs):
        kwargs.pop("_context", None)
        return predict(*args, **kwargs)

    monkeypatch.setattr(model, "predict", rebuild)
    expected, expected_trace = reference.generate_segment(
        (2, 2, 2), cond, evict=evict, return_trace=True
    )
    monkeypatch.setattr(model, "predict", predict)
    prepare = model.prepare_segment_context
    contexts = []

    def record(*args, **kwargs):
        context = prepare(*args, **kwargs)
        contexts.append(context)
        return context

    monkeypatch.setattr(model, "prepare_segment_context", record)
    actual, trace = optimized.generate_segment(
        (2, 2, 2), cond, evict=evict, return_trace=True
    )
    assert len(contexts) == (2 if evict else 1)
    if evict:
        assert contexts[0].rope is contexts[1].rope
        assert contexts[0].history_length == 16
        assert contexts[1].history_length == 0
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual_step, expected_step in zip(trace, expected_trace, strict=True):
        for key in ("prediction", "latent"):
            torch.testing.assert_close(
                actual_step[key], expected_step[key], atol=0, rtol=0
            )
    for actual_layer, expected_layer in zip(
        optimized.cache.layers, reference.cache.layers, strict=True
    ):
        for key in ("k", "v"):
            torch.testing.assert_close(
                actual_layer[key], expected_layer[key], atol=0, rtol=0
            )
    assert torch.equal(optimized.generator.get_state(), reference.generator.get_state())


@pytest.mark.parametrize("bad_tensor", ["latent", "k", "v"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_segment_finite_reduction_checks_every_tensor(bad_tensor, bad_value):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
        _segment_is_finite,
    )

    latent = torch.ones(1)
    entries = [{"k": torch.ones(2), "v": torch.ones(2)} for _ in range(2)]
    assert _segment_is_finite(latent, entries)
    target = latent if bad_tensor == "latent" else entries[-1][bad_tensor]
    target[0] = bad_value
    assert not _segment_is_finite(latent, entries)


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


def test_local_token_range_requires_divisible_tokens():
    from sglang.multimodal_gen.runtime.models.dits.nomas_layers import (
        ParallelInfo,
        local_token_range,
    )

    assert local_token_range(16, ParallelInfo(sp_size=2, sp_rank=1)) == (8, 16)
    with pytest.raises(ValueError, match="not divisible"):
        local_token_range(15, ParallelInfo(sp_size=2))


def _run_shared_memory_budget(rank, port):
    """Use a real CPU collective without allocating on either GPU."""
    from unittest.mock import patch

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas import (
        stages,
    )

    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        group = SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD)
        with (
            patch.object(torch.cuda, "mem_get_info", return_value=(16 + rank * 16, 64)),
            patch.object(stages, "model_parallel_is_initialized", return_value=True),
            patch.object(stages, "get_replica_group", return_value=group),
        ):
            assert stages._free_bytes(torch.device("cuda", rank)) == 16
    finally:
        dist.destroy_process_group()


def test_memory_budget_uses_least_free_rank(monkeypatch):
    monkeypatch.setenv("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
    mp.spawn(
        _run_shared_memory_budget, args=(get_free_port_below_ephemeral(),), nprocs=2
    )
