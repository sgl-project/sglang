# SPDX-License-Identifier: Apache-2.0
"""Weightless Nomas tests: DiT segment session, pipeline stages and request contracts."""

import json
import tempfile
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.dits.nomas import NomasArchConfig, NomasConfig
from sglang.multimodal_gen.configs.models.vaes.nomas import (
    NomasVAEArchConfig,
    NomasVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.nomas import (
    NomasPipelineConfig,
    nomas_postprocess_text,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nomas_fp8_config import (
    CHECKPOINT_FORMAT,
    inspect_nomas_fp8_checkpoint,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.nomas_int8_config import (
    NomasInt8Config,
    inspect_nomas_int8_checkpoint,
)
from sglang.multimodal_gen.runtime.layers.quantization.nomas_fp8 import (
    NomasFP8Linear,
    quantize_nomas_fp8_activation,
)
from sglang.multimodal_gen.runtime.layers.quantization.nomas_int8 import (
    NomasInt8Linear,
    dequantize_nomas_int8,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel
from sglang.multimodal_gen.runtime.models.dits.nomas_kv_cache import (
    NomasAttentionPlan,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.models.vaes.nomas import (
    GenExpertArch,
    NomasPixelUMMDecoder,
    window_starts,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
    NomasLatentSession,
    NomasSegmentCondition,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

# The framework's parallel linears resolve the TP group at construction.
pytestmark = pytest.mark.usefixtures("single_process_model_parallel")

# On ROCm, Inductor's fused kernels are not bitwise equal to eager, and the
# dynamic-shape create_block_mask compile fails, so these checks are CUDA-only.
is_nvidia_cuda = torch.cuda.is_available() and current_platform.is_cuda()


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


@torch.inference_mode()
def test_int8_checkpoint_roundtrip_matches_dequantized_dit(tmp_path):
    """Official key aliases and mixed-dtype loading must retain full DiT output."""
    from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
    from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan

    reference = tiny_model().bfloat16()
    state = reference.state_dict()
    targets = (
        "time_in.mlp.0",
        "txt_in.input_embedder",
        "double_blocks.0.img_attn_q",
        "double_blocks.0.img_mlp.fc_in",
        "double_blocks.0.img_mlp.fc_out",
        "double_blocks.0.img_mod.linear",
    )
    manifest = {}
    generator = torch.Generator().manual_seed(789)
    for name in targets:
        weight = state.pop(name + ".weight")
        out_features, in_features = weight.shape
        scales = (
            torch.rand(out_features, in_features // 128, generator=generator) * 0.001
            + 0.0001
        )
        quantized = torch.randint(
            -127, 128, weight.shape, generator=generator, dtype=torch.int8
        )
        dequantized = (
            (quantized.float().reshape(out_features, -1, 128) * scales[..., None])
            .reshape_as(weight)
            .bfloat16()
        )
        dict(reference.named_parameters())[name + ".weight"].copy_(dequantized)
        source_name = name.replace(".fc_in", ".fc1").replace(".fc_out", ".fc2")
        state[source_name + ".weight_q"] = quantized
        state[source_name + ".weight_scale"] = scales
        manifest[source_name] = list(weight.shape)
    path = tmp_path / "nomas-int8.safetensors"
    save_file(
        state,
        path,
        metadata={
            "format": "nomas-int8-bf16-single-v1",
            "group_size": "128",
            "quantization": "symmetric_int8_absmax",
            "compute_dtype": "bfloat16",
            "rounding": "half_away_from_zero",
            "model_type": "HY-Nomas-G/2-32324-MMDiT",
            "layers": json.dumps(manifest),
        },
    )
    quant = inspect_nomas_int8_checkpoint(path)
    loaded = maybe_load_fsdp_model(
        NomasTransformer3DModel,
        {
            "config": NomasConfig(arch_config=reference.config),
            "quant_config": quant,
            "attention_backend": "sdpa",
        },
        [str(path)],
        torch.device("cpu"),
        1,
        1,
        None,
        torch.float32,
        component_starts_on_cpu=True,
        pin_cpu_memory=False,
        strict=True,
        weight_load_plan=WeightLoadPlan(checkpoint_load_device=torch.device("cpu")),
    )
    scales_before = {
        name: module.weight_scale.clone()
        for name, module in loaded.named_modules()
        if hasattr(module, "weight_scale")
    }
    # The loader sets mixed parameter dtypes; residency only moves devices.
    loaded.to("cpu")
    for name, scales in scales_before.items():
        layer = loaded.get_submodule(name)
        assert layer.weight.dtype == torch.int8
        assert layer.weight_scale.dtype == torch.float32
        assert torch.equal(layer.weight_scale, scales)
    latent = torch.randn(1, 32, 2, 2, 2, generator=generator).bfloat16()
    cond = condition(dtype=torch.bfloat16)
    timestep = torch.tensor([0.5], dtype=torch.bfloat16)

    def predict(model):
        return model(
            latent,
            cond.text_states,
            timestep,
            text_mask=cond.text_mask,
            cache=model.allocate_cache(32),
        )

    torch.testing.assert_close(predict(loaded), predict(reference), rtol=0, atol=0)


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


def tiny_decoder():
    torch.manual_seed(5)
    arch = NomasVAEArchConfig(
        hidden_size=64,
        intermediate_size=128,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=32,
    )
    return NomasPixelUMMDecoder(NomasVAEConfig(arch_config=arch)).eval()


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


@pytest.mark.skipif(not is_nvidia_cuda, reason="fused kernels need CUDA")
@torch.inference_mode()
def test_fused_qk_norm_rope_and_modulate_cast_are_bitwise_exact():
    from sglang.multimodal_gen.runtime.models.dits import nomas_layers as layers

    torch.manual_seed(3)
    for heads, dim, tokens in ((16, 128, 2000), (16, 64, 1500)):
        norm = layers.RMSNorm(dim).cuda().bfloat16()
        norm.weight.normal_(1, 0.1)
        x = torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16) * 3
        angles = torch.randn(tokens, dim, device="cuda")
        freqs = (angles.cos(), angles.sin())
        expected = layers._rotate(
            norm(x).to(torch.bfloat16), *layers._rope_tables(x, freqs)
        )
        assert torch.equal(
            layers.rms_norm_rotate(norm, x, freqs, torch.bfloat16), expected
        )
        assert torch.equal(
            layers.apply_rotary_emb_single(x, freqs),
            layers._rotate(x, *layers._rope_tables(x, freqs)),
        )
    x = torch.randn(1, 3000, 2048, device="cuda") * 2
    shift, scale = torch.randn(2, 1, 2048, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fused = layers.modulate_for_linear(x, shift, scale)
    assert fused.dtype == torch.bfloat16
    assert torch.equal(fused, layers.modulate(x, shift, scale).to(torch.bfloat16))


@pytest.mark.skipif(not is_nvidia_cuda, reason="fused kernels need CUDA")
@torch.inference_mode()
def test_pixelumm_fused_norm_and_swiglu_are_bitwise_exact():
    from sglang.multimodal_gen.runtime.models.vaes import nomas as vae

    torch.manual_seed(9)
    norm = vae.RMSNorm(4096, 1e-6).cuda().bfloat16()
    norm.weight.normal_(1, 0.1)
    x = torch.randn(1, 3000, 4096, device="cuda", dtype=torch.bfloat16) * 3
    xf = x.float()
    expected = norm.weight * (
        xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + norm.eps)
    ).to(x.dtype)
    assert torch.equal(norm(x), expected)
    gate, up = (
        torch.randn(1, 3000, 1024, device="cuda", dtype=torch.bfloat16) * 4
        for _ in range(2)
    )
    assert torch.equal(vae._compiled_silu_mul()(gate, up), F.silu(gate) * up)


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
    cache = {"k": k[:, :3], "v": v[:, :3]}
    torch.testing.assert_close(
        plan(q, k[:, 3:], v[:, 3:], cache, 3).flatten(2), result, atol=0, rtol=0
    )


requires_hopper = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="FA3 segment attention is enabled on Hopper",
)


@requires_hopper
@torch.inference_mode()
def test_fa3_segment_attention_matches_dense_mask():
    """Visible-prefix FA3 plus per-action LSE merge equals the masked reference."""
    torch.manual_seed(5)
    frames, spatial, text, valid = 6, 64, 96, 72
    # Two history segments: video visible, padding and action tails hidden.
    history = torch.zeros(1, 2 * (frames * spatial + text), dtype=torch.bool)
    for start in (0, frames * spatial + text):
        history[:, start : start + frames * spatial + 20] = True
    history = history.cuda()
    text_mask = (torch.arange(text, device="cuda") < valid)[None]
    # Head [0, 24), action 0 [24, 40), empty action 1, action 2 [40, 72);
    # frames 2-3 are a gap that sees no action text.
    offsets, windows = (24, 40, 40, 72), ((0, 2, 0), (2, 4, -1), (4, 6, 2))
    n, h = frames * spatial + text, history.shape[1]
    q = torch.randn(1, n, 4, 128, device="cuda", dtype=torch.bfloat16)
    k, v = (
        torch.randn(1, h + n, 4, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(2)
    )
    cache = {"k": k[:, :h], "v": v[:, :h]}
    plans = {
        backend: NomasAttentionPlan(
            text_mask, history, frames, spatial, offsets, windows, backend
        )
        for backend in ("fa3", "sdpa")
    }
    expected = plans["sdpa"](
        q.float(),
        k[:, h:].float(),
        v[:, h:].float(),
        {key: x.float() for key, x in cache.items()},
        h,
    )
    actual = plans["fa3"](q, k[:, h:], v[:, h:], cache, h)
    torch.testing.assert_close(actual.float(), expected, atol=1e-2, rtol=1e-2)
    assert not actual[:, frames * spatial + valid :].any()


@torch.inference_mode()
def test_session_commit_reset_isolation_capacity_and_retry():
    model = tiny_model()
    a = NomasLatentSession(model, cache_capacity=32, seed=42, steps=2)
    cond = condition()
    first = a.generate_segment((2, 2, 2), cond)
    assert (
        first.abs().max() > 0
        and a.cache.valid_length == 16
        and a.cache.frame_offset == 2
    )
    assert a.cache.visible[0, :14].all() and not a.cache.visible[0, 14:16].any()
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
    kwargs = {
        "hidden_states": latent,
        "timestep": torch.tensor([600.0]),
        "encoder_hidden_states": cond.text_states,
        "text_mask": cond.text_mask,
        "camera_embedding": cond.camera_embedding,
        "camera_has_mask": cond.camera_has_mask,
        "text_offsets": cond.text_offsets,
        "video_windows": cond.video_windows,
        "cache": cache,
    }
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


def test_registration_cli_routing_and_full_parameter_count():
    from sglang.cli.utils import get_is_diffusion_model
    from sglang.multimodal_gen.runtime.models.registry import (
        _discover_and_register_models,
    )

    # The official repo has no model_index.json; detection relies on the registry.
    assert get_is_diffusion_model("tencent/Hy-Nomas")
    assert "NomasTransformer3DModel" in _discover_and_register_models()
    with torch.device("meta"):
        model = NomasTransformer3DModel(NomasConfig())
    assert sum(p.numel() for p in model.parameters()) == 8_553_434_592


@pytest.mark.skipif(not is_nvidia_cuda, reason="CUDA FlexAttention")
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
    options = {
        "cache_capacity": 32,
        "seed": 7,
        "steps": 3,
        "max_cache_segments": 1 if evict else None,
    }
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


def test_latent_shapes_and_batching():
    config = NomasPipelineConfig()
    request = SimpleNamespace(height=64, width=96, num_frames=48, guidance_scale=1)
    assert config.prepare_latent_shape(request, 1, 12) == (1, 32, 12, 2, 3)
    request.num_frames = 49
    assert config.prepare_latent_shape(request, 1, 13) == (1, 32, 13, 2, 3)
    request.num_frames = 48
    with pytest.raises(ValueError, match="batch size"):
        config.prepare_latent_shape(request, 2, 12)
    assert not config.supports_dynamic_batching()


@pytest.mark.parametrize("overrides", [{}, {"generator_device": "cuda"}])
def test_http_requests_keep_nomas_fields_and_cpu_noise(monkeypatch, overrides):
    from sglang.multimodal_gen.configs.sample.nomas import NomasSamplingParams
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.openai import video_api
    from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
        VideoGenerationsRequest,
    )
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        request_model_kwargs,
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
    # Nomas-only fields reach Nomas sampling params and no other model's.
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
    assert {f.name for f in fields(GenExpertArch)} <= {
        f.name for f in fields(NomasVAEArchConfig)
    }
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
def test_pixelumm_fused_qk_norm_mrope_is_bitwise_exact(tokens, heads, batch):
    from sglang.multimodal_gen.runtime.models.vaes.nomas import (
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
    normalized = attention._split_norm(
        x.float(), attention.q_norm_t, attention.q_norm_hw
    )
    expected = apply_mrope(normalized, cos, sin, (64, 32, 32)).to(x.dtype)
    actual = attention._prepare_qk(
        x, cos, sin, attention.q_norm_t, attention.q_norm_hw, x.dtype
    )
    assert torch.equal(actual, expected)


def test_scheduler_loader_uses_effective_shift(tmp_path):

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
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.request_validation import (
        action_video_windows,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.stages import (
        segment_camera_condition,
    )

    drive = {"start_time": 0, "end_time": "00:00.5"}
    turn = {"start_time": 0.5, "end_time": 1}
    split, tail_gap = ((0, 3, 0), (3, 6, 1)), ((0, 3, 0), (3, 6, -1))
    for actions, expected in (
        ([drive, turn], split),
        # A null timestamp is 0 seconds upstream; here it empties the second span.
        ([drive, {"start_time": 0.5, "end_time": None}], tail_gap),
        ([drive], tail_gap),
        ([{"start_time": 0, "end_time": 9}], None),
        ([drive, {"start_time": 0.25, "end_time": 1}], None),
        ([drive, {"action": "no timing"}], None),
        ([{"start_time": 1, "end_time": 0.5}], None),
    ):
        assert action_video_windows(actions, 6, 24) == expected, actions

    plain = NomasSegmentCondition(
        torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)
    )
    assert segment_camera_condition(plain, None, 0, 2) is plain
    sliced = segment_camera_condition(plain, torch.arange(5.0).view(1, 5, 1), 1, 3)
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


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="requires SM89+ FP8 tensor cores",
)
@pytest.mark.parametrize("tokens", [1, 127, 128, 129, 257])
@torch.inference_mode()
def test_pixelumm_fp8_linear_quantized_reference(tokens):

    torch.manual_seed(17)
    layer = NomasFP8Linear(256, 256, device="cuda").eval()
    weight = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16) / 16
    layer.fp8_weight, weight_scales = quantize_nomas_fp8_activation(weight)
    layer.fp8_scale = weight_scales[::128].contiguous()
    # Noncontiguous, higher-rank input also exercises flatten/reshape behavior.
    x = torch.randn(1, 256, tokens, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    output = layer(x)
    q, scales = quantize_nomas_fp8_activation(x.reshape(tokens, 256))
    a = q.float() * scales.repeat_interleave(128, dim=1)
    b = layer.fp8_weight.float() * layer.fp8_scale.repeat_interleave(
        128, dim=0
    ).repeat_interleave(128, dim=1)
    reference = (a @ b.T).bfloat16().reshape(1, tokens, 256)
    assert output.dtype == torch.bfloat16
    # FP8 GEMM scales each K block before FP32 accumulation; BF16 rounding
    # can differ from multiplying fully dequantized operands once.
    torch.testing.assert_close(output, reference, rtol=0.008, atol=0.008)


class TestNomasFP8(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.safetensors"
        self.config = NomasVAEConfig(
            arch_config=NomasVAEArchConfig(
                hidden_size=128,
                intermediate_size=256,
                num_layers=1,
                num_heads=4,
                num_kv_heads=2,
                head_dim=32,
            )
        )
        self.original = NomasPixelUMMDecoder(self.config).bfloat16().eval()
        self.state = dict(self.original.state_dict())
        self.sources = []
        for name in self.original.fp8_quantized_layer_names():
            weight = self.state.pop(name + ".weight")
            self.state[name + ".fp8_weight"] = weight.to(torch.float8_e4m3fn)
            self.state[name + ".fp8_scale"] = torch.full(
                (weight.shape[0] // 128, weight.shape[1] // 128), 0.01003
            )
            self.sources.append(name + ".weight")
        # Official complete files contain an encoder unused by generation.
        self.state["encoder.project_in.conv.conv.bias"] = torch.zeros(
            32, dtype=torch.bfloat16
        )
        self.metadata = {
            "format": CHECKPOINT_FORMAT,
            "block_size": "128",
            "weight_dtype": "float8_e4m3fn",
            "scale_dtype": "float32",
            "unquantized_float_dtype": "bfloat16",
            "quantized_layers": "5",
            "quantized_source_keys": json.dumps(self.sources),
        }

    def save(self):
        save_file(self.state, self.path, metadata=self.metadata)

    def load(self, direct=False):
        self.save()
        pipeline = SimpleNamespace(
            vae_config=self.config,
            vae_precision="bf16",
            native_only_components=("vae",),
        )
        args = SimpleNamespace(
            pipeline_config=pipeline,
            component_precisions={},
            component_weights_paths={},
            model_paths={},
            should_direct_gpu_weight_load_component=lambda _: direct,
            should_start_component_on_cpu=lambda _: True,
        )
        loader = VAELoader()
        # The config file is external to the weights-only release.
        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader.get_diffusers_component_config",
                return_value={"_class_name": "NomasPixelUMMDecoder"},
            ),
            patch.object(loader, "target_device", return_value=torch.device("cpu")),
        ):
            return loader.load_customized(str(self.path), args, "vae").eval()

    def test_complete_checkpoint_load_and_offload_dtype_roundtrip(self):
        """Meta assignment must retain FP8/FP32 and ignore only the unused encoder."""
        for direct in (False, True):
            with self.subTest(direct=direct):
                model = self.load(direct)
                model.bfloat16().to("cpu")
                state = model.state_dict()
                self.assertFalse(any(t.is_meta for t in state.values()))
                for name, value in state.items():
                    expected = self.state[name]
                    self.assertEqual(value.dtype, expected.dtype)
                    torch.testing.assert_close(
                        value.float(), expected.float(), rtol=0, atol=0
                    )
                self.assertEqual(
                    sum(isinstance(m, NomasFP8Linear) for m in model.modules()), 5
                )

    def test_rejects_missing_scale_and_invalid_scale_values(self):
        """Partial or corrupt serialized blocks must fail before inference."""
        name = self.sources[0].removesuffix(".weight") + ".fp8_scale"
        scale = self.state.pop(name)
        self.save()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            inspect_nomas_fp8_checkpoint([self.path])
        self.state[name] = scale.fill_(float("nan"))
        self.save()
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            inspect_nomas_fp8_checkpoint([self.path])

    def test_bf16_checkpoint_ignores_encoder_but_keeps_decoder_strict(self):
        """Encoder filtering also applies to ordinary and direct BF16 loads."""
        original = dict(self.original.state_dict())
        original["encoder.project_in.conv.conv.bias"] = torch.zeros(
            32, dtype=torch.bfloat16
        )
        self.metadata = {}
        for direct in (False, True):
            with self.subTest(direct=direct):
                self.state = dict(original)
                model = self.load(direct)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, original[name], rtol=0, atol=0)

                del self.state["decoder.layers.0.attn.k_proj.weight"]
                with self.assertRaisesRegex((RuntimeError, ValueError), "[Mm]issing"):
                    self.load(direct)

                self.state = dict(original)
                self.state["decoder.unknown.weight"] = torch.zeros(
                    1, dtype=torch.bfloat16
                )
                with self.assertRaisesRegex(
                    (RuntimeError, ValueError), "[Uu]nexpected"
                ):
                    self.load(direct)

    def test_rejects_incomplete_manifest(self):
        """A self-consistent partial manifest must not silently leave BF16 projections."""
        self.metadata["quantized_source_keys"] = json.dumps(self.sources[:-1])
        self.metadata["quantized_layers"] = "4"
        name = self.sources[-1].removesuffix(".weight")
        del self.state[name + ".fp8_weight"], self.state[name + ".fp8_scale"]
        with self.assertRaisesRegex(ValueError, "manifest"):
            self.load()

    def test_rejects_missing_decoder_parameters(self):
        """Ignoring the unused encoder must not weaken decoder completeness checks."""
        del self.state["decoder.layers.0.attn.k_proj.weight"]
        with self.assertRaisesRegex(ValueError, "missing tensors"):
            self.load()

    def test_activation_scale_is_shared_across_128_tokens(self):
        """Token 128 starts a new scale block, including a partial final block."""
        for tokens in (127, 128, 129):
            x = torch.ones(tokens, 256, dtype=torch.bfloat16)
            x[0, 0] = 224
            x[-1, 128] = 448
            q, scales = quantize_nomas_fp8_activation(x)
            torch.testing.assert_close(scales[0, 0], torch.tensor(0.5))
            torch.testing.assert_close(q[0, 0].float(), torch.tensor(448.0))
            torch.testing.assert_close(scales[-1, 1], torch.tensor(1.0))
            if tokens == 129:
                torch.testing.assert_close(scales[-1, 0], torch.tensor(1 / 448))
                torch.testing.assert_close(q[-1, 0].float(), torch.tensor(448.0))
            else:
                torch.testing.assert_close(scales[:, 0], torch.full((tokens,), 0.5))


class TestNomasInt8Linear(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "weights.safetensors"
        self.name = "double_blocks.0.img_attn_q"
        self.weight = torch.full((2, 130), 127, dtype=torch.int8)
        self.scales = torch.tensor([[0.01003, 0.02007], [0.03009, 0.04013]])
        self.bias = torch.tensor([0.5, -0.5], dtype=torch.bfloat16)
        self.metadata = {
            "format": "nomas-int8-bf16-single-v1",
            "group_size": "128",
            "quantization": "symmetric_int8_absmax",
            "compute_dtype": "bfloat16",
            "rounding": "half_away_from_zero",
            "model_type": "HY-Nomas-G/2-32324-MMDiT",
            "layers": json.dumps({self.name: [2, 130]}),
        }
        self.tensors = {
            f"{self.name}.weight_q": self.weight,
            f"{self.name}.weight_scale": self.scales,
            f"{self.name}.bias": self.bias,
        }

    def copy_parameters(self, layer):
        for name in ("weight", "weight_scale", "bias"):
            value = self.scales if name == "weight_scale" else getattr(self, name)
            getattr(layer, name).data.copy_(value)

    def load(self):
        save_file(self.tensors, self.path, metadata=self.metadata)
        config = inspect_nomas_int8_checkpoint(self.path)
        out_features, in_features = config.layers[self.name]
        with safe_open(str(self.path), framework="pt", device="cpu") as reader:
            checkpoint_keys = reader.keys()
            layer = NomasInt8Linear(
                in_features,
                out_features,
                bias=f"{self.name}.bias" in checkpoint_keys,
                params_dtype=torch.bfloat16,
                quant_config=config,
                prefix=self.name,
            )
            for target, source in (
                ("weight", "weight_q"),
                ("weight_scale", "weight_scale"),
                ("bias", "bias"),
            ):
                param = getattr(layer, target)
                if param is not None:
                    param.weight_loader(
                        param, reader.get_tensor(f"{self.name}.{source}")
                    )
        layer.quant_method.process_weights_after_loading(layer)
        return layer

    def test_group_boundary_and_partial_final_group(self):
        """Inputs 127/128 straddle groups; a partial final group is not dropped."""
        layer = self.load()
        inputs = torch.zeros(2, 130, dtype=torch.bfloat16)
        inputs[0, 127] = 1
        inputs[1, 128] = 1
        expected = (127 * self.scales).bfloat16().T + self.bias
        torch.testing.assert_close(layer(inputs), expected, rtol=0, atol=0)

    def test_dtype_moves_preserve_scale_precision_and_output(self):
        """Rounding scales to BF16 before multiplication changes reconstructed weights."""
        layer = self.load()
        scales = layer.weight_scale.clone()
        reference = dequantize_nomas_int8(layer.weight, layer.weight_scale)
        early_rounded = (127 * scales.bfloat16()).bfloat16()
        self.assertFalse(torch.equal(early_rounded[:, 0], reference[:, 0]))
        layer.to(dtype=torch.bfloat16)
        self.assertEqual(layer.weight.dtype, torch.int8)
        self.assertEqual(layer.weight_scale.dtype, torch.float32)
        torch.testing.assert_close(layer.weight_scale, scales, rtol=0, atol=0)
        torch.testing.assert_close(
            dequantize_nomas_int8(layer.weight, layer.weight_scale),
            reference,
            rtol=0,
            atol=0,
        )
        self.assertEqual(set(layer.state_dict()), {"weight", "weight_scale", "bias"})

    def test_component_dtype_moves_preserve_parallel_scales(self):
        """Component residency casts must preserve FP32 scales on every linear."""
        inputs = torch.ones(1, 130, dtype=torch.bfloat16)
        expected = self.load()(inputs)
        for cls in (ColumnParallelLinear, RowParallelLinear):
            with self.subTest(layer=cls.__name__):
                model = torch.nn.Module()
                model.projection = cls(
                    130,
                    2,
                    bias=True,
                    params_dtype=torch.bfloat16,
                    tp_group=SimpleNamespace(world_size=1, rank_in_group=0),
                )
                NomasInt8Config({"projection": [2, 130]}).configure_model(model)
                layer = model.projection
                self.assertIsInstance(layer, cls)
                self.copy_parameters(layer)
                loader = layer.weight_scale.weight_loader
                for dtype in (torch.bfloat16, torch.float16, torch.bfloat16):
                    model.to(device="cpu", dtype=dtype)
                    self.assertEqual(layer.weight.dtype, torch.int8)
                    self.assertEqual(layer.weight_scale.dtype, torch.float32)
                    torch.testing.assert_close(
                        layer.weight_scale, self.scales, rtol=0, atol=0
                    )
                    self.assertEqual(layer.weight_scale.weight_loader, loader)
                torch.testing.assert_close(layer(inputs)[0], expected, rtol=0, atol=0)

    def test_reject_incompatible_checkpoint(self):
        for key, value in (("group_size", "64"), ("compute_dtype", "float16")):
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = value
                with self.assertRaisesRegex(ValueError, key):
                    self.load()
                self.metadata[key] = original
        self.metadata["layers"] = json.dumps({self.name: [2, 256]})
        with self.assertRaisesRegex(ValueError, "shape/dtype mismatch"):
            self.load()

    def test_reject_invalid_scales_and_inputs(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                scales = self.scales.clone()
                scales[0, 0] = value
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    self.tensors[f"{self.name}.weight_scale"] = scales
                    self.load()
        self.tensors[f"{self.name}.weight_scale"] = self.scales
        layer = self.load()
        with self.assertRaisesRegex(TypeError, "BF16 input"):
            layer(torch.zeros(1, 130))

    def test_manifest_configures_only_declared_layers(self):
        """Checkpoint metadata selects quantized modules without touching BF16 camera layers."""
        reference = self.load()
        config = inspect_nomas_int8_checkpoint(self.path)
        model = torch.nn.Module()
        model.double_blocks = torch.nn.ModuleList([torch.nn.Module()])
        model.double_blocks[0].img_attn_q = torch.nn.Linear(
            130, 2, dtype=torch.bfloat16
        )
        model.camera = torch.nn.Linear(64, 64, dtype=torch.bfloat16)
        camera = model.camera
        config.configure_model(model)
        projection = model.double_blocks[0].img_attn_q
        self.copy_parameters(projection)
        inputs = torch.ones(1, 130, dtype=torch.bfloat16)
        torch.testing.assert_close(
            projection(inputs), reference(inputs), rtol=0, atol=0
        )
        self.assertIs(model.camera, camera)
        self.assertEqual(projection.weight.dtype, torch.int8)

    def test_manifest_requires_complete_scale_pairs(self):
        self.tensors.pop(f"{self.name}.weight_scale")
        save_file(self.tensors, self.path, metadata=self.metadata)
        with self.assertRaisesRegex(ValueError, "manifest does not match"):
            inspect_nomas_int8_checkpoint(self.path)

    def test_parallel_weight_loaders_preserve_group_scales(self):
        """Row TP shards scales along groups; column TP shards along outputs."""
        weight = (torch.arange(4 * 256).reshape(4, 256) % 255 - 127).to(torch.int8)
        scales = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2) * 0.01003
        full = (
            (weight.float().reshape(4, 2, 128) * scales[..., None])
            .reshape(4, 256)
            .bfloat16()
        )
        inputs = torch.ones(3, 256, dtype=torch.bfloat16)
        for cls in (ColumnParallelLinear, RowParallelLinear):
            for rank in (0, 1):
                with self.subTest(layer=cls.__name__, rank=rank):
                    kwargs = (
                        {"gather_output": False}
                        if cls is ColumnParallelLinear
                        else {"input_is_parallel": True, "reduce_results": False}
                    )
                    layer = cls(
                        256,
                        4,
                        bias=False,
                        params_dtype=torch.bfloat16,
                        quant_config=NomasInt8Config({"projection": [4, 256]}),
                        prefix="projection",
                        tp_group=SimpleNamespace(world_size=2, rank_in_group=rank),
                        **kwargs,
                    )
                    layer.weight_loader(layer.weight, weight)
                    layer.weight_loader(layer.weight_scale, scales)
                    if cls is ColumnParallelLinear:
                        x, w = inputs, full[rank * 2 : (rank + 1) * 2]
                        expected_scales = scales[rank * 2 : (rank + 1) * 2]
                    else:
                        x, w = (
                            inputs[:, rank * 128 : (rank + 1) * 128],
                            full[:, rank * 128 : (rank + 1) * 128],
                        )
                        expected_scales = scales[:, rank : rank + 1]
                    torch.testing.assert_close(
                        layer.weight_scale, expected_scales, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        layer(x)[0], torch.nn.functional.linear(x, w), rtol=0, atol=0
                    )
        with self.assertRaisesRegex(ValueError, "align to 128"):
            RowParallelLinear(
                130,
                4,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=NomasInt8Config({"projection": [4, 130]}),
                prefix="projection",
                tp_group=SimpleNamespace(world_size=2, rank_in_group=0),
            )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
