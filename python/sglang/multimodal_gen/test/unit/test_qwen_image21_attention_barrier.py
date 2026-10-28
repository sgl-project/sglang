# SPDX-License-Identifier: Apache-2.0
"""CPU correctness tests for Qwen Image 2.1 attention-barrier tiling.

Run from the SGLang checkout that contains the candidate module, for example:
    PYTHONPATH=python python -m pytest -q \
        python/sglang/multimodal_gen/test/unit/test_qwen_image21_attention_barrier.py

These tests intentionally use FP64 CPU tensors and do not initialize distributed
state. They exercise the production module, not the standalone oracle.
"""

from unittest.mock import Mock

import pytest
import torch

from sglang.multimodal_gen.configs.models.vaes.qwenimage21 import (
    QwenImage21VAEArchConfig,
    QwenImage21VAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes import autoencoder_kl_qwenimage21 as q
from sglang.multimodal_gen.runtime.models.vaes import common


def tiny_vae(**arch_overrides):
    """Use the real five-stage VAE graph, but keep its channels tiny."""
    torch.manual_seed(42)
    arch = dict(
        in_channels=3,
        out_channels=3,
        base_dim=2,
        decoder_base_dim=2,
        z_dim=2,
        dim_mult=(1, 2, 4, 8, 8),
        num_res_blocks=2,
        is_residual=True,
    )
    arch.update(arch_overrides)
    model = (
        q.AutoencoderKLQwenImage21(
            QwenImage21VAEConfig(
                arch_config=QwenImage21VAEArchConfig(**arch),
                load_encoder=True,
                load_decoder=True,
                use_tiling=True,
                use_temporal_tiling=False,
                use_parallel_tiling=False,
            )
        )
        .double()
        .eval()
    )
    ratio = model.spatial_compression_ratio
    model.tile_sample_min_height = 24 * ratio
    model.tile_sample_min_width = 32 * ratio
    model.tile_sample_stride_height = 16 * ratio
    model.tile_sample_stride_width = 19 * ratio
    model.parallel_decode_mode = "none"
    return model


@pytest.fixture(autouse=True)
def disable_spatial_parallel_dispatch(monkeypatch):
    """Do not query or initialize distributed state in any test."""
    monkeypatch.setattr(
        common, "should_run_spatial_shard_parallel_decode", lambda *_a, **_k: False
    )
    monkeypatch.setattr(
        q, "should_run_spatial_shard_parallel_decode", lambda *_a, **_k: False
    )


def _full_prefix(model, source, *, encode):
    if encode:
        branch = model.encoder
        x = branch.conv_in(source)
        for block in branch.down_blocks:
            x = block(x)
    else:
        branch = model.decoder
        x = model.post_quant_conv(source)
        x = branch.conv_in(x)
    return branch.mid_block.resnets[0](x)


@pytest.mark.parametrize("encode", [True, False], ids=["encoder", "decoder"])
def test_prefix_assembly_native_full_attention_and_retiled_suffix(monkeypatch, encode):
    model = tiny_vae(dropout=0.25, attn_scales=(1.0, 0.5), patch_size=1)
    source = torch.randn(
        (1, 3, 1, 512, 608) if encode else (1, 2, 1, 23, 26),
        dtype=torch.float64,
        requires_grad=True,
    )
    branch = model.encoder if encode else model.decoder
    expected = model._encode(source) if encode else model._decode(source)
    with torch.no_grad():
        prefix_full = _full_prefix(model, source, encode=encode)
    monkeypatch.setattr(
        q.ParallelTiledVAE,
        "spatial_tiled_encode" if encode else "spatial_tiled_decode",
        Mock(side_effect=AssertionError("supported input must use context tiling")),
    )
    shape = tuple(prefix_full.shape[-2:])
    _, _, step_h, step_w = model._spatial_tile_geometry()
    cores = [
        ((a, min(a + step_h, shape[0])), (b, min(b + step_w, shape[1])))
        for a in range(0, shape[0], step_h)
        for b in range(0, shape[1], step_w)
    ]
    trace, windows, sdpa_calls = [], [], []
    original_prefix = model._context_prefix

    def prefix(value, window, *, encode):
        trace.append("prefix")
        windows.append(window)
        return original_prefix(value, window, encode=encode)

    monkeypatch.setattr(model, "_context_prefix", prefix)
    attention = branch.mid_block.attentions[0]
    native_attention = attention.forward

    def full_attention(value):
        trace.append("attention")
        torch.testing.assert_close(value, prefix_full, rtol=5e-9, atol=2e-10)
        return native_attention(value)

    monkeypatch.setattr(attention, "forward", full_attention)
    native_sdpa = q.F.scaled_dot_product_attention

    def sdpa(query, key, value, *args, **kwargs):
        sdpa_calls.append(
            (tuple(query.shape), tuple(key.shape), tuple(value.shape), args, kwargs)
        )
        return native_sdpa(query, key, value, *args, **kwargs)

    monkeypatch.setattr(q.F, "scaled_dot_product_attention", sdpa)
    handle = branch.mid_block.resnets[1].register_forward_pre_hook(
        lambda _module, _inputs: trace.append("suffix")
    )
    try:
        actual = (
            model.spatial_tiled_encode(source)
            if encode
            else model.spatial_tiled_decode(source)
        )
    finally:
        handle.remove()
    torch.testing.assert_close(actual, expected, rtol=5e-9, atol=2e-10)
    assert windows == cores
    assert trace == ["prefix"] * len(cores) + ["attention"] + ["suffix"] * len(cores)
    assert len(sdpa_calls) == 1
    query, key, value, args, kwargs = sdpa_calls[0]
    assert query == key == value == (1, 1, shape[0] * shape[1], branch.mid_block.dim)
    assert args == () and kwargs == {}
    assert actual.shape == expected.shape
    inputs = (
        source,
        branch.conv_in.weight,
        attention.to_qkv.weight,
        branch.conv_out.weight,
    )
    weight = torch.randn_like(expected)
    expected_grads = torch.autograd.grad((expected * weight).mean(), inputs)
    actual_grads = torch.autograd.grad((actual * weight).mean(), inputs)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=5e-8, atol=1e-9)


def _assert_suffix_matches_full_stages(model, source, *, encode, core):
    branch = model.encoder if encode else model.decoder
    ratio = model.spatial_compression_ratio
    shape = tuple(n // ratio if encode else n for n in source.shape[-2:])
    output_scale = 1 if encode else ratio
    target = tuple((a * output_scale, b * output_scale) for a, b in core)
    stages = [(branch.mid_block.resnets[1], 2, 1)]
    if not encode:
        stages += [
            (block, 2 * len(block.resnets), 2 if block.upsampler is not None else 1)
            for block in branch.up_blocks
        ]
    stages.append((None, 1, 1))
    plan = model._context_plan(target, stages, shape)
    attention = branch.mid_block.attentions[0]
    prefix_full = _full_prefix(model, source, encode=encode)
    full = attention(prefix_full)
    (a, b), (c, d) = plan[0]
    tiled = full[..., a:b, c:d].contiguous()

    for index, (block, _, scale) in enumerate(stages):
        if block is None:
            full = branch.conv_out(branch.nonlinearity(branch.norm_out(full)))
            tiled = branch.conv_out(branch.nonlinearity(branch.norm_out(tiled)))
        elif isinstance(block, q.QwenImage21ResidualUpBlock):
            full = block(full, first_chunk=True)
            tiled = block(tiled, first_chunk=True)
        else:
            full = block(full)
            tiled = block(tiled)

        (top, _), (left, _) = plan[index]
        (a, b), (c, d) = plan[index + 1]
        tiled = tiled[
            ..., a - top * scale : b - top * scale, c - left * scale : d - left * scale
        ].contiguous()
        full_core = full[..., a:b, c:d]
        assert tiled.shape == full_core.shape
        torch.testing.assert_close(tiled, full_core, rtol=5e-9, atol=2e-10)

    if encode:
        tiled = model.quant_conv(tiled)
        full = model.quant_conv(full)
        expected = full[..., core[0][0] : core[0][1], core[1][0] : core[1][1]]
    else:
        tiled = tiled.clamp(-1, 1)
        full = full.clamp(-1, 1)
        (a, b), (c, d) = target
        expected = full[..., a:b, c:d]
    torch.testing.assert_close(tiled, expected, rtol=5e-9, atol=2e-10)


@pytest.mark.parametrize("encode", [True, False], ids=["encoder", "decoder"])
def test_suffix_progressively_crops_dead_halo_and_matches_raw_full_stages(encode):
    model = tiny_vae()
    if encode:
        source = torch.randn((1, 3, 1, 128, 160), dtype=torch.float64)
        core = ((2, 6), (3, 8))
    else:
        source = torch.randn((1, 2, 1, 23, 26), dtype=torch.float64)
        core = ((8, 12), (10, 14))

    with torch.inference_mode():
        assert model._supports_context_tiles(source, encode=encode)
        _assert_suffix_matches_full_stages(model, source, encode=encode, core=core)


@pytest.mark.parametrize(
    "reason", ["training_dropout", "patch", "nonresidual", "compression_mismatch"]
)
def test_unsupported_configuration_uses_native_tile_fallback(monkeypatch, reason):
    model = tiny_vae(
        dropout=0.25 if reason == "training_dropout" else 0.0,
        patch_size=2 if reason == "patch" else None,
        is_residual=reason != "nonresidual",
        spatial_compression_ratio=8 if reason == "compression_mismatch" else 16,
    )
    if reason == "training_dropout":
        model.train()
    z = torch.zeros((1, 2, 1, 3, 4), dtype=torch.float64)
    marker = torch.full_like(z, 0.375)
    monkeypatch.setattr(
        q.ParallelTiledVAE, "spatial_tiled_decode", lambda self, value: marker
    )
    assert not model._supports_context_tiles(z, encode=False)
    assert model.spatial_tiled_decode(z) is marker


@pytest.mark.parametrize("encode", [True, False], ids=["encoder", "decoder"])
def test_compile_keeps_context_tiling(monkeypatch, encode):
    model = tiny_vae()
    model.tile_sample_stride_height = 2 * model.spatial_compression_ratio
    model.tile_sample_stride_width = 3 * model.spatial_compression_ratio
    source = torch.randn(
        (1, 3, 1, 64, 80) if encode else (1, 2, 1, 3, 4), dtype=torch.float64
    )
    method = model.spatial_tiled_encode if encode else model.spatial_tiled_decode
    monkeypatch.setattr(
        q.ParallelTiledVAE,
        "spatial_tiled_encode" if encode else "spatial_tiled_decode",
        Mock(side_effect=AssertionError("compilation must not select old tiles")),
    )
    with torch.inference_mode():
        expected = model._encode(source) if encode else model._decode(source)
        actual = torch.compile(method, backend="eager", fullgraph=True)(source)
    torch.testing.assert_close(actual, expected, rtol=5e-9, atol=2e-10)


@pytest.mark.parametrize("world_size", [1, 2])
def test_context_tiling_is_limited_to_single_process(monkeypatch, world_size):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    model = tiny_vae()
    image = torch.zeros((1, 3, 1, 64, 80), dtype=torch.float64)
    z = torch.zeros((1, 2, 1, 3, 4), dtype=torch.float64)
    encoded = torch.full((1, 4, 1, 4, 5), 0.375, dtype=torch.float64)
    decoded = torch.full((1, 3, 1, 48, 64), 0.625, dtype=torch.float64)
    native_encode = Mock(return_value=encoded)
    native_decode = Mock(return_value=decoded)
    monkeypatch.setattr(q.ParallelTiledVAE, "spatial_tiled_encode", native_encode)
    monkeypatch.setattr(q.ParallelTiledVAE, "spatial_tiled_decode", native_decode)
    model._context_tiled = Mock(
        side_effect=lambda value, *, encode: encoded if encode else decoded
    )
    assert model._supports_context_tiles(image, encode=True) == (world_size == 1)
    assert model._supports_context_tiles(z, encode=False) == (world_size == 1)
    torch.testing.assert_close(
        model.spatial_tiled_encode(image), encoded, rtol=0, atol=0
    )
    torch.testing.assert_close(model.spatial_tiled_decode(z), decoded, rtol=0, atol=0)
    assert model._context_tiled.call_count == (2 if world_size == 1 else 0)
    assert native_encode.call_count == native_decode.call_count == int(world_size > 1)


@pytest.mark.parametrize("shape", [(1, 3, 1, 64, 97), (1, 3, 2, 64, 80)])
def test_unsupported_encoder_input_uses_native_tile_fallback(monkeypatch, shape):
    model = tiny_vae()
    image = torch.zeros(shape, dtype=torch.float64)
    marker = torch.full((1, 4, 1, 4, 6), 0.625, dtype=torch.float64)
    with monkeypatch.context() as mp:
        mp.setattr(
            q.ParallelTiledVAE, "spatial_tiled_encode", lambda self, value: marker
        )
        with torch.inference_mode():
            assert not model._supports_context_tiles(image, encode=True)
            actual = model.spatial_tiled_encode(image)
    assert actual is marker


def test_nonpositive_context_tile_stride_fails_clearly():
    model = tiny_vae()
    model.tile_sample_stride_height = 0
    z = torch.zeros((1, 2, 1, 3, 4), dtype=torch.float64)
    with torch.inference_mode():
        assert model._supports_context_tiles(z, encode=False)
        with pytest.raises(ValueError, match="VAE tile strides must be positive"):
            model.spatial_tiled_decode(z)


def test_disabling_tiling_keeps_native_encode_decode_dispatch(monkeypatch):
    model = tiny_vae()
    model.use_tiling = False
    image = torch.zeros((1, 3, 1, 32, 48), dtype=torch.float64)
    z = torch.zeros((1, 2, 1, 2, 3), dtype=torch.float64)
    encoded = torch.randn((1, 4, 1, 2, 3), dtype=torch.float64)
    decoded = torch.randn((1, 3, 1, 32, 48), dtype=torch.float64).clamp(-1, 1)
    model._encode = Mock(return_value=encoded)
    model._decode = Mock(return_value=decoded)
    model.spatial_tiled_encode = Mock(
        side_effect=AssertionError("tiling must be disabled")
    )
    model.spatial_tiled_decode = Mock(
        side_effect=AssertionError("tiling must be disabled")
    )

    distribution = model.encode(image)
    torch.testing.assert_close(distribution.parameters, encoded, rtol=0, atol=0)
    model._encode.assert_called_once_with(image)
    model.spatial_tiled_encode.assert_not_called()

    actual = model.decode(z)
    torch.testing.assert_close(actual, decoded, rtol=0, atol=0)
    model._decode.assert_called_once_with(z)
    model.spatial_tiled_decode.assert_not_called()


@pytest.mark.parametrize("tiled", [False, True], ids=["untiled", "context-tiled"])
def test_decoder_output_range_is_preserved(tiled):
    model = tiny_vae()
    model.use_tiling = tiled
    with torch.no_grad():
        model.decoder.conv_out.weight.zero_()
        model.decoder.conv_out.bias.fill_(2.0)
    z = torch.zeros((1, 2, 1, 25, 34), dtype=torch.float64)
    with torch.inference_mode():
        actual = model.decode(z)
        assert torch.equal(actual, torch.ones_like(actual))
