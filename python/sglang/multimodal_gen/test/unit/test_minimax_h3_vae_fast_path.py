# SPDX-License-Identifier: Apache-2.0
"""H3 batched decode wiring, batching order and bit-exact output gates (CPU)."""

import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.kernels.ops.diffusion.sites.bitexact_gate import BitExactFusionGate
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.minimax_h3 import (
    FastH3PipelineConfig,
    MiniMaxH3PipelineConfig,
)
from sglang.multimodal_gen.runtime.models.vaes import minimax_h3_vae_cuda_opt
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    use_vae_fast_path,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
    klvae,
    processor,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae.batching import (
    decode_windows,
    forward_many,
    partitions,
)
from sglang.test.kernels.minimax_h3_vae import _h3_install_gate as _install_gate
from sglang.test.kernels.minimax_h3_vae import _h3_tiny_vae as _tiny_vae


def test_adjacent_batches_preserve_complete_sample_order_and_capacity():
    values = [torch.full((2, 3, 4), float(i)) for i in range(5)]
    calls = []

    def forward(x):
        calls.append(x.clone())
        return x + 7

    outputs = forward_many(values, forward=forward, capacity=5)
    assert [x.shape[0] for x in calls] == [4, 4, 2]
    for original, actual in zip(values, outputs):
        torch.testing.assert_close(actual, original + 7)
    values.insert(2, torch.zeros(2, 3, 5))
    assert list(partitions(values, capacity=8)) == [(0, 2), (2, 3), (3, 6)]


@pytest.mark.parametrize("limit", [0, 1, 2])
@pytest.mark.parametrize("use_head_tail", [False, True])
def test_windows_keep_time_axis_and_boundary_shapes(limit, use_head_tail):
    z = torch.arange(2 * 3 * 13).reshape(2, 3, 13, 1, 1)
    head = torch.full((2, 3, 1, 1, 1), -1) if use_head_tail else None
    tail = torch.full((2, 3, 1, 1, 1), -2) if use_head_tail else None
    calls = []

    def decode(x):
        calls.append(x.clone())
        return x.repeat_interleave(2, dim=2)

    vae = SimpleNamespace(tokens_chunk_size=3, token_overlap=1, _adaptive_decode=decode)
    outputs = dict(
        decode_windows(vae, z, head=head, tail=tail, count=4, window_limit=limit)
    )
    for index in range(4):
        expected = z[:, :, index * 3 : index * 3 + 4]
        if index == 0 and head is not None:
            expected = torch.cat([head, expected], dim=2)
        if index == 3 and tail is not None:
            expected = torch.cat([expected, tail], dim=2)
        torch.testing.assert_close(outputs[index], expected.repeat_interleave(2, dim=2))
    assert all(x.shape[0] % 2 == 0 for x in calls)
    if limit:
        assert max(x.shape[0] for x in calls) <= 2 * limit
    if use_head_tail:
        assert calls[0].shape[2] == 5 and calls[-1].shape[2] == 5


def test_config_caps_reach_the_vae_only_while_the_gate_is_on():
    config = MiniMaxH3VideoVAEConfig()
    assert (config.decoder_tile_batch, config.window_batch) == (64, 1)
    config.post_init()
    for name, value in (("decoder_tile_batch", -1), ("window_batch", True)):
        broken = MiniMaxH3VideoVAEConfig(**{name: value})
        with pytest.raises(ValueError, match=name):
            broken.post_init()
    # FastVideo's FastH3 recipe decodes all of a rank's tiles in one call.
    assert FastH3PipelineConfig().vae_config.decoder_tile_batch == 0
    assert MiniMaxH3PipelineConfig().vae_config.decoder_tile_batch == 64

    vae = _tiny_vae(decoder_tile_batch=5, window_batch=0)
    assert (vae.decoder_tile_batch, vae.window_batch) == (5, 0)
    assert vae._sgl_gate is None and vae._batch_caps() == (1, 1)
    gate = _install_gate(vae)
    assert vae._batch_caps() == (1, 1)
    with use_vae_fast_path(vae, True):
        assert gate.enabled and vae._batch_caps() == (5, 0)
    assert vae._batch_caps() == (1, 1)


def test_installer_shares_one_gate_between_attention_and_batched_decode(monkeypatch):
    monkeypatch.delenv("MINIMAX_H3_VAE_DECODER_VIT_FP32_NORM", raising=False)
    vae = _tiny_vae(decoder_tile_batch=0)
    # The installer admits the production wrapper class; the tiny VAE stands in.
    with mock.patch(
        "sglang.multimodal_gen.runtime.models.vaes.minimax_h3.MiniMaxH3VideoVAE",
        AutoencoderKLLegacy,
    ):
        assert minimax_h3_vae_cuda_opt.maybe_optimize_minimax_h3_vae(vae) is vae
    gate = vae._sgl_gate
    assert isinstance(gate, VaeFastPathGate) and not gate.enabled
    assert all(block.attn._sgl_gate is gate for block in vae.decoder.transformer_blocks)
    with use_vae_fast_path(vae, True):
        assert gate.enabled and vae._batch_caps() == (0, 1)
    assert not gate.enabled


@pytest.mark.parametrize(
    "caps,expected_batches",
    [
        # 2 windows x 9 tiles x 2 samples. Gate off: one tile of one window.
        (None, [2] * 18),
        # Both windows share a forward, then all nine tiles of the pair.
        ({"decoder_tile_batch": 0, "window_batch": 0}, [36]),
        # One window at a time, nine tiles per forward.
        ({"decoder_tile_batch": 64, "window_batch": 1}, [18, 18]),
        # Four tiles per forward, two windows per forward.
        ({"decoder_tile_batch": 4, "window_batch": 2}, [16, 16, 4]),
    ],
)
def test_batched_decode_matches_eager_on_cpu(monkeypatch, caps, expected_batches):
    vae = _tiny_vae(**(caps or {}))
    z = torch.randn(2, 4, 11, 16, 16)
    with torch.no_grad():
        eager = vae.decode_temporal(z)
    gate = _install_gate(vae)
    batches = []
    decode = vae.decode

    def recording_decode(latent):
        batches.append(int(latent.shape[0]))
        return decode(latent)

    monkeypatch.setattr(vae, "decode", recording_decode)
    with torch.no_grad(), use_vae_fast_path(vae, caps is not None):
        batched = vae.decode_temporal(z)
    assert gate.enabled is False
    assert batches == expected_batches
    torch.testing.assert_close(batched, eager, rtol=1e-4, atol=1e-4)


def _candidate_on_cpu(value, gate, sig=None):
    """Stand in for the device check so the gate logic runs on CPU tensors."""
    if sig is None:
        return gate.can_attempt_once() and not torch.is_grad_enabled()
    return klvae._gate_admits(gate, sig) and not torch.is_grad_enabled()


@pytest.mark.parametrize("bit_exact", [True, False])
def test_temporal_blend_write_restores_the_frames_on_a_mismatch(monkeypatch, bit_exact):
    vae = _tiny_vae()
    gate = BitExactFusionGate("test temporal blend", per_signature=True)
    monkeypatch.setattr(klvae, "_TEMPORAL_BLEND_FUSION", gate)
    monkeypatch.setattr(klvae, "_output_kernel_candidate", _candidate_on_cpu)
    monkeypatch.setattr(
        klvae, "can_use_minimax_h3_vae_temporal_blend_write", lambda *a, **k: True
    )

    def fake_kernel(part, overlap, *, blend_extent, out, start, count):
        blended = vae._blended_frames(part, overlap)[:, :, :count]
        out[:, :, start : start + count].copy_(blended if bit_exact else blended + 1)

    monkeypatch.setattr(klvae, "minimax_h3_vae_temporal_blend_write", fake_kernel)
    overlap = torch.randn(1, 3, 6, 8, 8)
    part = torch.randn(1, 3, 6, 8, 8)
    expected = torch.zeros(1, 3, 9, 8, 8)
    expected[:, :, 2:7] = vae._blended_frames(part, overlap)[:, :, :5]
    dec = torch.zeros(1, 3, 9, 8, 8)
    with torch.no_grad():
        vae._write_frames(dec, part, overlap, start=2, count=5)
        assert torch.equal(dec, expected)
        vae._write_frames(dec, part, None, start=2, count=5)
    expected[:, :, 2:7] = part[:, :, :5]
    assert torch.equal(dec, expected)
    assert gate.disabled is not bit_exact
    # The blend and the plain copy are separate signatures.
    assert gate.is_verified(True) is bit_exact and gate.is_verified(False) is bit_exact


def test_temporal_blend_copy_cannot_vouch_for_the_blend(monkeypatch):
    """Production writes the first window without an overlap. A kernel that
    copies correctly but blends wrongly must still be caught on the first
    blended write, so the no-overlap copy must not verify the blend path."""
    vae = _tiny_vae()
    gate = BitExactFusionGate("test temporal blend", per_signature=True)
    monkeypatch.setattr(klvae, "_TEMPORAL_BLEND_FUSION", gate)
    monkeypatch.setattr(klvae, "_output_kernel_candidate", _candidate_on_cpu)
    monkeypatch.setattr(
        klvae, "can_use_minimax_h3_vae_temporal_blend_write", lambda *a, **k: True
    )

    def copy_right_blend_wrong(part, overlap, *, blend_extent, out, start, count):
        blended = vae._blended_frames(part, overlap)[:, :, :count]
        out[:, :, start : start + count].copy_(
            blended if overlap is None else blended + 1
        )

    monkeypatch.setattr(
        klvae, "minimax_h3_vae_temporal_blend_write", copy_right_blend_wrong
    )
    first = torch.randn(1, 3, 6, 8, 8)
    second = torch.randn(1, 3, 6, 8, 8)
    dec = torch.zeros(1, 3, 12, 8, 8)
    with torch.no_grad():
        vae._write_frames(dec, first, None, start=0, count=6)
        assert gate.is_verified(False) and not gate.is_verified(True)
        vae._write_frames(dec, second, first, start=6, count=6)
    expected = torch.cat([first, vae._blended_frames(second, first)[:, :, :6]], dim=2)
    assert torch.equal(dec, expected)
    assert gate.disabled and not gate.is_verified(True)


@pytest.mark.parametrize("bit_exact", [True, False])
@pytest.mark.parametrize("runtime_owned", [False, True])
def test_denorm_kernel_runs_in_place_only_on_owned_frames(
    monkeypatch, caplog, runtime_owned, bit_exact
):
    vae = _tiny_vae()
    # warning_once caches by message, so each case needs its own gate name.
    gate = BitExactFusionGate(f"test denorm owned={runtime_owned}")
    monkeypatch.setattr(processor, "_DENORM_FUSION", gate)
    monkeypatch.setattr(processor, "_denorm_kernel_candidate", _candidate_on_cpu)
    monkeypatch.setattr(
        processor, "can_use_minimax_h3_vae_denorm_clamp", lambda *a, **k: True
    )
    seen = []

    def fake_kernel(value, *, mean, std, out=None):
        seen.append((tuple(value.shape), out is value))
        result = vae.processor._revert_tensor_eager(value, runtime_owned=False)
        if not bit_exact:
            result = result + 1
        if out is None:
            return result
        out.copy_(result)
        return out

    monkeypatch.setattr(processor, "minimax_h3_vae_denorm_clamp", fake_kernel)
    frames = torch.randn(2, 3, 4, 8, 8) * 0.6
    expected = vae.processor._revert_tensor_eager(frames.clone(), runtime_owned=False)
    with torch.no_grad(), caplog.at_level(logging.WARNING):
        result = vae.processor.revert_tensor(frames, runtime_owned=runtime_owned)
    assert torch.equal(result, expected)
    if bit_exact:
        # First sight verifies one frame out of place, then the full run follows.
        assert gate.verified and seen == [
            ((1, 3, 1, 8, 8), False),
            ((2, 3, 4, 8, 8), runtime_owned),
        ]
        # The kernel rewrites owned frames in place and allocates otherwise.
        assert (result.data_ptr() == frames.data_ptr()) is runtime_owned
    else:
        # The probe fails, nothing is written through the kernel, eager takes over.
        assert gate.disabled and seen == [((1, 3, 1, 8, 8), False)]
        assert (
            f"test denorm owned={runtime_owned} fast path is not bit-exact"
            in caplog.text
        )
