"""Equivalence oracle for the Wan/QwenImage VAE temporal causal cache.

The reference implementation below restates the cache protocol independently of
the production code: it re-derives the causal padding from an explicit geometry
spec and drives the cache with a plain ``list`` + integer cursor. Only weights
and the stateless sub-layers (norms, activations, spatial resample) are shared
with the module under test.

Keeping the oracle independent lets it stay a valid reference across the whole
refactor, including after the spatial padding is handed back to cuDNN.
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.causal_conv3d_cache import (
    CausalCacheMode,
    CausalConv3d,
    CausalConvCache,
    TimeDownsampleCausalConv3d,
    TimeUpsampleCausalConv3d,
    assign_causal_cache_keys,
    causal_cache_scope,
    interleave_time,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae

CACHE_T = 2


class _RefCausalConv3d:
    """Causal Conv3d reference: explicit zero padding, no hidden state.

    ``padding`` is the *original* symmetric padding the layer was built with;
    the time dimension is padded causally (``2 * padding[0]`` on the left).
    """

    def __init__(self, conv: torch.nn.Conv3d, padding: tuple[int, int, int]):
        self.conv = conv
        self.pad = (
            padding[2],
            padding[2],
            padding[1],
            padding[1],
            2 * padding[0],
            0,
        )

    def __call__(self, x: torch.Tensor, cache_x: torch.Tensor | None = None):
        pad = list(self.pad)
        if cache_x is not None and pad[4] > 0:
            x = torch.cat([cache_x, x], dim=2)
            pad[4] -= cache_x.shape[2]
        if any(pad):
            x = F.pad(x, pad)
        return F.conv3d(x, self.conv.weight, self.conv.bias, self.conv.stride)


def _ref_grow_cache(x: torch.Tensor, prev) -> torch.Tensor:
    """Reproduce the short-chunk fixup: borrow the last frame of the previous cache."""
    cache_x = x[:, :, -CACHE_T:, :, :].clone()
    if cache_x.shape[2] < 2 and prev is not None:
        cache_x = torch.cat(
            [prev[:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
        )
    return cache_x


def _ref_residual_block(block, x, feat_cache, feat_idx):
    conv1 = _RefCausalConv3d(block.conv1, padding=(1, 1, 1))
    conv2 = _RefCausalConv3d(block.conv2, padding=(1, 1, 1))

    if isinstance(block.conv_shortcut, torch.nn.Identity):
        h = x
    else:
        h = _RefCausalConv3d(block.conv_shortcut, padding=(0, 0, 0))(x)

    x = block.nonlinearity(block.norm1(x))
    idx = feat_idx[0]
    cache_x = _ref_grow_cache(x, feat_cache[idx])
    x = conv1(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1

    x = block.nonlinearity(block.norm2(x))
    x = block.dropout(x)
    idx = feat_idx[0]
    cache_x = _ref_grow_cache(x, feat_cache[idx])
    x = conv2(x, feat_cache[idx])
    feat_cache[idx] = cache_x
    feat_idx[0] += 1

    return x + h


def _ref_resample(resample, x, feat_cache, feat_idx):
    b, c, t, h, w = x.size()

    if resample.mode == "upsample3d":
        time_conv = _RefCausalConv3d(resample.time_conv, padding=(1, 0, 0))
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx] = "Rep"
            feat_idx[0] += 1
        else:
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] != "Rep":
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2
                )
            if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
            if feat_cache[idx] == "Rep":
                x = time_conv(x)
            else:
                x = time_conv(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

            x = x.reshape(b, 2, c, t, h, w)
            x = torch.stack((x[:, 0], x[:, 1]), 3)
            x = x.reshape(b, c, t * 2, h, w)

    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = resample.resample(x)
    x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

    if resample.mode == "downsample3d":
        time_conv = _RefCausalConv3d(resample.time_conv, padding=(0, 0, 0))
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx] = x.clone()
            feat_idx[0] += 1
        else:
            cache_x = x[:, :, -1:, :, :].clone()
            x = time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
            feat_cache[idx] = cache_x
            feat_idx[0] += 1

    return x


def _run_production(module, chunks, num_slots):
    """Drive the production module chunk by chunk through its contextvar protocol."""
    feat_map = [None] * num_slots
    outs = []
    with wanvae.forward_context(feat_cache_arg=feat_map, feat_idx_arg=0):
        for chunk in chunks:
            wanvae.feat_idx.set(0)
            outs.append(module(chunk))
    return outs


def _run_reference(ref_fn, module, chunks, num_slots):
    feat_map = [None] * num_slots
    outs = []
    for chunk in chunks:
        feat_idx = [0]
        outs.append(ref_fn(module, chunk, feat_map, feat_idx))
    return outs


def _make_chunks(shape, num_chunks, *, generator):
    return [
        torch.randn(shape, dtype=torch.float64, generator=generator)
        for _ in range(num_chunks)
    ]


class TestCausalCacheOracle(unittest.TestCase):
    """Pin the current cache protocol so the refactor can be checked against it."""

    def setUp(self):
        torch.manual_seed(0)
        self.generator = torch.Generator().manual_seed(1234)

    def _assert_chunks_equal(self, produced, expected):
        self.assertEqual(len(produced), len(expected))
        for i, (got, want) in enumerate(zip(produced, expected, strict=True)):
            self.assertEqual(got.shape, want.shape, msg=f"chunk {i} shape mismatch")
            torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"chunk {i}")

    def test_causal_conv3d_streaming(self):
        conv = wanvae.WanCausalConv3d(4, 6, 3, padding=1).double().eval()
        ref = _RefCausalConv3d(conv, padding=(1, 1, 1))
        chunks = _make_chunks((1, 4, 1, 5, 7), 4, generator=self.generator)

        prod_out, prod_cache = [], None
        ref_out, ref_cache = [], None
        for chunk in chunks:
            prod_out.append(conv(chunk, prod_cache))
            prod_cache = _ref_grow_cache(chunk, prod_cache)
            ref_out.append(ref(chunk, ref_cache))
            ref_cache = _ref_grow_cache(chunk, ref_cache)
        self._assert_chunks_equal(prod_out, ref_out)

    def test_residual_block_streaming(self):
        block = wanvae.WanResidualBlock(4, 6).double().eval()
        chunks = _make_chunks((1, 4, 1, 5, 7), 5, generator=self.generator)
        produced = _run_production(block, chunks, num_slots=2)
        expected = _run_reference(_ref_residual_block, block, chunks, num_slots=2)
        self._assert_chunks_equal(produced, expected)

    def test_residual_block_streaming_multi_frame_chunks(self):
        block = wanvae.WanResidualBlock(4, 4).double().eval()
        chunks = _make_chunks((1, 4, 4, 5, 7), 3, generator=self.generator)
        produced = _run_production(block, chunks, num_slots=2)
        expected = _run_reference(_ref_residual_block, block, chunks, num_slots=2)
        self._assert_chunks_equal(produced, expected)

    def test_upsample3d_resample_streaming(self):
        resample = wanvae.WanResample(4, mode="upsample3d").double().eval()
        chunks = _make_chunks((1, 4, 1, 5, 7), 4, generator=self.generator)
        produced = _run_production(resample, chunks, num_slots=1)
        expected = _run_reference(_ref_resample, resample, chunks, num_slots=1)
        self._assert_chunks_equal(produced, expected)

    def test_downsample3d_resample_streaming(self):
        resample = wanvae.WanResample(4, mode="downsample3d").double().eval()
        # Encoder feeds 1 frame for the first chunk, then 4 frames per chunk.
        chunks = [
            torch.randn((1, 4, 1, 8, 8), dtype=torch.float64, generator=self.generator),
            torch.randn((1, 4, 4, 8, 8), dtype=torch.float64, generator=self.generator),
            torch.randn((1, 4, 4, 8, 8), dtype=torch.float64, generator=self.generator),
        ]
        produced = _run_production(resample, chunks, num_slots=1)
        expected = _run_reference(_ref_resample, resample, chunks, num_slots=1)
        self._assert_chunks_equal(produced, expected)


def _ref_time_upsample_stream(conv, chunks):
    """Temporal branch of ``resample_forward`` for ``upsample3d``, spatial part removed."""
    ref = _RefCausalConv3d(conv, padding=(1, 0, 0))
    slot = None
    outs = []
    for x in chunks:
        b, c, t, h, w = x.size()
        if slot is None:
            slot = "Rep"
            outs.append(x)
            continue
        cache_x = x[:, :, -CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and slot != "Rep":
            cache_x = torch.cat([slot[:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
        if cache_x.shape[2] < 2 and slot == "Rep":
            cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
        y = ref(x) if slot == "Rep" else ref(x, slot)
        slot = cache_x
        y = y.reshape(b, 2, c, t, h, w)
        y = torch.stack((y[:, 0], y[:, 1]), 3).reshape(b, c, t * 2, h, w)
        outs.append(y)
    return outs


def _ref_time_downsample_stream(conv, chunks):
    """Temporal branch of ``resample_forward`` for ``downsample3d``."""
    ref = _RefCausalConv3d(conv, padding=(0, 0, 0))
    slot = None
    outs = []
    for x in chunks:
        if slot is None:
            slot = x.clone()
            outs.append(x)
            continue
        cache_x = x[:, :, -1:, :, :].clone()
        outs.append(ref(torch.cat([slot[:, :, -1:, :, :], x], 2)))
        slot = cache_x
    return outs


def _run_managed(conv, chunks, mode=CausalCacheMode.STREAMING):
    """Drive a cache-managed conv chunk by chunk inside one cache scope."""
    cache = CausalConvCache(mode)
    outs = []
    with causal_cache_scope(cache):
        for chunk in chunks:
            outs.append(conv(chunk))
            cache.advance_chunk()
    return outs


def _clone_weights(dst: nn.Conv3d, src: nn.Conv3d) -> None:
    dst.load_state_dict(src.state_dict())


class TestManagedCausalConv3d(unittest.TestCase):
    """The cache-managed convs must reproduce the oracle bit for bit."""

    def setUp(self):
        torch.manual_seed(0)
        self.generator = torch.Generator().manual_seed(1234)

    def _assert_chunks_equal(self, produced, expected):
        self.assertEqual(len(produced), len(expected))
        for i, (got, want) in enumerate(zip(produced, expected, strict=True)):
            self.assertEqual(got.shape, want.shape, msg=f"chunk {i} shape mismatch")
            torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"chunk {i}")

    def test_cache_frames_derivation(self):
        """Pointwise-in-time convs must not be given a cache slot at all."""
        self.assertEqual(CausalConv3d(4, 6, 3, padding=1).cache_frames, 2)
        self.assertEqual(CausalConv3d(4, 6, 1).cache_frames, 0)
        self.assertEqual(
            TimeUpsampleCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).cache_frames, 2
        )
        self.assertEqual(
            TimeDownsampleCausalConv3d(
                4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            ).cache_frames,
            1,
        )

    def test_pointwise_conv_is_transparent(self):
        """A 1x1x1 conv must behave exactly like plain Conv3d, cache or not."""
        conv = CausalConv3d(4, 6, 1).double().eval()
        conv.cache_key = "pointwise"
        x = torch.randn((1, 4, 3, 5, 7), dtype=torch.float64, generator=self.generator)
        expected = F.conv3d(x, conv.weight, conv.bias)
        cache = CausalConvCache(CausalCacheMode.STREAMING)
        with causal_cache_scope(cache):
            torch.testing.assert_close(conv(x), expected, rtol=0, atol=0)
        self.assertFalse(cache.contains("pointwise"))

    def test_assign_cache_keys_uses_module_path(self):
        root = nn.Module()
        root.encoder = nn.Module()
        root.encoder.conv_in = CausalConv3d(4, 4, 3, padding=1)
        root.decoder = nn.Module()
        root.decoder.conv_in = CausalConv3d(4, 4, 3, padding=1)
        assign_causal_cache_keys(root)
        self.assertEqual(root.encoder.conv_in.cache_key, "encoder.conv_in")
        self.assertEqual(root.decoder.conv_in.cache_key, "decoder.conv_in")

    def test_streaming_matches_reference(self):
        legacy = wanvae.WanCausalConv3d(4, 6, 3, padding=1).double().eval()
        conv = CausalConv3d(4, 6, 3, padding=1).double().eval()
        _clone_weights(conv, legacy)
        conv.cache_key = "conv"
        chunks = _make_chunks((1, 4, 1, 5, 7), 5, generator=self.generator)

        ref_out, ref_cache = [], None
        for chunk in chunks:
            ref_out.append(
                _RefCausalConv3d(legacy, padding=(1, 1, 1))(chunk, ref_cache)
            )
            ref_cache = _ref_grow_cache(chunk, ref_cache)
        self._assert_chunks_equal(_run_managed(conv, chunks), ref_out)

    def test_streaming_matches_reference_multi_frame(self):
        legacy = wanvae.WanCausalConv3d(4, 4, 3, padding=1).double().eval()
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        _clone_weights(conv, legacy)
        conv.cache_key = "conv"
        chunks = _make_chunks((1, 4, 4, 5, 7), 4, generator=self.generator)

        ref_out, ref_cache = [], None
        for chunk in chunks:
            ref_out.append(
                _RefCausalConv3d(legacy, padding=(1, 1, 1))(chunk, ref_cache)
            )
            ref_cache = _ref_grow_cache(chunk, ref_cache)
        self._assert_chunks_equal(_run_managed(conv, chunks), ref_out)

    def test_stateless_pads_every_call_independently(self):
        conv = CausalConv3d(4, 6, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        ref = _RefCausalConv3d(conv, padding=(1, 1, 1))
        chunks = _make_chunks((1, 4, 3, 5, 7), 3, generator=self.generator)
        expected = [ref(chunk) for chunk in chunks]
        produced = _run_managed(conv, chunks, mode=CausalCacheMode.STATELESS)
        self._assert_chunks_equal(produced, expected)

    def test_time_upsample_streaming_matches_reference(self):
        legacy = (
            wanvae.WanCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        conv = (
            TimeUpsampleCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        _clone_weights(conv, legacy)
        conv.cache_key = "time_conv"
        chunks = _make_chunks((1, 4, 1, 5, 7), 5, generator=self.generator)
        expected = _ref_time_upsample_stream(legacy, chunks)
        self._assert_chunks_equal(_run_managed(conv, chunks), expected)

    def test_time_upsample_stateless_matches_whole_tensor_path(self):
        legacy = (
            wanvae.WanCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        conv = (
            TimeUpsampleCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        _clone_weights(conv, legacy)
        conv.cache_key = "time_conv"
        x = torch.randn((1, 4, 5, 5, 7), dtype=torch.float64, generator=self.generator)

        b, c, t, h, w = x.shape
        legacy_out = legacy(x).reshape(b, 2, c, t, h, w)
        expected = torch.stack((legacy_out[:, 0], legacy_out[:, 1]), 3).reshape(
            b, c, t * 2, h, w
        )
        produced = _run_managed(conv, [x], mode=CausalCacheMode.STATELESS)
        self._assert_chunks_equal(produced, [expected])

    def test_time_upsample_first_frame_is_skipped(self):
        conv = (
            TimeUpsampleCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        conv.cache_key = "time_conv"
        x = torch.randn((1, 4, 1, 5, 7), dtype=torch.float64, generator=self.generator)
        produced = _run_managed(conv, [x], mode=CausalCacheMode.FIRST_FRAME)
        self._assert_chunks_equal(produced, [x])

    def test_time_downsample_streaming_matches_reference(self):
        legacy = (
            wanvae.WanCausalConv3d(4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
            .double()
            .eval()
        )
        conv = (
            TimeDownsampleCausalConv3d(
                4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
            .double()
            .eval()
        )
        _clone_weights(conv, legacy)
        conv.cache_key = "time_conv"
        chunks = [
            torch.randn((1, 4, 1, 5, 7), dtype=torch.float64, generator=self.generator),
            torch.randn((1, 4, 4, 5, 7), dtype=torch.float64, generator=self.generator),
            torch.randn((1, 4, 4, 5, 7), dtype=torch.float64, generator=self.generator),
        ]
        expected = _ref_time_downsample_stream(legacy, chunks)
        self._assert_chunks_equal(_run_managed(conv, chunks), expected)

    def test_time_downsample_stateless_keeps_ceil_half_length(self):
        """The whole-tensor path pads two frames so the output stays at ceil(T/2).

        This is what the old ``time_conv._padding[4] = 2`` mutation did, except
        it silently missed Wan 2.2 where the resample hides inside a residual
        down block.
        """
        conv = (
            TimeDownsampleCausalConv3d(
                4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
            .double()
            .eval()
        )
        conv.cache_key = "time_conv"
        for num_frames in (4, 5, 8, 17):
            x = torch.randn(
                (1, 4, num_frames, 3, 3), dtype=torch.float64, generator=self.generator
            )
            out = _run_managed(conv, [x], mode=CausalCacheMode.STATELESS)[0]
            self.assertEqual(
                out.shape[2],
                -(-num_frames // 2),
                msg=f"T={num_frames} must map to ceil(T/2)",
            )

    def test_interleave_time_matches_legacy_reshape(self):
        x = torch.randn((2, 8, 3, 4, 5), dtype=torch.float64, generator=self.generator)
        b, c2, t, h, w = x.shape
        c = c2 // 2
        legacy = x.reshape(b, 2, c, t, h, w)
        legacy = torch.stack((legacy[:, 0], legacy[:, 1]), 3).reshape(b, c, t * 2, h, w)
        torch.testing.assert_close(interleave_time(x), legacy, rtol=0, atol=0)

    def test_cache_scope_is_restored_after_exception(self):
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        x = torch.randn((1, 4, 1, 5, 7), dtype=torch.float64, generator=self.generator)
        with self.assertRaises(RuntimeError):
            with causal_cache_scope(CausalConvCache()):
                conv(x)
                raise RuntimeError("boom")
        # A fresh scope must not see the aborted pass's state.
        cache = CausalConvCache()
        with causal_cache_scope(cache):
            conv(x)
        self.assertEqual(cache.chunk_index, 0)
        self.assertTrue(cache.contains("conv"))

    def test_cached_tail_does_not_pin_the_activation(self):
        """The retained tail must be a copy, not a view of the whole chunk."""
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        x = torch.randn((1, 4, 8, 5, 7), dtype=torch.float64, generator=self.generator)
        cache = CausalConvCache()
        with causal_cache_scope(cache):
            conv(x)
        tail = cache.get("conv")
        self.assertEqual(tail.shape[2], conv.cache_frames)
        self.assertLess(tail.untyped_storage().nbytes(), x.numel() * x.element_size())


if __name__ == "__main__":
    unittest.main()
