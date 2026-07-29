"""Regression tests for the Wan/QwenImage VAE temporal causal cache.

The reference implementation below restates the cache protocol independently of
the production code: it re-derives the causal padding from an explicit geometry
spec and drives the cache with a plain ``list`` + integer cursor, the way the
VAEs did before the cache moved into a manager. Only weights are shared with the
module under test, so the reference stays a valid oracle even after the spatial
padding is handed back to cuDNN.
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
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
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan

CACHE_T = 2


class _RefCausalConv3d:
    """Causal Conv3d reference: explicit zero padding, no hidden state.

    ``padding`` is the *original* symmetric padding the layer was built with;
    the time dimension is padded causally (``2 * padding[0]`` on the left).
    """

    def __init__(self, conv: nn.Conv3d, padding: tuple[int, int, int]):
        self.conv = conv
        self.pad = (padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0)

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
        cache_x = torch.cat([prev[:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
    return cache_x


def _ref_plain_stream(conv, chunks):
    """The old per-call protocol for a symmetrically padded causal conv."""
    ref = _RefCausalConv3d(conv, padding=(1, 1, 1))
    outs, cache = [], None
    for x in chunks:
        outs.append(ref(x, cache))
        cache = _ref_grow_cache(x, cache)
    return outs


def _ref_time_upsample_stream(conv, chunks):
    """Temporal branch of the old ``resample_forward`` for ``upsample3d``."""
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
    """Temporal branch of the old ``resample_forward`` for ``downsample3d``."""
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
    with causal_cache_scope(cache, [conv]):
        for chunk in chunks:
            outs.append(conv(chunk))
            cache.advance_chunk()
    return outs


def _make_chunks(shape, num_chunks, *, generator):
    return [
        torch.randn(shape, dtype=torch.float64, generator=generator)
        for _ in range(num_chunks)
    ]


def _tiny_wan(*, is_residual: bool) -> AutoencoderKLWan:
    """A few-channel Wan VAE with random weights; no checkpoint needed."""
    arch = WanVAEArchConfig(
        base_dim=8,
        z_dim=4,
        dim_mult=(1, 2, 2),
        num_res_blocks=1,
        temperal_downsample=(True, True) if is_residual else (False, True),
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
        is_residual=is_residual,
        # Wan 2.2 folds a 2x2 pixel patch into the channel dim before encoding.
        patch_size=2 if is_residual else None,
        in_channels=12 if is_residual else 3,
        out_channels=12 if is_residual else 3,
        # One temporal downsample for 2.1, two for 2.2.
        scale_factor_temporal=4 if is_residual else 2,
        scale_factor_spatial=4,
    )
    config = WanVAEConfig(arch_config=arch)
    config.use_parallel_encode = False
    config.use_parallel_decode = False
    torch.manual_seed(0)
    return AutoencoderKLWan(config).double().eval()


class TestCausalConv3dCache(unittest.TestCase):
    """The cache-managed convs must reproduce the old protocol bit for bit."""

    def setUp(self):
        torch.manual_seed(0)
        self.generator = torch.Generator().manual_seed(1234)

    def _assert_chunks_equal(self, produced, expected):
        self.assertEqual(len(produced), len(expected))
        for i, (got, want) in enumerate(zip(produced, expected, strict=True)):
            self.assertEqual(got.shape, want.shape, msg=f"chunk {i} shape mismatch")
            torch.testing.assert_close(got, want, rtol=0, atol=0, msg=f"chunk {i}")

    def test_cache_frames_derivation(self):
        """Pointwise-in-time convs must not be given a cache slot at all.

        Handing them one would prepend frames and change the output length,
        which then breaks the residual add downstream.
        """
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
        conv = CausalConv3d(4, 6, 1).double().eval()
        conv.cache_key = "pointwise"
        x = torch.randn((1, 4, 3, 5, 7), dtype=torch.float64, generator=self.generator)
        expected = F.conv3d(x, conv.weight, conv.bias)
        cache = CausalConvCache(CausalCacheMode.STREAMING)
        with causal_cache_scope(cache, [conv]):
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

    def test_streaming_matches_reference_single_frame_chunks(self):
        conv = CausalConv3d(4, 6, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        chunks = _make_chunks((1, 4, 1, 5, 7), 5, generator=self.generator)
        self._assert_chunks_equal(
            _run_managed(conv, chunks), _ref_plain_stream(conv, chunks)
        )

    def test_streaming_matches_reference_multi_frame_chunks(self):
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        chunks = _make_chunks((1, 4, 4, 5, 7), 4, generator=self.generator)
        self._assert_chunks_equal(
            _run_managed(conv, chunks), _ref_plain_stream(conv, chunks)
        )

    def test_stateless_pads_every_call_independently(self):
        conv = CausalConv3d(4, 6, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        ref = _RefCausalConv3d(conv, padding=(1, 1, 1))
        chunks = _make_chunks((1, 4, 3, 5, 7), 3, generator=self.generator)
        self._assert_chunks_equal(
            _run_managed(conv, chunks, mode=CausalCacheMode.STATELESS),
            [ref(chunk) for chunk in chunks],
        )

    def test_time_upsample_streaming_matches_reference(self):
        conv = (
            TimeUpsampleCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        conv.cache_key = "time_conv"
        chunks = _make_chunks((1, 4, 1, 5, 7), 5, generator=self.generator)
        self._assert_chunks_equal(
            _run_managed(conv, chunks), _ref_time_upsample_stream(conv, chunks)
        )

    def test_time_upsample_stateless_matches_whole_tensor_path(self):
        conv = (
            TimeUpsampleCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        conv.cache_key = "time_conv"
        x = torch.randn((1, 4, 5, 5, 7), dtype=torch.float64, generator=self.generator)

        b, c, t, h, w = x.shape
        ref = _RefCausalConv3d(conv, padding=(1, 0, 0))(x).reshape(b, 2, c, t, h, w)
        expected = torch.stack((ref[:, 0], ref[:, 1]), 3).reshape(b, c, t * 2, h, w)
        self._assert_chunks_equal(
            _run_managed(conv, [x], mode=CausalCacheMode.STATELESS), [expected]
        )

    def test_time_conv_is_skipped_for_a_leading_single_frame(self):
        up = (
            TimeUpsampleCausalConv3d(4, 8, (3, 1, 1), padding=(1, 0, 0)).double().eval()
        )
        down = (
            TimeDownsampleCausalConv3d(
                4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
            .double()
            .eval()
        )
        up.cache_key, down.cache_key = "up", "down"
        x = torch.randn((1, 4, 1, 5, 7), dtype=torch.float64, generator=self.generator)
        for conv in (up, down):
            self._assert_chunks_equal(
                _run_managed(conv, [x], mode=CausalCacheMode.FIRST_FRAME), [x]
            )

    def test_time_downsample_streaming_matches_reference(self):
        conv = (
            TimeDownsampleCausalConv3d(
                4, 4, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )
            .double()
            .eval()
        )
        conv.cache_key = "time_conv"
        # The encoder feeds one frame for the first chunk, then four per chunk.
        chunks = [
            torch.randn((1, 4, 1, 5, 7), dtype=torch.float64, generator=self.generator),
            torch.randn((1, 4, 4, 5, 7), dtype=torch.float64, generator=self.generator),
            torch.randn((1, 4, 4, 5, 7), dtype=torch.float64, generator=self.generator),
        ]
        self._assert_chunks_equal(
            _run_managed(conv, chunks), _ref_time_downsample_stream(conv, chunks)
        )

    def test_time_downsample_stateless_keeps_ceil_half_length(self):
        """The whole-tensor path pads two frames so the output stays at ceil(T/2).

        That is what the old ``time_conv._padding[4] = 2`` mutation did, except
        it scanned ``down_blocks`` for ``WanResample`` and so silently missed
        Wan 2.2, where the resample hides inside a residual down block.
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

    def test_time_batch_flatten_is_a_layout_preserving_view(self):
        """Merging a size-1 dimension must not degrade the strides.

        ``permute().reshape()`` yields a view whose merged batch dimension has
        stride ``c`` rather than ``h * w * c`` when ``b * t == 1``.
        ``is_contiguous`` accepts that but ATen's ``suggest_memory_format`` does
        not, so interpolate/conv/pad/cat downstream all fall back to NCHW.
        """
        from sglang.multimodal_gen.runtime.layers.causal_conv3d_cache import (
            flatten_time_into_batch,
            unflatten_batch_into_time,
        )

        for b, t in ((1, 1), (1, 2), (2, 1), (2, 3)):
            c, h, w = 8, 4, 6
            x = torch.randn(
                (b, c, t, h, w), dtype=torch.float64, generator=self.generator
            ).contiguous(memory_format=torch.channels_last_3d)

            flat = flatten_time_into_batch(x)
            legacy = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            self.assertEqual(flat.data_ptr(), x.data_ptr(), msg=f"b={b} t={t}")
            torch.testing.assert_close(flat, legacy, rtol=0, atol=0)
            self.assertEqual(
                flat.stride(), (c * h * w, 1, c * w, c), msg=f"b={b} t={t}"
            )

            back = unflatten_batch_into_time(flat, b)
            torch.testing.assert_close(back, x, rtol=0, atol=0)
            self.assertTrue(
                back.is_contiguous(memory_format=torch.channels_last_3d),
                msg=f"round trip lost the layout at b={b} t={t}",
            )

    def test_time_batch_flatten_matches_legacy_for_contiguous(self):
        """With channels_last off, both helpers must be the old expressions."""
        from sglang.multimodal_gen.runtime.layers.causal_conv3d_cache import (
            flatten_time_into_batch,
            unflatten_batch_into_time,
        )

        b, c, t, h, w = 1, 8, 1, 4, 6
        x = torch.randn((b, c, t, h, w), dtype=torch.float64, generator=self.generator)
        flat = flatten_time_into_batch(x)
        legacy = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        torch.testing.assert_close(flat, legacy, rtol=0, atol=0)
        self.assertEqual(flat.stride(), legacy.stride())
        torch.testing.assert_close(
            unflatten_batch_into_time(flat, b), x, rtol=0, atol=0
        )

    def test_interleave_time_matches_legacy_reshape(self):
        """Both layouts must produce the same values, and keep their layout."""
        for t in (1, 2, 3):
            base = torch.randn(
                (2, 8, t, 4, 5), dtype=torch.float64, generator=self.generator
            )
            b, c2, _, h, w = base.shape
            c = c2 // 2
            legacy = base.reshape(b, 2, c, t, h, w)
            legacy = torch.stack((legacy[:, 0], legacy[:, 1]), 3).reshape(
                b, c, t * 2, h, w
            )

            contiguous_out = interleave_time(base)
            torch.testing.assert_close(contiguous_out, legacy, rtol=0, atol=0)
            # A contiguous t == 1 input gets a free view from rearrange; an
            # explicit buffer would turn that into a copy.
            self.assertEqual(
                contiguous_out.stride(),
                torch.reshape(
                    base.reshape(b, 2, c, t, h, w).permute(0, 2, 3, 1, 4, 5),
                    (b, c, t * 2, h, w),
                ).stride(),
                msg=f"contiguous path changed its strides at t={t}",
            )

            channels_last = base.contiguous(memory_format=torch.channels_last_3d)
            cl_out = interleave_time(channels_last)
            torch.testing.assert_close(cl_out, legacy, rtol=0, atol=0)
            self.assertTrue(
                cl_out.is_contiguous(memory_format=torch.channels_last_3d),
                msg=f"channels_last path lost the layout at t={t}",
            )

    def test_cache_scope_is_restored_after_exception(self):
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        x = torch.randn((1, 4, 1, 5, 7), dtype=torch.float64, generator=self.generator)
        with self.assertRaises(RuntimeError):
            with causal_cache_scope(CausalConvCache(), [conv]):
                conv(x)
                raise RuntimeError("boom")
        self.assertIsNone(conv._active_cache)
        # A fresh scope must not observe the aborted pass.
        cache = CausalConvCache()
        with causal_cache_scope(cache, [conv]):
            conv(x)
        self.assertEqual(cache.chunk_index, 0)
        self.assertTrue(cache.contains("conv"))

    def test_nested_scope_hands_back_the_outer_cache(self):
        """A tiled encode opens an inner scope inside an outer forward pass."""
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        outer, inner = CausalConvCache(), CausalConvCache()
        with causal_cache_scope(outer, [conv]):
            self.assertIs(conv._active_cache, outer)
            with causal_cache_scope(inner, [conv]):
                self.assertIs(conv._active_cache, inner)
            self.assertIs(conv._active_cache, outer)
        self.assertIsNone(conv._active_cache)

    def test_platform_capabilities_are_resolved_at_construction(self):
        """Querying the platform per call breaks the compiled graph."""
        conv = CausalConv3d(4, 4, 3, padding=1)
        self.assertIsInstance(conv.channels_last_supported, bool)
        self.assertIsInstance(conv.needs_dtype_cast, bool)

    def test_cached_tail_does_not_pin_the_activation(self):
        """The retained tail must be a copy, not a view of the whole chunk."""
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        x = torch.randn((1, 4, 8, 5, 7), dtype=torch.float64, generator=self.generator)
        cache = CausalConvCache()
        with causal_cache_scope(cache, [conv]):
            conv(x)
        tail = cache.get("conv")
        self.assertEqual(tail.shape[2], conv.cache_frames)
        self.assertLess(tail.untyped_storage().nbytes(), x.numel() * x.element_size())


class TestConvImplIsTheOnlyBackendSeam(unittest.TestCase):
    """Backends swap out `_conv_impl`; that must not bypass the cache.

    ROCm's `optimize_vae` replaces the convolution with a temporally unfolded
    batched Conv2D. It used to replace `forward` wholesale, which after this
    refactor would have skipped the cache update entirely and silently produced
    wrong output rather than merely slower output.
    """

    def setUp(self):
        self.generator = torch.Generator().manual_seed(99)

    def test_patching_conv_impl_still_updates_the_cache(self):
        conv = CausalConv3d(4, 4, 3, padding=1).double().eval()
        conv.cache_key = "conv"
        calls = []
        real_conv_impl = conv._conv_impl

        def spy(x):
            calls.append(x.shape[2])
            return real_conv_impl(x)

        conv._conv_impl = spy
        chunks = _make_chunks((1, 4, 1, 5, 7), 3, generator=self.generator)
        _run_managed(conv, chunks)

        # Every chunk must reach the backend with its cached frames prepended.
        self.assertEqual(calls, [1 + conv.cache_frames] * len(chunks))

    def test_spatial_padding_is_reported_for_the_backend(self):
        """A backend padding by hand needs to know what the conv would have used."""
        self.assertEqual(CausalConv3d(4, 4, 3, padding=1).spatial_padding(), (0, 1, 1))
        self.assertEqual(CausalConv3d(4, 4, 1).spatial_padding(), (0, 0, 0))
        self.assertEqual(
            TimeUpsampleCausalConv3d(
                4, 8, (3, 1, 1), padding=(1, 0, 0)
            ).spatial_padding(),
            (0, 0, 0),
        )

    def test_rocm_conv2d_decomposition_matches_conv3d(self):
        """The ROCm unfold path must agree with the conv it replaces.

        Checked here rather than only on a ROCm runner: the decomposition is
        plain torch, so its arithmetic is verifiable anywhere.
        """
        from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform

        conv = CausalConv3d(4, 6, 3, padding=1).double().eval()
        weight_2d = (
            conv.weight.data.permute(0, 2, 1, 3, 4)
            .reshape(conv.out_channels, 3 * conv.in_channels, 3, 3)
            .contiguous()
        )
        x = torch.randn((1, 4, 5, 6, 7), dtype=torch.float64, generator=self.generator)
        expected = F.conv3d(x, conv.weight, conv.bias, conv.stride, (0, 1, 1))
        produced = RocmPlatform._conv3d_as_batched_conv2d(
            x, weight_2d, conv.bias, conv.stride, 3, spatial_padding=(1, 1)
        )
        torch.testing.assert_close(produced, expected)


class TestWanVaeCacheLifecycle(unittest.TestCase):
    """End-to-end properties of the cache across a whole tiny Wan VAE."""

    def setUp(self):
        self.generator = torch.Generator().manual_seed(4321)

    def _latents(self, vae, num_frames):
        return torch.randn(
            (1, vae.z_dim, num_frames, 4, 4),
            dtype=torch.float64,
            generator=self.generator,
        )

    def test_every_cache_key_is_assigned_and_unique(self):
        vae = _tiny_wan(is_residual=False)
        keys = [
            m.cache_key
            for m in vae.modules()
            if isinstance(m, CausalConv3d) and m.cache_frames > 0
        ]
        self.assertTrue(keys)
        self.assertTrue(all(keys), msg="every cached conv needs a key")
        self.assertEqual(len(keys), len(set(keys)), msg="keys must be unique")
        # The pointwise convs are deliberately left out of the cache.
        self.assertEqual(vae.quant_conv.cache_frames, 0)
        self.assertEqual(vae.post_quant_conv.cache_frames, 0)

    def test_decode_frame_count_follows_the_temporal_ratio(self):
        """A wrong first-chunk flag would make every chunk trim, shrinking output."""
        for is_residual in (False, True):
            vae = _tiny_wan(is_residual=is_residual)
            ratio = vae.temporal_compression_ratio
            for num_latent_frames in (1, 2, 4):
                with torch.no_grad():
                    out = vae.decode(self._latents(vae, num_latent_frames))
                self.assertEqual(
                    out.shape[2],
                    (num_latent_frames - 1) * ratio + 1,
                    msg=f"is_residual={is_residual} n={num_latent_frames}",
                )

    def test_causal_decode_in_chunks_matches_one_shot(self):
        """The core streaming invariant: splitting a clip must not change it."""
        for is_residual in (False, True):
            vae = _tiny_wan(is_residual=is_residual)
            latents = self._latents(vae, 4)
            with torch.no_grad():
                vae.reset_causal_decode_state()
                whole = vae.causal_decode(latents)
                vae.reset_causal_decode_state()
                parts = [
                    vae.causal_decode(latents[:, :, :1]),
                    vae.causal_decode(latents[:, :, 1:3]),
                    vae.causal_decode(latents[:, :, 3:]),
                ]
                vae.reset_causal_decode_state()
            torch.testing.assert_close(torch.cat(parts, dim=2), whole, rtol=0, atol=0)

    def test_modes_do_not_leak_into_each_other(self):
        """A stateless pass in between must not perturb the streaming result."""
        vae = _tiny_wan(is_residual=False)
        latents = self._latents(vae, 3)
        with torch.no_grad():
            clean = vae.decode(latents)
            vae._decode(latents)
            after_stateless = vae.decode(latents)
        torch.testing.assert_close(after_stateless, clean, rtol=0, atol=0)

    def test_encode_decode_round_trip_shapes(self):
        for is_residual in (False, True):
            vae = _tiny_wan(is_residual=is_residual)
            pixels = torch.randn(
                (1, 3, 9, 16, 16), dtype=torch.float64, generator=self.generator
            )
            with torch.no_grad():
                latents = vae.encode(pixels).mode()
                out = vae.decode(latents)
            self.assertEqual(latents.shape[1], vae.z_dim)
            self.assertEqual(out.shape[0], 1)
            self.assertEqual(out.shape[1], 3)
            self.assertEqual(out.shape[2], pixels.shape[2])


if __name__ == "__main__":
    unittest.main()
