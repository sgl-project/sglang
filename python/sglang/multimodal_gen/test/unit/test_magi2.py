# SPDX-License-Identifier: Apache-2.0
import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.magi2 import (
    Magi2TextEncoderArchConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.magi2 import Magi2PipelineConfig
from sglang.multimodal_gen.configs.sample.magi2 import (
    MAGI2_NEGATIVE_PROMPT,
    Magi2SamplingParams,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.layers.attention import magi2_sink_attention
from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import SDPABackend
from sglang.multimodal_gen.runtime.layers.attention.magi2_block_grid_attention import (
    SEQ_BUCKET,
    Magi2BlockGrid,
    Magi2BlockGridAttention,
    block_scan_order,
    build_mask_mod,
    cached_block_mask,
)
from sglang.multimodal_gen.runtime.layers.attention.magi2_sink_attention import (
    Magi2SinkAttention,
)
from sglang.multimodal_gen.runtime.layers.moe_multihead import (
    MAX_KERNEL_EXPERTS,
    Magi2MultiHeadExperts,
    Magi2MultiHeadRouter,
)
from sglang.multimodal_gen.runtime.loader.component_loaders import magi2_loader
from sglang.multimodal_gen.runtime.loader.component_loaders.magi2_loader import (
    _is_ep_sharded,
)
from sglang.multimodal_gen.runtime.models.dits.magi2_common import (
    MAGI2_MODALITY_AUDIO,
    MAGI2_MODALITY_TEXT,
    MAGI2_MODALITY_VIDEO,
    Magi2FourierRope,
    Magi2ModalityLinear,
    Magi2ModalityRMSNorm,
    apply_partial_rope,
    modality_runs,
    pad_rows_to_multiple,
    sharded_cu_seqlens,
)
from sglang.multimodal_gen.runtime.models.dits.magi2_mhc import sinkhorn_knopp
from sglang.multimodal_gen.runtime.models.dits.magi2_preview import Magi2PreviewDiT
from sglang.multimodal_gen.runtime.models.dits.magi2_refiner import Magi2RefinerDiT
from sglang.multimodal_gen.runtime.pipelines.magi2 import Magi2Pipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    build_layout,
    build_timesteps,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    coords as magi2_coords,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    ref_images,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    renoise as magi2_renoise,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.guidance import (
    apply_guidance,
    skimming_mask,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.stages.decoding import (
    Magi2DecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2.stages.denoising import (
    Magi2DenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.warmup_request_builder import lighten_warmup_req


class TestPerTokenTimesteps(unittest.TestCase):
    def _layout(self, ref_counts=()):
        return build_layout(
            video_latent_thw=(2, 4, 6),
            audio_tokens=16,
            text_tokens=8,
            device=torch.device("cpu"),
            ref_patch_counts=ref_counts,
        )

    def test_only_noised_modalities_carry_the_step(self):
        layout = self._layout(ref_counts=[15])
        t = build_timesteps(
            layout=layout,
            video_t=torch.tensor(500.0),
            audio_t=torch.tensor(500.0),
        )
        self.assertTrue(bool((t[layout.video_index] == 500.0).all()))
        self.assertTrue(bool((t[layout.audio_index] == 500.0).all()))
        # Text and reference images are clean conditioning: a non-zero step here
        # presents them to the DiT as partially noised.
        self.assertTrue(bool((t[layout.text_index] == 0.0).all()))
        self.assertTrue(bool((t[layout.ref_special_index] == 0.0).all()))
        self.assertTrue(bool((t[layout.ref_patch_index] == 0.0).all()))


class TestRefImageSegments(unittest.TestCase):
    def test_special_token_routes_as_text_and_patches_as_video(self):
        layout = build_layout(
            video_latent_thw=(2, 4, 6),
            audio_tokens=16,
            text_tokens=8,
            device=torch.device("cpu"),
            ref_patch_counts=[15],
        )
        special = int(layout.ref_special_index[0])
        self.assertEqual(int(layout.modality_ids[special]), 2)
        self.assertTrue(bool((layout.modality_ids[layout.ref_patch_index] == 0).all()))
        # The pooled prompt embedding immediately precedes its own patches.
        self.assertEqual(special + 1, int(layout.ref_patch_index[0]))


class TestFigurePhrase(unittest.TestCase):
    def test_plain_prompt_gains_the_phrase(self):
        out = ref_images.ensure_figure_phrase("a lake at dawn")
        self.assertIn(ref_images.FIGURE_ONE, out)


class TestResizePad(unittest.TestCase):
    def test_letterboxes_without_cropping(self):
        from PIL import Image

        source = Image.new("RGB", (200, 100), (10, 20, 30))
        out = ref_images.resize_pad(source, height=128, width=128)
        self.assertEqual(out.size, (128, 128))
        # A crop would fill the canvas; letterboxing leaves white bars.
        corner = out.getpixel((0, 0))
        self.assertEqual(corner, (255, 255, 255))

    def test_reference_asset_reaches_the_reference_latent_grid(self):
        from PIL import Image

        # The reference run logs `images: (1, 1, 48, 1, 30, 56)` for this asset at
        # the 512x896 preview stage, so the chain must land on 30x56 latent
        # patches: aligning to 16 instead of 32 gives 31 rows and misses.
        source = Image.new("RGB", (2710, 1510))
        height, width = ref_images.target_size(
            source, generation_height=512, generation_width=896
        )
        self.assertEqual((height, width), (499, 896))

        height, width = ref_images.aligned_size(height, width, align=32)
        self.assertEqual((height, width), (480, 896))
        self.assertEqual((height // 16, width // 16), (30, 56))


class TestShippedDefaults(unittest.TestCase):
    def test_negative_prompt_covers_video_audio_and_voice(self):
        params = Magi2SamplingParams(prompt="x")
        self.assertEqual(params.negative_prompt, MAGI2_NEGATIVE_PROMPT)
        # Audio guidance runs at the highest scale in the model, so its negatives
        # have to be present.
        for marker in ("JPEG compression residue", "hiss", "monotone"):
            self.assertIn(marker, params.negative_prompt)

    def test_skimmed_guidance_is_off_by_default(self):
        self.assertFalse(Magi2SamplingParams(prompt="x").use_skimmed_guidance)

    def test_text_conditioning_skips_two_layers(self):
        self.assertEqual(Magi2TextEncoderArchConfig().skip_layer, 2)


class TestVideoRequestLowering(unittest.TestCase):
    """The video API is the only public entry point, so its synthesized timing
    defaults have to reach this fixed-length model as *unset*."""

    def _lower(self, request):
        generic = {
            "prompt": request.prompt,
            "fps": request.fps if request.fps is not None else 24,
            "num_frames": request.num_frames,
        }
        if generic["num_frames"] is None:
            generic["num_frames"] = generic["fps"] * (request.seconds or 4)
        return Magi2SamplingParams.lower_video_request_kwargs(request, generic)

    def test_bare_prompt_request_keeps_the_shipped_clip_length(self):
        lowered = self._lower(VideoGenerationsRequest(prompt="x"))
        self.assertNotIn("num_frames", lowered)
        self.assertNotIn("fps", lowered)
        params = Magi2SamplingParams(**lowered)
        self.assertEqual((params.num_frames, params.fps), (249, 25))

    def test_explicit_timing_is_preserved(self):
        lowered = self._lower(
            VideoGenerationsRequest(prompt="x", num_frames=241, fps=24)
        )
        self.assertEqual((lowered["num_frames"], lowered["fps"]), (241, 24))


COLUMNS = 9


class TestMagi2VideoAndTextCoords(unittest.TestCase):
    def test_text_is_offset_before_time_zero_with_a_singleton_reference(self):
        num_tokens = 7
        out = magi2_coords.text_coords(num_tokens=num_tokens)

        self.assertEqual(out.shape, (num_tokens, COLUMNS))
        self.assertTrue(
            torch.equal(
                out[:, 0],
                torch.arange(num_tokens, dtype=torch.float32) - num_tokens,
            )
        )
        self.assertLess(float(out[:, 0].max()), 0.0)
        self.assertEqual(out[:, 6].unique().tolist(), [1.0])


class TestMagi2AudioCoords(unittest.TestCase):
    def test_reference_time_compresses_by_eight_with_ceiling(self):
        for num_tokens in (1, 8, 9, 16, 17, 348):
            with self.subTest(num_tokens=num_tokens):
                out = magi2_coords.audio_coords(num_tokens=num_tokens)
                self.assertEqual(out.shape, (num_tokens, COLUMNS))
                self.assertEqual(float(out[0, 6]), float((num_tokens - 1) // 8 + 1))
                self.assertEqual(float(out[0, 3]), float(num_tokens))


class TestMagi2RefImageCoords(unittest.TestCase):
    def test_reference_images_sit_past_the_clip_with_the_documented_gap(self):
        video_time_steps = 6
        parts = magi2_coords.ref_image_coords(
            token_counts=[9, 4],
            feat_shapes=[(3, 3), (2, 2)],
            video_time_steps=video_time_steps,
        )
        self.assertEqual(len(parts), 4)

        for index, (sentinel, grid) in enumerate(zip(parts[::2], parts[1::2])):
            want_time = video_time_steps + magi2_coords.REF_IMAGE_TIME_GAP + index
            self.assertEqual(float(sentinel[0, 0]), float(want_time))
            self.assertEqual(grid[:, 0].unique().tolist(), [float(want_time)])
            self.assertGreater(want_time, video_time_steps)

    def test_sentinel_row_marks_the_whole_image_with_negative_positions(self):
        sentinel, grid = magi2_coords.ref_image_coords(
            token_counts=[9], feat_shapes=[(3, 3)], video_time_steps=4
        )
        self.assertEqual(sentinel.shape, (1, COLUMNS))
        self.assertEqual(sentinel[0, 1:3].tolist(), [-1.0, -1.0])
        self.assertEqual(sentinel[0, 3:6].tolist(), [1.0, 3.0, 3.0])
        self.assertTrue(torch.equal(sentinel[0, 3:6], sentinel[0, 6:9]))
        self.assertTrue((grid[:, 1:3] >= 0).all())


NUM_BANDS = 16


class TestMagi2FourierRope(unittest.TestCase):
    def test_width_is_six_bands_and_splits_into_equal_halves(self):
        rope = Magi2FourierRope(NUM_BANDS)
        embedding = rope(magi2_coords.video_coords(latent_shape=(3, 4, 5)))

        self.assertEqual(embedding.shape, (3 * 4 * 5, 6 * NUM_BANDS))
        sin, cos = embedding.tensor_split(2, -1)
        self.assertEqual(sin.shape[-1], 3 * NUM_BANDS)
        self.assertEqual(cos.shape[-1], 3 * NUM_BANDS)
        self.assertTrue(torch.allclose(sin.pow(2) + cos.pow(2), torch.ones_like(sin)))

    def test_text_coordinates_are_exactly_unrotated(self):
        rope = Magi2FourierRope(NUM_BANDS)
        with torch.no_grad():
            rope.bands.copy_(torch.rand(NUM_BANDS) + 0.5)

        embedding = rope(magi2_coords.text_coords(num_tokens=11))
        sin, cos = embedding.tensor_split(2, -1)
        self.assertTrue(torch.equal(sin, torch.zeros_like(sin)))
        self.assertTrue(torch.equal(cos, torch.ones_like(cos)))

    def test_singleton_axis_does_not_divide_by_zero(self):
        rope = Magi2FourierRope(NUM_BANDS)
        embedding = rope(magi2_coords.video_coords(latent_shape=(1, 1, 1)))
        self.assertTrue(torch.isfinite(embedding).all())


class TestApplyPartialRope(unittest.TestCase):
    def _cos_sin(self, tokens, rotary_half):
        torch.manual_seed(0)
        angle = torch.randn(tokens, rotary_half) * 2.0
        return angle.cos(), angle.sin()

    def test_channels_beyond_the_rotary_span_are_bit_identical(self):
        tokens, heads, head_dim, rotary_dim = 9, 4, 128, 6 * NUM_BANDS
        x = torch.randn(tokens, heads, head_dim)
        cos, sin = self._cos_sin(tokens, rotary_dim // 2)

        out = apply_partial_rope(x, cos, sin)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.equal(out[..., rotary_dim:], x[..., rotary_dim:]))
        self.assertFalse(torch.equal(out[..., :rotary_dim], x[..., :rotary_dim]))


class TestZeroSnrSigmas(unittest.TestCase):
    def test_schedule_descends_to_zero_signal(self):
        sigmas = magi2_renoise.zero_snr_sigmas()

        self.assertEqual(sigmas.shape, (magi2_renoise.NUM_TRAIN_TIMESTEPS,))
        self.assertTrue((sigmas.diff() < 0).all())
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertAlmostEqual(float(sigmas[0]), 0.999575, places=5)

    def test_production_index_selects_the_reference_signal_coefficient(self):
        # The order is only observable through the index the refiner actually
        # uses. An ascending table holds 0.152 here, which renoises to 98.8%
        # noise instead of 54.4% and quietly flattens the output.
        sigmas = magi2_renoise.zero_snr_sigmas()
        self.assertAlmostEqual(float(sigmas[220]), 0.838910, places=5)


class TestUpsampleLatent(unittest.TestCase):
    def test_time_axis_maps_t_frames_to_two_t_minus_one(self):
        for frames in (1, 3, 8):
            with self.subTest(frames=frames):
                out = magi2_renoise.upsample_latent(
                    torch.randn(1, 4, frames, 8, 8), height=16, width=16
                )
                self.assertEqual(out.shape, (1, 4, 2 * frames - 1, 16, 16))

    def test_aligned_corners_keep_original_frames_on_even_slots(self):
        latent = torch.randn(1, 2, 4, 4, 4)
        out = magi2_renoise.upsample_latent(latent, height=4, width=4)
        self.assertTrue(torch.allclose(out[:, :, ::2], latent, atol=1e-5))


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    return (
        torch.randn(4, 8),
        torch.randn(4, 8),
        torch.randn(4, 8),
    )


def test_plain_guidance_is_the_textbook_formula():
    latent, cond, uncond = _inputs()
    out = apply_guidance(latent=latent, cond=cond, uncond=uncond, guidance_scale=5.0)
    torch.testing.assert_close(out, uncond + 5.0 * (cond - uncond))


def test_skimming_changes_the_result_but_only_where_masked():
    latent, cond, uncond = _inputs()
    plain = apply_guidance(latent=latent, cond=cond, uncond=uncond, guidance_scale=5.0)
    skimmed = apply_guidance(
        latent=latent,
        cond=cond,
        uncond=uncond,
        guidance_scale=5.0,
        skimmed_scale=3.0,
    )

    # Two rounds, so the bound is the union of both masks: the second round
    # recomputes from the damped uncond with the mask's arguments swapped, which
    # is how it reaches elements the first round's sign tests excluded.
    first = skimming_mask(latent=latent, cond=cond, uncond=uncond, guidance_scale=5.0)
    damped = torch.where(first, torch.lerp(cond, uncond, 0.5), uncond)
    second = skimming_mask(latent=latent, cond=damped, uncond=cond, guidance_scale=5.0)
    touched = first | second

    assert first.any() and not first.all()
    assert touched.any() and not touched.all()
    torch.testing.assert_close(plain[~touched], skimmed[~touched])
    assert not torch.allclose(plain[touched], skimmed[touched])


class _Plan:
    """Stands in for SpShard, which needs a live process group to build."""

    def __init__(self, orig_len: int, local_len: int, num_pad: int, sp_size: int):
        self.orig_len = orig_len
        self.local_len = local_len
        self.num_pad = num_pad
        self.sp_size = sp_size


class TestModalityRuns(unittest.TestCase):
    def test_contiguous_segments_collapse_to_one_run_each(self):
        ids = torch.tensor(
            [MAGI2_MODALITY_VIDEO] * 5
            + [MAGI2_MODALITY_AUDIO] * 3
            + [MAGI2_MODALITY_TEXT] * 2
        )
        self.assertEqual(
            modality_runs(ids),
            [
                (0, 5, MAGI2_MODALITY_VIDEO),
                (5, 3, MAGI2_MODALITY_AUDIO),
                (8, 2, MAGI2_MODALITY_TEXT),
            ],
        )


class TestShardedCuSeqlens(unittest.TestCase):
    def test_padding_becomes_its_own_segment(self):
        # 62 real rows over 4 ranks pads to 64, and the 2 pad rows must not share
        # a segment with real tokens or they would attract attention mass.
        cu, max_seqlen = sharded_cu_seqlens(
            plan=_Plan(orig_len=62, local_len=16, num_pad=2, sp_size=4),
            device=torch.device("cpu"),
        )
        self.assertEqual(cu.tolist(), [0, 62, 64])
        self.assertEqual(max_seqlen, 62)


# 1080p refiner grid: 32 latent frames of 68x120.
_VIDEO_TOKENS = 32 * 68 * 120


class TestSequenceBucket(unittest.TestCase):
    def test_whole_prompt_range_collapses_to_a_handful_of_lengths(self):
        # Each distinct length costs a retained compiled graph, so what matters is
        # the count over every prompt the encoder accepts, not any one length.
        lengths = {
            total + -total % SEQ_BUCKET
            for total in (_VIDEO_TOKENS + text for text in range(7001))
        }
        self.assertLess(len(lengths), 16)

    def test_bucket_absorbs_the_shard_padding(self):
        # Only the refiner uses the block grid (grid-local window attention), and
        # its 8 GQA kv heads cap sp there at these degrees. Preview-only 3/6/12
        # do not divide the bucket, which only costs shard padding.
        for sp_size in (1, 2, 4, 8):
            with self.subTest(sp_size=sp_size):
                self.assertEqual(SEQ_BUCKET % sp_size, 0)

    def test_padding_repeats_the_last_row(self):
        rows = torch.arange(SEQ_BUCKET + 5, dtype=torch.float32)[:, None]
        (padded,), num_pad = pad_rows_to_multiple(rows, multiple=SEQ_BUCKET)

        self.assertEqual(num_pad, SEQ_BUCKET - 5)
        self.assertEqual(padded.shape[0] % SEQ_BUCKET, 0)
        self.assertTrue(torch.equal(padded[: rows.shape[0]], rows))
        # Repeated, not zeroed: a zero coordinate row is a valid grid position.
        self.assertTrue(bool((padded[rows.shape[0] :] == rows[-1]).all()))


def _expected_allowed(grid: Magi2BlockGrid, order: torch.Tensor) -> torch.Tensor:
    """Ground-truth neighborhood: block membership from the raster token
    ``order`` points at, whereas the mask derives it from the position."""
    t_dim, h_dim, w_dim = grid.latent_thw
    block_t, block_h, block_w = grid.block_thw
    seq = grid.seq_len

    allowed = torch.zeros(seq, seq, dtype=torch.bool)
    coords = []
    for position in range(seq):
        raster = int(order[position])
        if raster >= grid.num_video_tokens:
            coords.append(None)
            continue
        t = raster // (h_dim * w_dim)
        h = raster // w_dim % h_dim
        w = raster % w_dim
        coords.append((t // block_t, h // block_h, w // block_w))

    for q in range(seq):
        for kv in range(seq):
            if coords[q] is None or coords[kv] is None:
                allowed[q, kv] = True
                continue
            allowed[q, kv] = all(
                abs(a - b) <= r
                for a, b, r in zip(coords[q], coords[kv], grid.radius_thw)
            )
    return allowed


def _mask_matrix(grid: Magi2BlockGrid) -> torch.Tensor:
    mask_mod = build_mask_mod(grid=grid, device=torch.device("cpu"))
    positions = torch.arange(grid.seq_len)
    q_idx = positions[:, None].expand(grid.seq_len, grid.seq_len)
    kv_idx = positions[None, :].expand(grid.seq_len, grid.seq_len)
    return mask_mod(None, None, q_idx, kv_idx)


class TestBlockScanOrder(unittest.TestCase):
    def setUp(self):
        # The block grid has to be wider than the radius on some axis, or every
        # block is a neighbor of every other and the mask is all-True, which makes
        # any ordering look correct.
        self.grid = Magi2BlockGrid(
            latent_thw=(4, 8, 8),
            block_thw=(2, 2, 2),
            radius_thw=(1, 1, 1),
            num_tail_tokens=3,
        )

    def assert_restrictive(self, mask: torch.Tensor) -> None:
        video = self.grid.num_video_tokens
        self.assertTrue(mask[:video, :video].any())
        self.assertFalse(mask[:video, :video].all())

    def test_permutation_is_a_bijection_with_the_tail_appended(self):
        order, restore = block_scan_order(grid=self.grid, device=torch.device("cpu"))
        self.assertEqual(order.numel(), self.grid.seq_len)
        self.assertEqual(sorted(order.tolist()), list(range(self.grid.seq_len)))
        self.assertTrue(torch.equal(order[restore], torch.arange(order.numel())))
        # The tail must stay put: only video tokens live on the block grid.
        tail = torch.arange(self.grid.num_video_tokens, self.grid.seq_len)
        self.assertTrue(torch.equal(order[-tail.numel() :], tail))

    def test_mask_matches_the_grid_neighborhood_in_scan_order(self):
        order, _ = block_scan_order(grid=self.grid, device=torch.device("cpu"))
        mask = _mask_matrix(self.grid)
        self.assert_restrictive(mask)
        self.assertTrue(torch.equal(mask, _expected_allowed(self.grid, order)))

    def test_mask_is_wrong_without_the_permutation(self):
        # The mask assigns a token's block from its position, so feeding it a
        # raster-ordered sequence silently attends over the wrong neighborhood.
        identity = torch.arange(self.grid.seq_len)
        mask = _mask_matrix(self.grid)
        self.assert_restrictive(mask)
        self.assertFalse(torch.equal(mask, _expected_allowed(self.grid, identity)))

    def test_shard_padding_is_not_attendable(self):
        # Pad rows repeat the last real row, and the tail is globally visible, so
        # leaving them unmasked lets every query attend to a duplicate.
        padded = Magi2BlockGrid(
            latent_thw=(4, 8, 8),
            block_thw=(2, 2, 2),
            radius_thw=(1, 1, 1),
            num_tail_tokens=3,
            num_pad_tokens=2,
        )
        mask = _mask_matrix(padded)
        valid = padded.num_valid_tokens
        self.assertEqual(mask.shape[0], valid + 2)
        self.assertFalse(mask[:, valid:].any())
        # Pad rows keep their own keys: a fully masked query row softmaxes over
        # all -inf and returns NaN. Their outputs are dropped on gather instead.
        self.assertTrue(mask[valid:, :valid].any())
        self.assertTrue(torch.equal(mask[:valid, :valid], _mask_matrix(self.grid)))

    def test_factory_forwards_the_pad_count(self):
        # Built through from_arch_config, not the struct: dropping num_pad_tokens
        # here sizes the mask short of the tensor, and flex_attention rejects it
        # only for prompts whose token count is not a multiple of the world size.
        from sglang.multimodal_gen.configs.models.dits.magi2 import (
            Magi2RefinerArchConfig,
        )

        grid = Magi2BlockGrid.from_arch_config(
            arch_config=Magi2RefinerArchConfig(),
            latent_thw=(4, 8, 8),
            num_tail_tokens=3,
            num_pad_tokens=2,
        )
        self.assertEqual(grid.num_pad_tokens, 2)
        self.assertEqual(grid.seq_len, grid.num_valid_tokens + 2)


def _grid(num_tail: int, num_pad: int) -> Magi2BlockGrid:
    return Magi2BlockGrid(
        latent_thw=(4, 8, 8),
        block_thw=(2, 2, 2),
        radius_thw=(1, 1, 1),
        num_tail_tokens=num_tail,
        num_pad_tokens=num_pad,
    )


def _closed_over_ints(fn) -> list[int]:
    return [
        cell.cell_contents
        for cell in (fn.__closure__ or ())
        if isinstance(cell.cell_contents, int)
        and not isinstance(cell.cell_contents, bool)
    ]


class TestMaskModGuards(unittest.TestCase):
    def test_no_captured_int_varies_with_prompt_length(self):
        # dynamo guards on the value of a captured int, so any int that follows the
        # prompt recompiles flex_attention per prompt length and defeats the
        # sequence bucketing. Only the tier-fixed video token count may be an int.
        device = torch.device("cpu")
        seen = {
            tuple(
                _closed_over_ints(build_mask_mod(grid=_grid(t, 64 - t), device=device))
            )
            for t in (5, 17, 40)
        }
        self.assertEqual(len(seen), 1, f"captured ints vary with prompt length: {seen}")

    def test_valid_length_is_carried_as_a_tensor(self):
        mask_mod = build_mask_mod(grid=_grid(5, 59), device=torch.device("cpu"))
        tensors = [
            cell.cell_contents
            for cell in mask_mod.__closure__
            if isinstance(cell.cell_contents, torch.Tensor)
        ]
        # Bucket-shaped, so only its shape is guarded and the bucket pins that.
        self.assertTrue(any(t.dtype == torch.bool for t in tensors))

    def test_mask_still_bars_pad_as_a_key(self):
        grid = _grid(5, 59)
        mask_mod = build_mask_mod(grid=grid, device=torch.device("cpu"))
        positions = torch.arange(grid.seq_len)
        mask = mask_mod(
            None,
            None,
            positions[:, None].expand(grid.seq_len, grid.seq_len),
            positions[None, :].expand(grid.seq_len, grid.seq_len),
        )
        valid = grid.num_valid_tokens
        self.assertFalse(mask[:, valid:].any())
        self.assertTrue(mask[:valid, :valid].any())


class TestBlockMaskWidth(unittest.TestCase):
    def test_kv_indices_keeps_every_tile_column(self):
        # from_kv_blocks sizes its transposed buffer from kv_indices.shape[-1], so
        # trimming the column count to counts.max() cannot hold the tile ids stored
        # in it and asserts inside the index kernel.
        grid = _grid(0, 0)
        mask = cached_block_mask(grid=grid, device=torch.device("cpu"))
        self.assertEqual(mask.kv_indices.shape[-1], grid.num_tiles)


class TestCompileRegions(unittest.TestCase):
    def test_preview_compiles_whole_blocks_only(self):
        conditions = Magi2PreviewDiT._compile_conditions
        self.assertTrue(conditions)
        self.assertTrue(any(c("blocks.0", object()) for c in conditions))
        # A submodule would compile a region nested inside an outer one.
        self.assertFalse(any(c("blocks.0.attention", object()) for c in conditions))
        self.assertFalse(any(c("blocks", object()) for c in conditions))

    def test_expert_forward_is_opaque_to_dynamo(self):
        # Autotuning these GEMMs asks more shared memory than sm90 has, so a
        # compiled region containing them fails to build.
        self.assertTrue(Magi2MultiHeadExperts.forward._torchdynamo_disable)

    def test_refiner_compiles_whole_blocks_only(self):
        # Both DiTs are compiled, so an empty condition list would make the
        # framework's regional path raise on the refiner pass.
        conditions = Magi2RefinerDiT._compile_conditions
        self.assertTrue(conditions)
        self.assertTrue(any(c("blocks.0", object()) for c in conditions))
        self.assertFalse(any(c("blocks.0.attention", object()) for c in conditions))

    def test_block_grid_attention_is_opaque_to_dynamo(self):
        # It compiles its own mask per bucket; an outer region would recompile it
        # per sequence length instead.
        self.assertTrue(Magi2BlockGridAttention.forward._torchdynamo_disable)


NUM_MODALITY = 3


def _modality_ids(num_tokens, *, seed=0):
    torch.manual_seed(seed)
    ids = torch.randint(0, NUM_MODALITY, (num_tokens,))
    ids[:NUM_MODALITY] = torch.arange(NUM_MODALITY)
    return ids


class TestMagi2ModalityLinear(unittest.TestCase):
    def test_matches_a_per_row_loop_over_each_rows_own_weight_block(self):
        in_features, out_features, num_tokens = 8, 5, 12
        layer = Magi2ModalityLinear(
            in_features, out_features, num_modality=NUM_MODALITY
        ).requires_grad_(False)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(NUM_MODALITY * out_features, in_features))

        x = torch.randn(num_tokens, in_features)
        ids = _modality_ids(num_tokens)
        blocks = layer.weight.view(NUM_MODALITY, out_features, in_features)
        want = torch.stack(
            [F.linear(x[row], blocks[int(ids[row])]) for row in range(num_tokens)]
        )

        self.assertTrue(torch.allclose(layer(x, ids), want, atol=1e-6))

    def test_different_modalities_use_different_weights(self):
        layer = Magi2ModalityLinear(4, 4, num_modality=NUM_MODALITY)
        with torch.no_grad():
            layer.weight.copy_(torch.randn(NUM_MODALITY * 4, 4))

        row = torch.randn(1, 4)
        x = row.repeat(NUM_MODALITY, 1)
        out = layer(x, torch.arange(NUM_MODALITY))
        for modality in range(1, NUM_MODALITY):
            self.assertFalse(torch.equal(out[0], out[modality]))


class TestMagi2ModalityRMSNorm(unittest.TestCase):
    def test_broadcasts_per_token_weight_over_a_qk_norm_head_axis(self):
        tokens, heads, head_dim = 10, 4, 8
        norm = Magi2ModalityRMSNorm(head_dim, num_modality=NUM_MODALITY)
        with torch.no_grad():
            norm.weight.copy_(torch.randn(NUM_MODALITY * head_dim) * 0.3 + 1.0)

        x = torch.randn(tokens, heads, head_dim)
        ids = _modality_ids(tokens)
        out = norm(x, ids)
        self.assertEqual(out.shape, x.shape)

        blocks = norm.weight.detach().view(NUM_MODALITY, head_dim)
        for token in range(tokens):
            row = x[token].float()
            row = row * torch.rsqrt(row.pow(2).mean(-1, keepdim=True) + norm.eps)
            want = row * blocks[int(ids[token])]
            self.assertTrue(torch.allclose(out[token], want, atol=1e-6))


ITERS = 20
EPS = 1e-12


class TestSinkhornKnopp(unittest.TestCase):
    def test_output_is_doubly_stochastic(self):
        torch.manual_seed(0)
        for tokens, n in ((7, 2), (5, 4), (3, 8)):
            with self.subTest(tokens=tokens, n=n):
                out = sinkhorn_knopp(
                    torch.randn(tokens, n, n) * 2.0, num_iters=ITERS, eps=EPS
                )
                ones = torch.ones(tokens, n)
                self.assertTrue(torch.allclose(out.sum(dim=-1), ones, atol=1e-4))
                self.assertTrue(torch.allclose(out.sum(dim=-2), ones, atol=1e-4))
                self.assertTrue((out >= 0).all())


def _make_router(
    *,
    num_heads,
    num_experts,
    head_dim,
    top_k,
    route_scale=1.0,
    route_norm=True,
    seed=0,
):
    torch.manual_seed(seed)
    router = Magi2MultiHeadRouter(
        num_heads=num_heads,
        num_experts=num_experts,
        head_dim=head_dim,
        top_k=top_k,
        route_scale=route_scale,
        route_norm=route_norm,
    )
    with torch.no_grad():
        router.gate.copy_(torch.randn(num_heads * num_experts, head_dim))
    return router.requires_grad_(False)


def _unbiased_scores(router, tokens):
    """Reference per-head sigmoid scores, ``[tokens, heads, experts]``."""
    gate = router.gate.detach().view(
        router.num_heads, router.num_experts, router.head_dim
    )
    return torch.einsum("thd,hed->the", tokens.float(), gate.float()).sigmoid()


class TestMagi2MoERouterSelection(unittest.TestCase):
    def test_each_head_selects_only_from_its_own_expert_bank(self):
        heads, experts, top_k = 4, 8, 2
        router = _make_router(
            num_heads=heads, num_experts=experts, head_dim=16, top_k=top_k
        )
        ids = router(torch.randn(6, heads, 16))[0].view(6, heads, top_k)

        for head in range(heads):
            low, high = head * experts, (head + 1) * experts
            self.assertTrue(((ids[:, head] >= low) & (ids[:, head] < high)).all())
        for row in ids.reshape(-1, top_k):
            self.assertEqual(len(set(row.tolist())), top_k)

    def test_flattened_rows_are_token_major(self):
        num_tokens, heads, experts, top_k = 5, 3, 8, 2
        router = _make_router(
            num_heads=heads, num_experts=experts, head_dim=16, top_k=top_k
        )
        tokens = torch.randn(num_tokens, heads, 16)
        ids, weights = router(tokens)

        scores = _unbiased_scores(router, tokens)
        local = scores.topk(k=top_k, dim=-1).indices
        head_offset = torch.arange(heads).view(1, heads, 1) * experts

        got = ids.view(num_tokens, heads, top_k).sort(dim=-1)
        want = (local + head_offset).sort(dim=-1)
        self.assertTrue(torch.equal(got.values, want.values))

        got_weights = weights.view(num_tokens, heads, top_k).gather(-1, got.indices)
        want_weights = F.normalize(scores.gather(-1, local), p=1, dim=-1).gather(
            -1, want.indices
        )
        self.assertTrue(torch.allclose(got_weights, want_weights, atol=1e-5))


class TestMagi2MoERouterWeights(unittest.TestCase):
    def test_expert_bias_steers_selection_but_never_reaches_the_weights(self):
        heads, experts, top_k = 2, 8, 2
        router = _make_router(
            num_heads=heads,
            num_experts=experts,
            head_dim=16,
            top_k=top_k,
            route_norm=False,
        )
        tokens = torch.randn(6, heads, 16)
        scores = _unbiased_scores(router, tokens)
        base_ids = router(tokens)[0]

        target = scores.sum(dim=0).argmin(dim=-1)
        promoted = target + torch.arange(heads) * experts
        with torch.no_grad():
            router.expert_bias[promoted] = 10.0
        ids, weights = router(tokens)

        self.assertFalse(torch.equal(ids, base_ids))
        hit = ids.view(-1, heads, top_k) == promoted.view(1, heads, 1)
        self.assertTrue(hit.any(dim=-1).all())

        # A leaked +10 bias would put these weights near 10.7; sigmoid caps at 1.
        self.assertLessEqual(float(weights.max()), 1.0)
        want = scores[:, torch.arange(heads), target].unsqueeze(-1).expand_as(hit)
        self.assertTrue(
            torch.allclose(weights.view(-1, heads, top_k)[hit], want[hit], atol=1e-5)
        )

    def test_route_scale_is_applied_exactly_once(self):
        scale = 2.5
        shape = dict(num_heads=2, num_experts=8, head_dim=16, top_k=4)
        scaled = _make_router(**shape, route_scale=scale)
        unit = _make_router(**shape, route_scale=1.0)
        tokens = torch.randn(6, 2, 16)

        scaled_ids, scaled_weights = scaled(tokens)
        unit_ids, unit_weights = unit(tokens)
        totals = scaled_weights.sum(-1)

        self.assertTrue(torch.equal(scaled_ids, unit_ids))
        self.assertTrue(
            torch.allclose(totals, torch.full_like(totals, scale), atol=1e-5)
        )
        self.assertTrue(torch.allclose(scaled_weights, unit_weights * scale, atol=1e-6))


# A rank owns ``(12 / ep_size) * 256`` experts, so ep=1, 2 and 3 cross the MoE
# kernel's 1023-expert ceiling; head chunking is what keeps them under it.
MOE_HEADS = 12
EXPERTS_PER_HEAD = 256


class TestHeadChunking(unittest.TestCase):
    def _experts(self, num_heads: int) -> Magi2MultiHeadExperts:
        return Magi2MultiHeadExperts(
            num_heads=num_heads,
            num_experts_per_head=EXPERTS_PER_HEAD,
            head_dim=256,
            intermediate_size=1280,
        )

    def test_every_ep_degree_stays_under_the_ceiling(self):
        for ep in (1, 2, 3, 4, 6, 12):
            experts = self._experts(MOE_HEADS // ep)
            per_call = experts.heads_per_call * EXPERTS_PER_HEAD
            self.assertLessEqual(
                per_call,
                MAX_KERNEL_EXPERTS,
                f"ep={ep} calls the kernel with {per_call} experts",
            )

    def test_ep4_and_below_still_run_in_one_call(self):
        for ep in (4, 6, 12):
            experts = self._experts(MOE_HEADS // ep)
            self.assertLessEqual(experts.num_heads, experts.heads_per_call)

    def test_ep3_chunks_rather_than_calling_with_1024(self):
        experts = self._experts(MOE_HEADS // 3)
        self.assertEqual(experts.num_local_experts, 1024)
        self.assertGreater(experts.num_heads, experts.heads_per_call)

    def test_single_head_over_the_ceiling_raises(self):
        with self.assertRaises(ValueError):
            Magi2MultiHeadExperts(
                num_heads=1,
                num_experts_per_head=MAX_KERNEL_EXPERTS + 1,
                head_dim=256,
                intermediate_size=1280,
            )


class TestExpertAxisSharding(unittest.TestCase):
    def test_matches_the_checkpoint_spelling_of_every_sharded_tensor(self):
        prefix = "blocks.5.mlp"
        for suffix in (
            "moe_mlp.router.gate",
            "moe_mlp.router.expert_bias",
            "moe_mlp.W_gate",
            "moe_mlp.W_up",
            "moe_mlp.W_down",
        ):
            with self.subTest(suffix=suffix):
                self.assertTrue(_is_ep_sharded(f"{prefix}.{suffix}"))

    def test_model_side_expert_names_are_never_matched(self):
        # w13_weight and w2 exist only after _relayout_experts, which runs once
        # every rank has finished reading. Listing them as sharded suffixes reads
        # as coverage but can never fire, so the names must stay out.
        for suffix in ("moe_mlp.experts.w13_weight", "moe_mlp.experts.w2"):
            with self.subTest(suffix=suffix):
                self.assertFalse(_is_ep_sharded(f"blocks.5.mlp.{suffix}"))

    def test_replicated_tensors_are_not_sharded(self):
        # The dangerous direction: a false positive here row-slices a replicated
        # tensor, so a rank silently loads a fraction of its own weights.
        for name in (
            "blocks.5.mlp.split_linear.weight",
            "blocks.5.mlp.shared_expert_fc1.weight",
            "blocks.5.attention.linear_qkv.weight",
            "pre_adapter.video_embedder.weight",
        ):
            with self.subTest(name=name):
                self.assertFalse(_is_ep_sharded(name))


class _M(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


class _Vae(_M):
    def decode(self, latents):
        return torch.zeros(1, 3, 2, 4, 4)


class _AudioVae(_M):
    def decode(self, latents):
        return torch.zeros(1, 2, 8)


def _validate_args(**overrides):
    args = dict(
        enable_cfg_parallel=False,
        attention_backend=None,
        dp_size=1,
        tp_size=1,
        ring_degree=1,
        num_gpus=4,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class _StageArgsTestCase(unittest.TestCase):
    def setUp(self):
        self._prev_args = server_args_module._global_server_args
        self.config = Magi2PipelineConfig()
        self.server_args = SimpleNamespace(
            pipeline_config=self.config,
            component_precisions={},
            comfyui_mode=False,
            enable_torch_compile=False,
            enable_breakable_cuda_graph=False,
            enable_cfg_parallel=False,
            attention_backend=None,
            dit_layerwise_offload=False,
        )
        set_global_server_args(self.server_args)

    def tearDown(self):
        set_global_server_args(self._prev_args)


class TestPipelineComponentDeclarations(_StageArgsTestCase):
    def _stages(self):
        pipeline = object.__new__(Magi2Pipeline)
        pipeline.server_args = self.server_args
        pipeline._disagg_role = RoleType.MONOLITHIC
        pipeline._stages = []
        pipeline._stage_name_mapping = {}
        pipeline.component_residency_strategies = {}
        pipeline.modules = {
            name: _M()
            for name in (
                "transformer",
                "transformer_2",
                "text_encoder",
                "vae",
                "turbo_vae",
                "audio_vae",
            )
        }
        pipeline.modules["tokenizer"] = object()
        pipeline.create_pipeline_stages(self.server_args)
        return {
            name: [
                (use.component_name, use.start_at_stage_entry)
                for use in stage.component_uses(self.server_args, name)
            ]
            for name, stage in pipeline._stage_name_mapping.items()
        }

    def test_each_stage_declares_the_component_it_runs(self):
        """Undeclared uses leave residency and offload blind to the stage's module."""
        uses = self._stages()
        # Without the pipeline handle the refiner resolves to "transformer" and
        # residency moves the preview DiT around the refiner pass.
        self.assertEqual(uses["magi2_denoising_preview"], [("transformer", True)])
        self.assertEqual(uses["magi2_denoising_refiner"], [("transformer_2", True)])
        # Non-source SP ranks and image-free requests never touch these modules.
        self.assertEqual(uses["Magi2TextEncodingStage"], [("text_encoder", False)])
        self.assertEqual(uses["Magi2ImageEncodingStage"], [("vae", False)])
        self.assertEqual(
            uses["Magi2DecodingStage"], [("turbo_vae", True), ("audio_vae", True)]
        )

    def test_decoder_declaration_follows_the_turbo_switch(self):
        self.config.use_turbo_vae = False
        stage = Magi2DecodingStage(
            video_vae=_Vae(),
            audio_vae=_AudioVae(),
            latents_mean=(0.0,) * 48,
            latents_std=(1.0,) * 48,
        )
        names = [u.component_name for u in stage.component_uses(self.server_args)]
        self.assertEqual(names, ["vae", "audio_vae"])


class TestDecodedFps(_StageArgsTestCase):
    def test_output_fps_matches_the_tier(self):
        """A preview-only clip saved at 25 fps plays 2x fast with audio drifting."""
        stage = Magi2DecodingStage(
            video_vae=_Vae(),
            audio_vae=_AudioVae(),
            turbo_vae=_Vae(),
            latents_mean=(0.0,) * 48,
            latents_std=(1.0,) * 48,
        )
        for refined, want in ((False, 12.5), (True, 25)):
            with self.subTest(refined=refined):
                batch = Req(sampling_params=Magi2SamplingParams(prompt="x"))
                batch.raw_latent_shape = (1, 48, 1, 2, 2)
                batch.latents = torch.zeros(4, 48)
                batch.audio_latents = torch.zeros(3, 4)
                batch.extra["magi2_enable_refiner"] = refined
                batch.extra["magi2_preview_audio_latents"] = torch.zeros(3, 4)
                self.assertEqual(stage.forward(batch, self.server_args).fps, want)


class TestDenoisingStageContract(_StageArgsTestCase):
    def test_real_request_restores_dits_offloaded_for_compile(self):
        """A compile-only layerwise offload must not outlive the warmup."""
        stage = Magi2DenoisingStage(transformer=_M())
        offloaded = mock.Mock()
        stage._offloaded_dit_modules_for_compile = [offloaded]
        batch = Req(sampling_params=Magi2SamplingParams(prompt="x"))
        with mock.patch.object(torch._dynamo, "reset"):
            # The packed layout is absent, so the loop itself fails after the hook.
            with self.assertRaises(KeyError):
                stage.forward(batch, self.server_args)
        offloaded.disable_offload.assert_called_once()

    def test_per_request_attention_overrides_are_rejected(self):
        """The model-owned loop never applies them, so they were silently ignored."""
        stage = Magi2DenoisingStage(transformer=_M())
        for name, value in (
            ("attention_backend_override", "torch_sdpa"),
            ("skip_softmax_params", {"threshold_scale_factor": 1.0}),
        ):
            with self.subTest(name=name):
                params = Magi2SamplingParams(prompt="x")
                setattr(params, name, value)
                with self.assertRaisesRegex(ValueError, name):
                    stage.forward(Req(sampling_params=params), self.server_args)


class TestServerArgsValidation(unittest.TestCase):
    def test_only_flash_attention_is_accepted(self):
        """Other backends drop the preview DiT's sink logits without an error."""
        config = Magi2PipelineConfig()
        for backend in (None, "fa"):
            config.validate_server_args(_validate_args(attention_backend=backend))
        for backend in ("torch_sdpa", "sage_attn"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "FlashAttention"):
                    config.validate_server_args(
                        _validate_args(attention_backend=backend)
                    )

    def test_data_parallel_is_rejected(self):
        """EP groups span the global world, so dp replicas would share experts."""
        with self.assertRaisesRegex(ValueError, "--dp-size"):
            Magi2PipelineConfig().validate_server_args(_validate_args(dp_size=2))

    def test_native_only_components_reach_the_instance(self):
        # An unannotated override leaves the inherited dataclass field default ().
        self.assertEqual(
            set(Magi2PipelineConfig().native_only_components),
            {
                "text_encoder",
                "tokenizer",
                "transformer",
                "transformer_2",
                "vae",
                "turbo_vae",
                "audio_vae",
            },
        )


class TestSinkAttentionBackend(unittest.TestCase):
    def test_a_backend_that_drops_sinks_raises(self):
        """The FA resolver can fall back to SDPA, which ignores sinks and window."""
        attention = Magi2SinkAttention(num_heads=4, head_dim=64, sink_token_num=1)
        with (
            mock.patch.object(
                magi2_sink_attention, "get_attn_backend", return_value=SDPABackend
            ),
            mock.patch.object(
                magi2_sink_attention,
                "get_ulysses_parallel_world_size",
                return_value=1,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "FlashAttention"):
                attention._ensure_impl(torch.bfloat16)


class TestLoaderPrecision(unittest.TestCase):
    _LOADERS = {
        "load_magi2_preview_dit": "transformer",
        "load_magi2_refiner_dit": "transformer_2",
        "load_magi2_text_encoder": "text_encoder",
        "load_magi2_video_vae": "vae",
        "load_magi2_turbo_vae": "turbo_vae",
        "load_magi2_audio_vae": "audio_vae",
    }

    def _loaded_dtypes(self, component_precisions):
        server_args = SimpleNamespace(
            pipeline_config=Magi2PipelineConfig(),
            component_precisions=component_precisions,
        )
        seen = {}

        def recorder(component):
            def load(**kwargs):
                seen[component] = kwargs["param_dtype"]
                return (_M(), object()) if component == "text_encoder" else _M()

            return load

        with contextlib.ExitStack() as stack:
            for loader, component in self._LOADERS.items():
                stack.enter_context(
                    mock.patch.object(magi2_loader, loader, recorder(component))
                )
            stack.enter_context(
                mock.patch.object(
                    magi2_loader,
                    "get_local_torch_device",
                    return_value=torch.device("cpu"),
                )
            )
            magi2_loader.load_magi2_modules(
                server_args=server_args,
                pipeline=SimpleNamespace(model_path="unused"),
                ep_group=None,
            )
        return seen

    def test_defaults_are_unchanged(self):
        bf16, fp32 = torch.bfloat16, torch.float32
        self.assertEqual(
            self._loaded_dtypes({}),
            {
                "transformer": bf16,
                "transformer_2": bf16,
                "text_encoder": bf16,
                "vae": fp32,
                "turbo_vae": bf16,
                "audio_vae": fp32,
            },
        )

    def test_component_precisions_override_only_their_own_entry(self):
        """--component-precisions was ignored by the MAGI-2 loader."""
        defaults = self._loaded_dtypes({})
        for component in self._LOADERS.values():
            with self.subTest(component=component):
                want = dict(defaults)
                want[component] = torch.float16
                # turbo_vae has no precision field of its own and follows the DiT.
                if component == "transformer":
                    want["turbo_vae"] = torch.float16
                self.assertEqual(self._loaded_dtypes({component: "fp16"}), want)


class TestSyntheticWarmup(unittest.TestCase):
    def _warmup_args(self, **overrides):
        args = dict(
            warmup_resolutions=None,
            warmup_num_frames=None,
            num_gpus=1,
            pipeline_config=Magi2PipelineConfig(),
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_default_warmup_runs_the_two_stage_tier_at_full_length(self):
        """The generic probe picked the preview tier, leaving the refiner cold."""
        params = Magi2SamplingParams(prompt="x")
        for num_gpus, frames in ((1, 249), (4, 253), (8, 253)):
            with self.subTest(num_gpus=num_gpus):
                req = SimpleNamespace(width=896, height=512, num_frames=17)
                params.prepare_synthetic_warmup_request_for_queue(
                    req, self._warmup_args(num_gpus=num_gpus)
                )
                self.assertEqual((req.width, req.height), (1920, 1088))
                self.assertEqual(req.num_frames, frames)

    def test_explicit_warmup_shape_is_kept(self):
        params = Magi2SamplingParams(prompt="x")
        req = SimpleNamespace(width=896, height=512, num_frames=17)
        params.prepare_synthetic_warmup_request_for_queue(
            req, self._warmup_args(warmup_resolutions=["896x512"])
        )
        self.assertEqual((req.width, req.height, req.num_frames), (896, 512, 17))

        req = SimpleNamespace(width=896, height=512, num_frames=17)
        params.prepare_synthetic_warmup_request_for_queue(
            req, self._warmup_args(warmup_num_frames=17)
        )
        self.assertEqual((req.width, req.height, req.num_frames), (1920, 1088, 17))

    def test_probe_is_not_lightened(self):
        """A halved probe fails MAGI-2's fixed clip-length and resolution checks."""
        server_args = self._warmup_args()
        server_args.pipeline_class_name = None
        req = Req(sampling_params=Magi2SamplingParams(prompt="x"))
        self.assertIsNone(lighten_warmup_req(server_args, req))


if __name__ == "__main__":
    unittest.main()
