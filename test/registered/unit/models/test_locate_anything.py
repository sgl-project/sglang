"""Unit tests for srt/models/locate_anything.py — no server, no weight loading.

Covers the InternVL-style ``mlp1`` projector shape and the optional box-grammar
logit processor's constrained-decoding state machine.
"""

import unittest

import numpy as np
import torch

from sglang.srt.configs import LocateAnythingConfig
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.locate_anything import (
    LocateAnythingBoxGrammarLogitProcessor,
    LocateAnythingForConditionalGeneration,
    LocateAnythingMultiModalProjector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _small_config():
    # Tiny dims keep the test fast and CPU-only.
    return LocateAnythingConfig(
        vision_config={"hidden_size": 8, "merge_kernel_size": [2, 2]},
        text_config={"hidden_size": 16},
    )


class TestLocateAnythingProjector(CustomTestCase):
    def test_merged_size_and_output_shape(self):
        cfg = _small_config()
        proj = LocateAnythingMultiModalProjector(cfg)
        # merged_size = hidden_size * merge_h * merge_w = 8 * 2 * 2 = 32
        self.assertEqual(proj.merged_size, 32)
        self.assertEqual(proj.pre_norm.normalized_shape, (32,))
        self.assertEqual(proj.linear_1.in_features, 32)
        self.assertEqual(proj.linear_1.out_features, 16)
        self.assertEqual(proj.linear_2.in_features, 16)
        self.assertEqual(proj.linear_2.out_features, 16)

    def test_forward_flattens_merged_patches(self):
        cfg = _small_config()
        proj = LocateAnythingMultiModalProjector(cfg).eval()
        # MoonViT patch_merger yields (num_merged_tokens, merge_h*merge_w, hidden).
        num_tokens = 5
        feats = torch.randn(num_tokens, 4, 8)
        with torch.no_grad():
            out = proj(feats)
        # One projected vector of text_hidden width per merged token.
        self.assertEqual(out.shape, (num_tokens, 16))

    def test_forward_handles_noncontiguous_input(self):
        cfg = _small_config()
        proj = LocateAnythingMultiModalProjector(cfg).eval()
        # A transposed/sliced tensor is non-contiguous; reshape (not view) must cope.
        feats = torch.randn(4, 5, 8).transpose(0, 1)  # (5, 4, 8), non-contiguous
        self.assertFalse(feats.is_contiguous())
        with torch.no_grad():
            out = proj(feats)
        self.assertEqual(out.shape, (5, 16))


class _FakeReq:
    def __init__(self, output_ids):
        self.origin_input_ids = [1, 2, 3]
        self.output_ids = output_ids


class TestBoxGrammarLogitProcessor(CustomTestCase):
    # Token-id layout mirroring nvidia/LocateAnything-3B.
    BOX_START = 151668
    BOX_END = 151669
    COORD_START = 151677
    COORD_END = 152677
    NONE = 4064
    VOCAB = 152681

    def _params(self, output_ids):
        return [
            {
                "__req__": _FakeReq(output_ids),
                "box_start_token_id": self.BOX_START,
                "box_end_token_id": self.BOX_END,
                "coord_start_token_id": self.COORD_START,
                "coord_end_token_id": self.COORD_END,
                "none_token_id": self.NONE,
            }
        ]

    def _allowed_ids(self, output_ids):
        proc = LocateAnythingBoxGrammarLogitProcessor()
        logits = torch.zeros(1, self.VOCAB)
        out = proc(logits, self._params(output_ids))
        # Allowed ids are those left finite after masking.
        return set(torch.nonzero(torch.isfinite(out[0])).flatten().tolist())

    def test_no_box_open_is_untouched(self):
        proc = LocateAnythingBoxGrammarLogitProcessor()
        logits = torch.randn(1, self.VOCAB)
        original = logits.clone()
        out = proc(logits, self._params([42, 43]))  # no box_start
        self.assertTrue(torch.equal(out, original))

    def test_just_after_box_start_allows_coords_or_none(self):
        allowed = self._allowed_ids([self.BOX_START])
        self.assertIn(self.NONE, allowed)
        self.assertIn(self.COORD_START, allowed)
        self.assertIn(self.COORD_END, allowed)
        self.assertNotIn(self.BOX_END, allowed)

    def test_after_none_must_close(self):
        allowed = self._allowed_ids([self.BOX_START, self.NONE])
        self.assertEqual(allowed, {self.BOX_END})

    def test_one_coord_forces_more_coords(self):
        allowed = self._allowed_ids([self.BOX_START, self.COORD_START])
        self.assertNotIn(self.BOX_END, allowed)
        self.assertNotIn(self.NONE, allowed)
        self.assertIn(self.COORD_START, allowed)

    def test_two_coords_may_close_point_or_continue(self):
        allowed = self._allowed_ids(
            [self.BOX_START, self.COORD_START, self.COORD_START]
        )
        self.assertIn(self.BOX_END, allowed)  # 2-coord point can close
        self.assertIn(self.COORD_START, allowed)  # or continue toward a bbox

    def test_three_coords_forces_fourth(self):
        allowed = self._allowed_ids([self.BOX_START] + [self.COORD_START] * 3)
        self.assertNotIn(self.BOX_END, allowed)
        self.assertIn(self.COORD_START, allowed)

    def test_four_coords_must_close(self):
        allowed = self._allowed_ids([self.BOX_START] + [self.COORD_START] * 4)
        self.assertEqual(allowed, {self.BOX_END})

    def test_more_than_four_coords_must_close(self):
        # The ">= 4 coords -> must close" branch must also fire if the model
        # somehow emitted a 5th coordinate.
        allowed = self._allowed_ids([self.BOX_START] + [self.COORD_START] * 5)
        self.assertEqual(allowed, {self.BOX_END})

    def test_coord_end_counts_as_a_coordinate(self):
        # The coord range check is inclusive of coord_end (coord_start <= t <=
        # coord_end); a body holding only coord_end must be treated as 1 coord.
        allowed = self._allowed_ids([self.BOX_START, self.COORD_END])
        self.assertNotIn(self.BOX_END, allowed)  # 1 coord -> need more
        self.assertNotIn(self.NONE, allowed)
        self.assertIn(self.COORD_START, allowed)

    def test_missing_token_id_is_noop(self):
        # If a client passes custom_params missing one of the five ids, the
        # processor must skip that request rather than crash or partially mask.
        proc = LocateAnythingBoxGrammarLogitProcessor()
        logits = torch.randn(1, self.VOCAB)
        original = logits.clone()
        params = self._params([self.BOX_START])
        del params[0]["none_token_id"]
        out = proc(logits, params)
        self.assertTrue(torch.equal(out, original))

    def test_closed_box_is_untouched(self):
        proc = LocateAnythingBoxGrammarLogitProcessor()
        logits = torch.randn(1, self.VOCAB)
        original = logits.clone()
        # A fully-formed bbox that is already closed.
        out = proc(
            logits,
            self._params([self.BOX_START] + [self.COORD_START] * 4 + [self.BOX_END]),
        )
        self.assertTrue(torch.equal(out, original))

    def test_empty_param_list_is_noop(self):
        proc = LocateAnythingBoxGrammarLogitProcessor()
        logits = torch.randn(1, self.VOCAB)
        original = logits.clone()
        self.assertTrue(torch.equal(proc(logits, None), original))

    def test_build_sampling_params_wires_config_token_ids(self):
        config = _small_config()
        params = LocateAnythingBoxGrammarLogitProcessor.build_sampling_params(config)
        # Serialized processor + the 5 token ids the processor reads per request.
        self.assertIn("custom_logit_processor", params)
        self.assertEqual(
            params["custom_logit_processor"],
            LocateAnythingBoxGrammarLogitProcessor.to_str(),
        )
        self.assertEqual(
            params["custom_params"],
            {
                "box_start_token_id": config.box_start_token_id,
                "box_end_token_id": config.box_end_token_id,
                "coord_start_token_id": config.coord_start_token_id,
                "coord_end_token_id": config.coord_end_token_id,
                "none_token_id": config.none_token_id,
            },
        )


class _StubVisionTower:
    """Stand-in for MoonViT in get_image_feature.

    The real vision tower has its own tests (kimi_vl_moonvit); here we only need
    it to (a) expose ``dtype``/``device`` and (b) return one ``(N, merge, hidden)``
    feature block per image so the projector + concat wiring is exercised with
    real shapes. ``patches_per_image`` mirrors ``prod(image_grid_hws)``.

    To keep the oracle honest, ``__call__`` asserts that get_image_feature fed
    it the inputs we expect — a ``(sum(patches), hidden)`` pixel tensor and a
    rank-2 ``(num_images, 2)`` ``image_grid_hws`` whose per-row product matches
    ``patches_per_image`` — so a regression in how the feature/grid are wired or
    coerced fails here rather than passing on a fabricated shape.
    """

    def __init__(self, hidden, merge, patches_per_image):
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self._hidden = hidden
        self._merge = merge
        self._patches = patches_per_image

    def __call__(self, pixel_values, image_grid_hws):
        # The concatenated raw patches across all images must line up.
        assert pixel_values.shape == (
            sum(self._patches),
            self._hidden,
        ), pixel_values.shape
        # image_grid_hws must be coerced to a rank-2 (num_images, 2) tensor whose
        # rows multiply to the expected patch counts.
        assert isinstance(image_grid_hws, torch.Tensor)
        assert image_grid_hws.shape == (len(self._patches), 2), image_grid_hws.shape
        assert image_grid_hws.prod(dim=-1).tolist() == list(self._patches)
        # MoonViT yields a list of (num_merged_tokens, merge, hidden) per image.
        return [
            torch.zeros(p // self._merge, self._merge, self._hidden)
            for p in self._patches
        ]


def _bare_model(config):
    """A LocateAnythingForConditionalGeneration with a real projector but a
    stubbed vision tower, bypassing the distributed Qwen2 __init__."""
    model = LocateAnythingForConditionalGeneration.__new__(
        LocateAnythingForConditionalGeneration
    )
    model.config = config
    model.multi_modal_projector = LocateAnythingMultiModalProjector(config).eval()
    return model


def _image_item(feature, grid_hws):
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, 1)],
        feature=feature,
        model_specific_data={"image_grid_hws": grid_hws},
    )


class TestGetImageFeatureWiring(CustomTestCase):
    """Forward-shape smoke test for get_image_feature.

    Guards the production path (pixel concat -> vision tower -> projector) and
    the precomputed-embedding passthrough so a future change to the wiring or
    the numpy->tensor image_grid_hws coercion doesn't silently regress. The
    heavy MoonViT forward is stubbed (covered by its own tests); the projector
    is real.
    """

    HIDDEN = 8  # vision hidden_size, must match _small_config()
    MERGE = 4  # merge_h * merge_w = 2 * 2
    TEXT_HIDDEN = 16  # text_config hidden_size

    def test_single_image_projects_to_text_hidden(self):
        cfg = _small_config()
        model = _bare_model(cfg)
        # grid [[2, 2]] -> prod = 4 patches.
        model.vision_tower = _StubVisionTower(self.HIDDEN, self.MERGE, [4])
        feature = torch.randn(4, self.HIDDEN)  # one image's raw patches
        out = model.get_image_feature([_image_item(feature, [[2, 2]])])
        # 4 patches / merge(4) = 1 merged token, projected to text hidden width.
        self.assertEqual(out.shape, (1, self.TEXT_HIDDEN))

    def test_multi_image_features_concatenated_in_order(self):
        cfg = _small_config()
        model = _bare_model(cfg)
        # Two images: [[2, 2]] -> 4 patches, [[4, 2]] -> 8 patches.
        model.vision_tower = _StubVisionTower(self.HIDDEN, self.MERGE, [4, 8])
        items = [
            _image_item(torch.randn(4, self.HIDDEN), [[2, 2]]),
            _image_item(torch.randn(8, self.HIDDEN), [[4, 2]]),
        ]
        out = model.get_image_feature(items)
        # Merged tokens: 4/4 + 8/4 = 1 + 2 = 3, each projected to text hidden.
        self.assertEqual(out.shape, (3, self.TEXT_HIDDEN))

    def test_image_grid_hws_numpy_is_coerced(self):
        # The HF image processor hands image_grid_hws back as a numpy array;
        # get_image_feature must torch.as_tensor it before torch.cat (else the
        # cat raises). A numpy grid must produce the same shape as a list grid.
        cfg = _small_config()
        model = _bare_model(cfg)
        model.vision_tower = _StubVisionTower(self.HIDDEN, self.MERGE, [4])
        feature = torch.randn(4, self.HIDDEN)
        grid = np.array([[2, 2]], dtype=np.int64)
        out = model.get_image_feature([_image_item(feature, grid)])
        self.assertEqual(out.shape, (1, self.TEXT_HIDDEN))

    def test_precomputed_embeddings_pass_through(self):
        # Already-projected embeddings (dim==2, last dim == text hidden) must be
        # returned untouched without invoking the vision tower forward. (dtype/
        # device are still read for the cast, so the stub exposes them but raises
        # if its forward is actually called.)
        cfg = _small_config()
        model = _bare_model(cfg)

        class _NoCallTower:
            dtype = torch.float32
            device = torch.device("cpu")

            def __call__(self, *args, **kwargs):
                raise AssertionError(
                    "vision_tower forward should not run on precomputed embeds"
                )

        model.vision_tower = _NoCallTower()
        embeds = torch.randn(5, self.TEXT_HIDDEN)
        out = model.get_image_feature([_image_item(embeds, [[2, 2]])])
        self.assertTrue(torch.equal(out, embeds))


if __name__ == "__main__":
    unittest.main()
