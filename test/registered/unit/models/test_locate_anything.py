"""Unit tests for srt/models/locate_anything.py — no server, no weight loading.

Covers the InternVL-style ``mlp1`` projector shape and the optional box-grammar
logit processor's constrained-decoding state machine.
"""

import unittest

import torch

from sglang.srt.configs import LocateAnythingConfig
from sglang.srt.models.locate_anything import (
    LocateAnythingBoxGrammarLogitProcessor,
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


if __name__ == "__main__":
    unittest.main()
