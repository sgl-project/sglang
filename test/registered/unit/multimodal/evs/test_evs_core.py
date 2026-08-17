"""Unit tests for srt/multimodal/evs/evs_core.py — no server, no model weights."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import torch

from sglang.srt.multimodal.evs import evs_core
from sglang.test.test_utils import CustomTestCase


class TestComputeRetainedTokensCount(CustomTestCase):
    def test_q_zero_retains_all_tokens(self):
        # total=8, (1-q)=1.0 → retain 8
        self.assertEqual(evs_core.compute_retained_tokens_count(4, 2, q=0.0), 8)

    def test_high_q_still_retains_first_frame(self):
        # total=8, (1-q)=0.1 → 0 (int) but min is tokens_per_frame=4
        self.assertEqual(evs_core.compute_retained_tokens_count(4, 2, q=0.9), 4)

    def test_q_mid_range_rounds_down(self):
        # total=10, q=0.25 → retain int(7.5)=7 but min is tokens_per_frame=5
        self.assertEqual(evs_core.compute_retained_tokens_count(5, 2, q=0.25), 7)

    def test_single_frame_never_below_tokens_per_frame(self):
        self.assertEqual(evs_core.compute_retained_tokens_count(16, 1, q=0.99), 16)


class TestComputeRetentionMask(CustomTestCase):
    def _make_video_embeds(self, T: int, H: int, W: int, hidden: int):
        # Create embeddings where frame 0 and 1 are identical → similarity=1, dissimilarity=0
        per_frame = H * W
        x0 = torch.randn((per_frame, hidden))
        x = torch.cat([x0, x0.clone()], dim=0)
        self.assertEqual(x.shape[0], T * per_frame)
        return x

    def test_q_high_retains_only_first_frame_tokens(self):
        T, H, W, hidden = 2, 2, 2, 8
        embeds = self._make_video_embeds(T, H, W, hidden)
        mask = evs_core.compute_retention_mask(
            video_embeds=embeds,
            video_size_thw=(T, H, W),
            spatial_merge_size=1,
            q=0.9,
        )
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.numel(), T * H * W)
        # With q=0.9, retained tokens count is min(tokens_per_frame=4, int(8*0.1)=0) => 4.
        self.assertEqual(mask.sum().item(), 4)
        self.assertTrue(mask[:4].all().item())
        self.assertFalse(mask[4:].any().item())

    def test_q_zero_retains_all_tokens(self):
        T, H, W, hidden = 2, 2, 2, 8
        embeds = self._make_video_embeds(T, H, W, hidden)
        mask = evs_core.compute_retention_mask(
            video_embeds=embeds,
            video_size_thw=(T, H, W),
            spatial_merge_size=1,
            q=0.0,
        )
        self.assertTrue(mask.all().item())

    def test_spatial_merge_size_reduces_tokens_per_frame(self):
        # H=W=4, sms=2 => (H/sms)*(W/sms)=4 tokens per frame; T=2 => 8 total
        T, H, W, hidden = 2, 4, 4, 8
        tokens_per_frame = (H // 2) * (W // 2)
        total_tokens = T * tokens_per_frame
        embeds = torch.randn((total_tokens, hidden))
        mask = evs_core.compute_retention_mask(
            video_embeds=embeds,
            video_size_thw=(T, H, W),
            spatial_merge_size=2,
            q=0.5,
        )
        self.assertEqual(mask.numel(), total_tokens)
        expected = evs_core.compute_retained_tokens_count(tokens_per_frame, T, q=0.5)
        self.assertEqual(mask.sum().item(), expected)

    def test_retention_mask_is_boolean_and_flat(self):
        T, H, W, hidden = 3, 2, 2, 4
        embeds = torch.randn((T * H * W, hidden))
        mask = evs_core.compute_retention_mask(
            video_embeds=embeds,
            video_size_thw=(T, H, W),
            spatial_merge_size=1,
            q=0.4,
        )
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.dim(), 1)


class TestEvsInputIdHelpers(CustomTestCase):
    def test_tokens_per_frame_sums_to_retained(self):
        out = evs_core.tokens_per_frame(q=0.25, num_frames=3, frame_num_tokens=4)
        retained = evs_core.compute_retained_tokens_count(
            tokens_per_frame=4, num_frames=3, q=0.25
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(sum(out), retained)
        # Contract: remainder goes to the final frame.
        self.assertGreaterEqual(out[-1], out[0])

    def test_replace_offsets_single_span(self):
        out = evs_core.replace_offsets_with_tokens_per_frame(
            pre_chunked_input_ids=[1, 0, 0, 4, 5],
            num_tokens_per_frame=[1, 2],
            frame_offsets_inclusive=[(1, 2)],
            filler_token_id=9,
        )
        self.assertEqual(out, [1, 9, 9, 9, 4, 5])

    def test_replace_offsets_multi_span(self):
        out = evs_core.replace_offsets_with_tokens_per_frame(
            pre_chunked_input_ids=[1, 0, 0, 4, 5, 0, 0, 9],
            num_tokens_per_frame=[1, 3],
            frame_offsets_inclusive=[(1, 2), (5, 6)],
            filler_token_id=7,
        )
        self.assertEqual(out, [1, 7, 4, 5, 7, 7, 7, 9])


if __name__ == "__main__":
    import unittest

    unittest.main()
