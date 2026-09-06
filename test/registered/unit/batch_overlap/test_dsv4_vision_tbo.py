"""Image prefill must veto TBO before multimodal metadata is discarded."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import sglang.srt.batch_overlap.two_batch_overlap as tbo
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")


class TestVisionTboVote(CustomTestCase):
    def vote(self, image):
        mode = Mock()
        mode.resolve.return_value.is_low_latency.return_value = False
        item = SimpleNamespace(
            is_image=lambda: True,
            offsets=[(10, 389)],
            model_specific_data={"types": [], "perm": []},
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            spec_info=None,
            extend_num_tokens=1024,
            extend_lens=[512, 512],
            is_extend_in_batch=True,
            multimodal_inputs=[SimpleNamespace(mm_items=[item])] if image else None,
        )
        preparer = tbo.TboDPAttentionPreparer()
        with (
            patch.object(tbo, "is_tbo_enabled", return_value=True),
            patch.object(tbo, "get_deepep_mode", return_value=mode),
            patch.object(
                tbo,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch.object(tbo, "compute_split_seq_index", return_value=1),
        ):
            vote, forward_mode = preparer.prepare_all_gather(batch)
        return preparer, vote, forward_mode

    def test_image_rank_vetoes_tbo_for_all_dp_ranks(self):
        _, image_vote, mode = self.vote(True)
        text_preparer, text_vote, _ = self.vote(False)
        self.assertFalse(image_vote)
        self.assertTrue(text_vote)
        split, global_mode = text_preparer.compute_output(
            torch.tensor([[int(image_vote), mode], [int(text_vote), mode]])
        )
        self.assertIsNone(split)
        self.assertIsNone(global_mode)


if __name__ == "__main__":
    unittest.main()
