import unittest

import torch

from sglang.srt.layers.moe.shared_ep.layout import SharedEpLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


class TestSharedEpLayout(unittest.TestCase):
    def test_registered_model_shapes_have_disjoint_owner_slots(self):
        """Owner/route address math must never alias a peer contribution."""
        for hidden_size, top_k in ((6144, 8), (4096, 6)):
            with self.subTest(hidden_size=hidden_size, top_k=top_k):
                layout = SharedEpLayout.build(
                    hidden_size=hidden_size,
                    top_k=top_k,
                    max_tokens_per_rank=32,
                )
                self.assertEqual(layout.scale_groups, hidden_size // 128)
                self.assertEqual(layout.input_row_bytes, 64 * 1024)
                self.assertEqual(layout.output_row_bytes, 16 * 1024)

                intervals = []
                for token in range(layout.max_tokens_per_rank):
                    for route in range(layout.top_k):
                        start = layout.output_slot_offset(token, route)
                        end = start + layout.output_payload_bytes
                        self.assertLessEqual(end, layout.output_rank_bytes)
                        intervals.append((start, end))
                intervals.sort()
                for left, right in zip(intervals, intervals[1:]):
                    self.assertLessEqual(left[1], right[0])

    def test_global_input_views_preserve_vmm_rank_padding(self):
        """A mapped-rank gap must not shift peer activation or route fields."""
        layout = SharedEpLayout.build(
            hidden_size=4096,
            top_k=6,
            max_tokens_per_rank=32,
        )
        world_size = 2
        mapped_rank_bytes = layout.input_rank_bytes + 64 * 1024
        storage = torch.zeros(
            world_size * mapped_rank_bytes,
            dtype=torch.uint8,
        )

        views = layout.input_views(
            storage,
            world_size=world_size,
            mapped_rank_bytes=mapped_rank_bytes,
        )
        views.topk_ids[1, 2, 3] = 197
        views.topk_weights[1, 2, 3] = 0.25

        row_start = mapped_rank_bytes + 2 * layout.input_row_bytes
        id_start = row_start + layout.topk_id_offset + 3 * 4
        weight_start = row_start + layout.topk_weight_offset + 3 * 4
        self.assertEqual(storage[id_start : id_start + 4].view(torch.int32).item(), 197)
        self.assertEqual(
            storage[weight_start : weight_start + 4].view(torch.float32).item(),
            0.25,
        )
        self.assertEqual(
            views.activations.stride(0) * views.activations.element_size(),
            mapped_rank_bytes,
        )

    def test_global_output_view_preserves_vmm_rank_padding(self):
        """Direct W2 stores must address owner slots across a padded rank stride."""
        layout = SharedEpLayout.build(
            hidden_size=6144,
            top_k=8,
            max_tokens_per_rank=32,
        )
        world_size = 2
        mapped_rank_bytes = layout.output_rank_bytes + 64 * 1024
        storage = torch.zeros(
            world_size * mapped_rank_bytes,
            dtype=torch.uint8,
        )

        output = layout.output_view(
            storage,
            world_size=world_size,
            mapped_rank_bytes=mapped_rank_bytes,
        )
        output[1, 4, 5, 7] = 3.5

        byte_offset = (
            mapped_rank_bytes
            + layout.output_slot_offset(4, 5)
            + 7 * torch.empty((), dtype=torch.bfloat16).element_size()
        )
        self.assertEqual(
            storage[byte_offset : byte_offset + 2].view(torch.bfloat16).item(),
            3.5,
        )
        self.assertEqual(
            output.stride(0) * output.element_size(),
            mapped_rank_bytes,
        )


if __name__ == "__main__":
    unittest.main()
