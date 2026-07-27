import unittest
from types import SimpleNamespace

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    TboDPAttentionPreparer,
    TboForwardBatchPreparer,
)
from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
from sglang.srt.layers.attention.tbo_backend import (
    _build_tbo_prefill_child_replay_fb_view,
    _split_prefill_replay_layout,
)
from sglang.srt.managers.overlap_utils import decide_needs_cpu_seq_lens
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepSeekV4PrefillTBO(CustomTestCase):
    def _assert_layout(
        self,
        extend_lens: list[int],
        prefix_lens: list[int],
        boundary: int,
        expected_left: list[int],
        expected_right: list[int],
    ) -> None:
        left, right = _split_prefill_replay_layout(
            extend_seq_lens=extend_lens,
            extend_prefix_lens=prefix_lens,
            split_token_index=boundary,
        )
        self.assertEqual(left.extend_seq_lens, expected_left)
        self.assertEqual(right.extend_seq_lens, expected_right)
        self.assertEqual(left.extend_num_tokens, sum(expected_left))
        self.assertEqual(right.extend_num_tokens, sum(expected_right))
        self.assertEqual(
            left.extend_num_tokens + right.extend_num_tokens,
            sum(extend_lens),
        )

        for index, (extend_len, prefix_len) in enumerate(
            zip(extend_lens, prefix_lens, strict=True)
        ):
            left_len = expected_left[index]
            right_len = expected_right[index]
            if left_len:
                self.assertEqual(left.extend_prefix_lens[index], prefix_len)
                self.assertEqual(left.seq_lens[index], prefix_len + left_len)
            else:
                self.assertEqual(left.extend_prefix_lens[index], 0)
                self.assertEqual(left.seq_lens[index], 0)
            if right_len:
                self.assertEqual(
                    right.extend_prefix_lens[index],
                    prefix_len + left_len,
                )
                self.assertEqual(
                    right.seq_lens[index],
                    prefix_len + extend_len,
                )
            else:
                self.assertEqual(right.extend_prefix_lens[index], 0)
                self.assertEqual(right.seq_lens[index], 0)

    def test_fixed_boundary_layouts(self) -> None:
        self._assert_layout([100], [7], 50, [50], [50])
        self._assert_layout(
            [10, 20, 30, 0],
            [1, 2, 3, 0],
            25,
            [10, 15, 0, 0],
            [0, 5, 30, 0],
        )
        self._assert_layout([8, 8, 8], [2, 4, 6], 16, [8, 8, 0], [0, 0, 8])
        self._assert_layout([7, 3, 2], [0, 10, 20], 32, [7, 3, 2], [0, 0, 0])

    def test_child_replay_view_preserves_request_geometry(self) -> None:
        parent = SimpleNamespace(
            batch_size=4,
            forward_mode=ForwardMode.EXTEND,
            actual_forward_mode=ForwardMode.EXTEND,
            input_ids=torch.arange(64, dtype=torch.int64),
            positions=torch.arange(64, dtype=torch.int64),
            out_cache_loc=torch.arange(64, dtype=torch.int64),
            req_pool_indices=torch.tensor([11, 12, 13, 0], dtype=torch.int64),
            seq_lens=torch.tensor([11, 22, 33, 0], dtype=torch.int64),
            extend_seq_lens=torch.tensor([10, 20, 30, 0], dtype=torch.int64),
            extend_prefix_lens=torch.tensor([1, 2, 3, 0], dtype=torch.int64),
            extend_start_loc=torch.tensor([0, 10, 30, 60], dtype=torch.int64),
        )
        left, right = _split_prefill_replay_layout(
            extend_seq_lens=[10, 20, 30, 0],
            extend_prefix_lens=[1, 2, 3, 0],
            split_token_index=25,
        )
        left_view = _build_tbo_prefill_child_replay_fb_view(
            parent,
            layout=left,
            tok_slice=slice(None, 32),
        )
        right_view = _build_tbo_prefill_child_replay_fb_view(
            parent,
            layout=right,
            tok_slice=slice(32, None),
        )
        self.assertEqual(left_view.req_pool_indices.tolist(), [11, 12, 0, 0])
        self.assertEqual(right_view.req_pool_indices.tolist(), [0, 12, 13, 0])
        self.assertEqual(left_view.extend_start_loc.tolist(), [0, 10, 25, 25])
        self.assertEqual(right_view.extend_start_loc.tolist(), [0, 0, 5, 35])
        self.assertEqual(left_view.extend_num_tokens, 25)
        self.assertEqual(right_view.extend_num_tokens, 35)

    def test_capture_child_padding_uses_zero_sentinels(self) -> None:
        child = SimpleNamespace(
            batch_size=1,
            input_ids=torch.arange(32, dtype=torch.int64),
            req_pool_indices=torch.tensor([9], dtype=torch.int64),
            seq_lens=torch.tensor([37], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([37], dtype=torch.int64),
            orig_seq_lens=torch.tensor([37], dtype=torch.int64),
            extend_seq_lens=torch.tensor([32], dtype=torch.int64),
            extend_prefix_lens=torch.tensor([5], dtype=torch.int64),
            extend_start_loc=torch.tensor([0], dtype=torch.int64),
            extend_seq_lens_cpu=[32],
            extend_prefix_lens_cpu=[5],
            extend_logprob_start_lens_cpu=[32],
        )
        TboForwardBatchPreparer.pad_sequence_axis_for_cuda_graph(
            child,
            target_batch_size=4,
        )
        self.assertEqual(child.batch_size, 4)
        self.assertEqual(child.req_pool_indices.tolist(), [9, 0, 0, 0])
        self.assertEqual(child.seq_lens.tolist(), [37, 0, 0, 0])
        self.assertEqual(child.extend_seq_lens.tolist(), [32, 0, 0, 0])
        self.assertEqual(child.extend_prefix_lens.tolist(), [5, 0, 0, 0])
        self.assertEqual(child.extend_start_loc.tolist(), [0, 32, 32, 32])

    def test_phase_and_dp_sync_gates(self) -> None:
        self.assertTrue(
            DeepseekV4AttnBackend.tbo_supports_cuda_graph_for(ForwardMode.EXTEND)
        )
        self.assertFalse(
            DeepseekV4AttnBackend.tbo_supports_cuda_graph_for(ForwardMode.DECODE)
        )
        self.assertFalse(
            DeepseekV4AttnBackend.tbo_supports_cuda_graph_for(ForwardMode.TARGET_VERIFY)
        )
        self.assertFalse(DeepseekV4AttnBackend.tbo_supports_decode_cuda_graph)
        self.assertFalse(DeepseekV4AttnBackend.tbo_requires_global_cpu_seq_lens)

        server_args = SimpleNamespace(
            enable_two_batch_overlap=True,
            speculative_algorithm="DSPARK",
        )
        self.assertFalse(
            decide_needs_cpu_seq_lens(
                server_args,
                [
                    SimpleNamespace(
                        tbo_requires_global_cpu_seq_lens=False,
                        needs_cpu_seq_lens=False,
                    )
                ],
            )
        )

        preparer = TboDPAttentionPreparer()
        preparer.enable_two_batch_overlap = True
        preparer.local_tbo_split_seq_index = 7
        split, mode = preparer.compute_output_from_values(
            local_can_run_tbo=[1, 1],
            forward_modes=[ForwardMode.EXTEND.value, ForwardMode.IDLE.value],
        )
        self.assertEqual(split, 7)
        self.assertEqual(mode, ForwardMode.EXTEND)
        split, mode = preparer.compute_output_from_values(
            local_can_run_tbo=[1, 0],
            forward_modes=[ForwardMode.EXTEND.value, ForwardMode.EXTEND.value],
        )
        self.assertIsNone(split)
        self.assertIsNone(mode)


if __name__ == "__main__":
    unittest.main()
