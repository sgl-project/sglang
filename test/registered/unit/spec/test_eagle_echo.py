import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.speculative.eagle_echo import (
    apply_eagle_retrieval_layout,
    build_eagle_ragged_verify_window,
    scatter_eagle_verify_output,
)
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.speculative.eagle_utils import (
    compute_echo_verify_lens,
    eagle_ragged_graph_tier_eligible,
)
from sglang.srt.speculative.eagle_worker_common import (
    _select_eagle_ragged_graph_num_tokens,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.srt.speculative.standalone_worker_v2 import StandaloneDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_cpu_ci(est_time=20, suite="base-a-test-cpu")


class TestEagleEcho(CustomTestCase):
    def test_standalone_worker_defaults_echo_to_none(self):
        server_args = SimpleNamespace(
            device="cpu",
            speculative_eagle_topk=2,
            speculative_num_steps=3,
            speculative_num_draft_tokens=6,
            speculative_algorithm="STANDALONE",
            enable_dp_attention=False,
        )
        draft_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                model_config=SimpleNamespace(hf_config=SimpleNamespace())
            )
        )

        with (
            patch(
                "sglang.srt.speculative.standalone_worker_v2.replace",
                side_effect=lambda ps, **_: ps,
            ),
            patch(
                "sglang.srt.speculative.standalone_worker_v2.TpModelWorker",
                return_value=draft_worker,
            ),
            patch(
                "sglang.srt.speculative.standalone_worker_v2.default_tree_mask_mode",
                return_value=None,
            ),
            patch(
                "sglang.srt.speculative.standalone_worker_v2.get_plan_stream",
                return_value=(None, None),
            ),
        ):
            worker = StandaloneDraftWorker(
                server_args=server_args,
                gpu_id=0,
                ps=object(),
                nccl_port=0,
                target_worker=object(),
            )

        self.assertIsNone(worker.echo_threshold)

    def test_computes_request_verify_lens(self):
        score_list = [
            torch.tensor([[[0.9, 0.7]], [[0.9, 0.8]]]),
            torch.tensor(
                [
                    [[0.4, 0.2], [0.3, 0.1]],
                    [[0.8, 0.7], [0.6, 0.5]],
                ]
            ),
            torch.tensor(
                [
                    [[0.8, 0.7], [0.6, 0.5]],
                    [[0.7, 0.6], [0.5, 0.4]],
                ]
            ),
        ]
        top_scores_index = torch.tensor(
            [[0, 1, 2, 6, 9], [0, 2, 4, 7, 9]], dtype=torch.long
        )

        verify_lens = compute_echo_verify_lens(
            score_list, top_scores_index, threshold=0.5
        )

        self.assertEqual(verify_lens.dtype, torch.int32)
        self.assertEqual(verify_lens.tolist(), [3, 6])

    def test_threshold_boundary_and_nan(self):
        score_list = [
            torch.tensor([[[0.5, 0.1]], [[float("nan"), 0.1]]]),
            torch.tensor(
                [
                    [[float("nan"), 0.1], [0.2, 0.1]],
                    [[0.7, 0.6], [0.5, 0.4]],
                ]
            ),
        ]
        top_scores_index = torch.tensor([[0, 1, 2], [0, 1, 3]])

        verify_lens = compute_echo_verify_lens(
            score_list, top_scores_index, threshold=0.5
        )

        self.assertEqual(verify_lens.tolist(), [4, 4])

    def test_minimum_verify_length(self):
        score_list = [
            torch.tensor([[[0.49, 0.2]]]),
            torch.tensor([[[0.9, 0.8], [0.7, 0.6]]]),
        ]
        top_scores_index = torch.tensor([[0, 1, 3]])

        verify_lens = compute_echo_verify_lens(
            score_list, top_scores_index, threshold=0.5
        )

        self.assertEqual(verify_lens.tolist(), [1])

    def test_applies_retrieval_bounds(self):
        retrieve_index = torch.tensor([[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        retrieve_next_token = torch.tensor([[1, 2, 3, 4, 5, -1], [1, 2, 3, 4, 5, -1]])
        retrieve_next_sibling = torch.tensor(
            [[2, 3, 4, 5, -1, -1], [2, 3, 4, 5, -1, -1]]
        )

        apply_eagle_retrieval_layout(
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            verify_lens=torch.tensor([3, 1]),
        )

        self.assertEqual(
            retrieve_index.tolist(), [[0, 1, 2, -1, -1, -1], [6, -1, -1, -1, -1, -1]]
        )
        self.assertEqual(
            retrieve_next_token.tolist(),
            [[1, 2, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]],
        )
        self.assertEqual(
            retrieve_next_sibling.tolist(),
            [[2, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]],
        )

    def test_compact_window_and_scatter_keep_fixed_tree_layout(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([2, 3]), graph_num_tokens=6
        )
        draft_tokens = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])
        positions = torch.arange(100, 108)
        out_cache_loc = torch.arange(200, 208)

        window = build_eagle_ragged_verify_window(
            draft_tokens=draft_tokens,
            positions=positions,
            out_cache_loc=out_cache_loc,
            layout=layout,
            draft_token_num=4,
            padded_bs=2,
        )

        # The final row is graph padding and is ignored below.
        self.assertEqual(window.input_ids.tolist(), [10, 11, 12, 20, 21, 22])
        compact = torch.arange(12, dtype=torch.float32).view(6, 2)
        strided = scatter_eagle_verify_output(
            compact=compact,
            layout=layout,
            query_layout=window.query_layout,
            draft_token_num=4,
        ).view(2, 4, 2)
        self.assertTrue(torch.equal(strided[0, :2], compact[:2]))
        self.assertTrue(torch.equal(strided[1, :3], compact[3:6]))
        self.assertTrue(torch.equal(strided[0, 2:], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(strided[1, 3:], torch.zeros(1, 2)))

    def test_exact_eager_window_does_not_add_padding(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([2, 3]), graph_num_tokens=5
        )
        dense = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23])

        window = build_eagle_ragged_verify_window(
            draft_tokens=dense,
            positions=dense + 100,
            out_cache_loc=dense + 200,
            layout=layout,
            draft_token_num=4,
            padded_bs=2,
        )

        self.assertEqual(window.query_layout.verify_lens.tolist(), [2, 3])
        self.assertEqual(window.input_ids.tolist(), [10, 11, 20, 21, 22])

    def test_graph_tier_selection_checks_token_and_slot_capacity(self):
        class FakeGraphRunner:
            ragged_verify_mode = True
            capture_num_tokens = [6, 12, 18, 24]

            def __init__(self, max_slots):
                self.max_slots = max_slots

            def _ragged_capture_slots(self, num_tokens):
                return min(num_tokens, self.max_slots)

        runner = FakeGraphRunner(max_slots=4)
        self.assertTrue(
            eagle_ragged_graph_tier_eligible(
                graph_num_tokens=6,
                batch_size=4,
                graph_runner=runner,
            )
        )
        self.assertFalse(
            eagle_ragged_graph_tier_eligible(
                graph_num_tokens=6,
                batch_size=6,
                graph_runner=runner,
            )
        )
        self.assertEqual(
            _select_eagle_ragged_graph_num_tokens(
                total_verify_tokens=13,
                batch_size=4,
                graph_runner=runner,
            ),
            18,
        )
        self.assertEqual(
            _select_eagle_ragged_graph_num_tokens(
                total_verify_tokens=5,
                batch_size=5,
                graph_runner=runner,
            ),
            5,
        )
        self.assertEqual(
            _select_eagle_ragged_graph_num_tokens(
                total_verify_tokens=25,
                batch_size=4,
                graph_runner=runner,
            ),
            25,
        )
        self.assertEqual(
            _select_eagle_ragged_graph_num_tokens(
                total_verify_tokens=13,
                batch_size=4,
                graph_runner=None,
            ),
            13,
        )

    def test_bucket_padding_respects_fixed_tree_width(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([1, 4, 4, 4]), graph_num_tokens=18
        )
        dense = torch.arange(24)
        window = build_eagle_ragged_verify_window(
            draft_tokens=dense,
            positions=dense + 100,
            out_cache_loc=dense + 200,
            layout=layout,
            draft_token_num=6,
            padded_bs=4,
        )

        padded = window.query_layout

        self.assertEqual(padded.verify_lens.tolist(), [3, 5, 5, 5])
        self.assertEqual(int(padded.verify_lens.sum()), 18)
        self.assertLessEqual(int(padded.verify_lens.max()), 6)

        compact = torch.arange(18, dtype=torch.float32).view(18, 1)
        strided = scatter_eagle_verify_output(
            compact=compact,
            layout=layout,
            query_layout=padded,
            draft_token_num=6,
        ).view(4, 6)
        self.assertEqual(strided[0, :1].tolist(), [0.0])
        self.assertEqual(strided[1, :4].tolist(), [3.0, 4.0, 5.0, 6.0])
        self.assertEqual(strided[2, :4].tolist(), [8.0, 9.0, 10.0, 11.0])
        self.assertEqual(strided[3, :4].tolist(), [13.0, 14.0, 15.0, 16.0])
        self.assertTrue(torch.equal(strided[:, 4:], torch.zeros(4, 2)))

    def test_graph_padding_rows_for_extra_request_slots_are_zeroed(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([2, 1]), graph_num_tokens=8
        )
        dense = torch.arange(8)

        window = build_eagle_ragged_verify_window(
            draft_tokens=dense,
            positions=dense + 100,
            out_cache_loc=dense + 200,
            layout=layout,
            draft_token_num=4,
            padded_bs=3,
        )

        self.assertEqual(window.query_layout.verify_lens.tolist(), [4, 3, 1])
        self.assertEqual(window.input_ids.tolist(), [0, 1, 2, 3, 4, 5, 6, 0])
        self.assertEqual(
            window.positions.tolist(), [100, 101, 102, 103, 104, 105, 106, 0]
        )
        self.assertEqual(
            window.out_cache_loc.tolist(),
            [200, 201, 202, 203, 204, 205, 206, 0],
        )

    def test_exact_token_bucket_allows_zero_length_dummy_slots(self):
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([2, 2]), graph_num_tokens=4
        )
        dense = torch.arange(8)

        window = build_eagle_ragged_verify_window(
            draft_tokens=dense,
            positions=dense + 100,
            out_cache_loc=dense + 200,
            layout=layout,
            draft_token_num=4,
            padded_bs=4,
        )

        self.assertEqual(window.query_layout.verify_lens.tolist(), [2, 2, 0, 0])
        self.assertEqual(window.query_layout.qo_indptr_device.tolist(), [0, 2, 4, 4, 4])
        self.assertEqual(window.input_ids.tolist(), [0, 1, 4, 5])

    def test_rejects_invalid_host_query_layout(self):
        dense = torch.arange(8)
        cases = [
            RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=torch.tensor([2, 5]), graph_num_tokens=7
            ),
            RaggedVerifyLayout.from_verify_lens_device(
                verify_lens=torch.tensor([2, 2]), graph_num_tokens=3
            ),
        ]

        for layout in cases:
            with self.subTest(verify_lens=layout.verify_lens.tolist()):
                with self.assertRaises(ValueError):
                    build_eagle_ragged_verify_window(
                        draft_tokens=dense,
                        positions=dense + 100,
                        out_cache_loc=dense + 200,
                        layout=layout,
                        draft_token_num=4,
                        padded_bs=2,
                    )

    def test_dense_topk_graph_padding_uses_only_real_tree_masks(self):
        backend = object.__new__(FlashAttentionBackend)
        backend.speculative_num_draft_tokens = 3
        backend.req_to_token = torch.arange(32).view(2, 16)
        spec_info = SimpleNamespace(
            custom_mask=(torch.arange(24) % 2 == 0),
            ragged_verify_layout=None,
        )

        mask, page_table = backend._build_target_verify_topk_expand_inputs(
            seq_lens=torch.tensor([5, 7]),
            req_pool_indices=torch.tensor([0, 1]),
            spec_info=spec_info,
            query_layout=None,
            real_bs=1,
        )

        self.assertEqual(mask.shape, (6, 3))
        self.assertEqual(page_table.shape, (6, 3))
        expected_padding_mask = torch.tensor([[True, False, False]]).expand(3, -1)
        self.assertTrue(torch.equal(mask[3:], expected_padding_mask))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_padding_compaction_and_scatter_match_expected_layout(self):
        device = torch.device("cuda")
        layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([1, 4, 4, 4], device=device),
            graph_num_tokens=18,
        )
        dense = torch.arange(24, device=device)

        window = build_eagle_ragged_verify_window(
            draft_tokens=dense,
            positions=dense + 100,
            out_cache_loc=dense + 200,
            layout=layout,
            draft_token_num=6,
            padded_bs=4,
        )

        self.assertEqual(window.query_layout.verify_lens.cpu().tolist(), [3, 5, 5, 5])
        self.assertEqual(
            window.input_ids.cpu().tolist(),
            [0, 1, 2, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
        )

        compact = torch.arange(1, 19, dtype=torch.float32, device=device).view(18, 1)
        strided = scatter_eagle_verify_output(
            compact=compact,
            layout=layout,
            query_layout=window.query_layout,
            draft_token_num=6,
        ).view(4, 6)
        expected = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0],
                [4, 5, 6, 7, 0, 0],
                [9, 10, 11, 12, 0, 0],
                [14, 15, 16, 17, 0, 0],
            ],
            dtype=torch.float32,
            device=device,
        )
        self.assertTrue(torch.equal(strided, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_extra_graph_slots_match_torch_reference(self):
        cpu_layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=torch.tensor([2, 1]), graph_num_tokens=8
        )
        cuda_layout = RaggedVerifyLayout.from_verify_lens_device(
            verify_lens=cpu_layout.verify_lens.cuda(),
            graph_num_tokens=8,
        )
        dense = torch.arange(8)

        cpu_window = build_eagle_ragged_verify_window(
            draft_tokens=dense,
            positions=dense + 100,
            out_cache_loc=dense + 200,
            layout=cpu_layout,
            draft_token_num=4,
            padded_bs=3,
        )
        cuda_window = build_eagle_ragged_verify_window(
            draft_tokens=dense.cuda(),
            positions=(dense + 100).cuda(),
            out_cache_loc=(dense + 200).cuda(),
            layout=cuda_layout,
            draft_token_num=4,
            padded_bs=3,
        )

        self.assertTrue(
            torch.equal(
                cuda_window.query_layout.verify_lens.cpu(),
                cpu_window.query_layout.verify_lens,
            )
        )
        self.assertTrue(torch.equal(cuda_window.input_ids.cpu(), cpu_window.input_ids))
        self.assertTrue(torch.equal(cuda_window.positions.cpu(), cpu_window.positions))
        self.assertTrue(
            torch.equal(cuda_window.out_cache_loc.cpu(), cpu_window.out_cache_loc)
        )

        compact = torch.arange(24, dtype=torch.float32).view(8, 3)
        cpu_output = scatter_eagle_verify_output(
            compact=compact,
            layout=cpu_layout,
            query_layout=cpu_window.query_layout,
            draft_token_num=4,
        )
        cuda_output = scatter_eagle_verify_output(
            compact=compact.cuda(),
            layout=cuda_layout,
            query_layout=cuda_window.query_layout,
            draft_token_num=4,
        )
        self.assertTrue(torch.equal(cuda_output.cpu(), cpu_output))


if __name__ == "__main__":
    unittest.main()
