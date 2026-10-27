"""Depth-buffer isolation and live metadata refresh for hybrid tree verification."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.ubtree_info import create_ubtree_verify_input
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _runner(is_draft_worker=False):
    return SimpleNamespace(
        server_args=SimpleNamespace(enable_mis=False),
        device="cpu",
        is_draft_worker=is_draft_worker,
        req_to_token_pool=SimpleNamespace(
            get_mamba_indices=lambda indices: indices.to(torch.int32).clone(),
            translate_mamba_indices=lambda indices: indices,
        ),
        token_to_kv_pool=None,
    )


class TestHybridTreeMetadata(CustomTestCase):
    def test_other_algorithms_and_draft_allocate_no_depth_buffers(self):
        for algorithm, topk, draft in (
            (None, None, False),
            ("EAGLE", 1, False),
            ("EAGLE", 2, False),
            ("UBTREE", 2, True),
        ):
            with self.subTest(algorithm=algorithm, topk=topk, draft=draft):
                with get_context().override_server_args(
                    speculative_algorithm=algorithm, speculative_eagle_topk=topk
                ):
                    backend = MambaAttnBackendBase(_runner(draft))
                    width = 1 if algorithm is None else 8
                    backend.init_cuda_graph_state(max_bs=3, max_num_tokens=3 * width)
                    self.assertIsNone(backend.tree_depth_list)
                    mode = (
                        ForwardMode.DECODE
                        if algorithm is None
                        else ForwardMode.TARGET_VERIFY
                    )
                    spec = SimpleNamespace(
                        ragged_verify_layout=None,
                        draft_token_num=width,
                        retrieve_next_token=torch.zeros((2, width), dtype=torch.int32),
                        retrieve_next_sibling=torch.full(
                            (2, width), -1, dtype=torch.int32
                        ),
                    )
                    requests = torch.tensor([1, 2], dtype=torch.int64)
                    captured = backend._capture_metadata(2, requests, mode, spec)
                    replayed = backend._replay_metadata(
                        2, requests, mode, spec, seq_lens_cpu=None
                    )
                    self.assertIsNone(captured.tree_depths)
                    self.assertIsNone(replayed.tree_depths)
                    torch.testing.assert_close(
                        replayed.query_start_loc,
                        torch.arange(0, 3 * width, width, dtype=torch.int32),
                    )
                    torch.testing.assert_close(
                        replayed.mamba_cache_indices, requests.to(torch.int32)
                    )
                    if topk == 2:
                        torch.testing.assert_close(
                            replayed.retrieve_next_sibling, spec.retrieve_next_sibling
                        )

    def test_ubtree_replay_and_post_draft_fixup_keep_captured_addresses(self):
        for width in (32, 49):
            with self.subTest(width=width):
                with get_context().override_server_args(
                    speculative_algorithm="UBTREE",
                    speculative_eagle_topk=2,
                    speculative_dflash_block_size=5,
                ):
                    backend = MambaAttnBackendBase(_runner())
                    backend.init_cuda_graph_state(1, width)
                    self.assertEqual(len(backend.tree_depth_list), 1)
                    spec = create_ubtree_verify_input(None, width)
                    requests = torch.tensor([1], dtype=torch.int64)
                    mode = ForwardMode.TARGET_VERIFY
                    captured = backend._capture_metadata(1, requests, mode, spec)
                    pointer = captured.tree_depths.data_ptr()
                    self.assertTrue(captured.has_direct_parent)
                    self.assertEqual(captured.tree_max_depth, 5)
                    for fanout in (3, 4):
                        parents = [-1] + [(i - 1) // fanout for i in range(1, width)]
                        depths = [0] * width
                        for i in range(1, width):
                            depths[i] = depths[parents[i]] + 1
                        spec.retrieve_parent_token = torch.tensor(
                            [parents], dtype=torch.int32
                        )
                        spec.tree_depths = torch.tensor([depths], dtype=torch.int32)
                        spec.retrieve_next_token = torch.zeros(
                            (1, width), dtype=torch.int32
                        )
                        spec.retrieve_next_sibling = torch.full(
                            (1, width), -1, dtype=torch.int32
                        )
                        replayed = backend._replay_metadata(
                            1, requests, mode, spec, seq_lens_cpu=None
                        )
                        self.assertEqual(replayed.tree_depths.data_ptr(), pointer)
                        torch.testing.assert_close(
                            captured.tree_depths, spec.tree_depths
                        )
                        torch.testing.assert_close(
                            captured.retrieve_parent_token, spec.retrieve_parent_token
                        )

                    # The plan stream may copy stale links before draft completes.
                    # The post-join hook must refresh both depths and parents.
                    spec.retrieve_parent_token = torch.tensor(
                        [[-1] + [0] * (width - 1)], dtype=torch.int32
                    )
                    spec.tree_depths = torch.tensor(
                        [[0] + [1] * (width - 1)], dtype=torch.int32
                    )
                    backend.update_verify_buffers_to_fill_after_draft(spec, 1)
                    self.assertEqual(captured.tree_depths.data_ptr(), pointer)
                    torch.testing.assert_close(captured.tree_depths, spec.tree_depths)
                    torch.testing.assert_close(
                        captured.retrieve_parent_token, spec.retrieve_parent_token
                    )


if __name__ == "__main__":
    unittest.main()
