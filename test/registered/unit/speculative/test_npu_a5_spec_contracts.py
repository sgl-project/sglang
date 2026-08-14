from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.speculative.npu_reference import (
    build_retrieval_links_reference,
    read_cache_locations_reference,
)
from sglang.srt.environ import envs
from sglang.srt.speculative.eagle_zero_bubble import (
    pad_zero_bubble_seed,
    validate_prefetched_topk1,
    validate_zero_bubble_config,
)
from sglang.srt.speculative.npu_sampling import validate_npu_target_only_sampling
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNpuA5SpecContracts(CustomTestCase):
    def test_cache_location_reference_has_no_max_step(self):
        token_pool = torch.arange(3 * 512, dtype=torch.long).view(3, 512)
        req_pool_indices = torch.tensor([1, 0])
        start_offset = torch.tensor([7, 3])
        end_offset = torch.tensor([264, 214])

        actual = read_cache_locations_reference(
            req_pool_indices=req_pool_indices,
            token_pool=token_pool,
            start_offset=start_offset,
            end_offset=end_offset,
        )
        expected = torch.cat((token_pool[1, 7:264], token_pool[0, 3:214]))

        self.assertEqual(actual.numel(), 468)
        self.assertTrue(torch.equal(actual, expected))

    def test_topk_two_tree_keeps_child_and_sibling_links(self):
        retrieve_index, next_token, next_sibling = build_retrieval_links_reference(
            parent_list=torch.tensor([[0, 0]]),
            selected_index=torch.tensor([[0, 1, 2]]),
            topk=2,
            draft_token_num=4,
        )

        self.assertEqual(retrieve_index.tolist(), [[0, 1, 2, 3]])
        self.assertEqual(int(next_token[0, 0]), 1)
        self.assertEqual(int(next_sibling[0, 1]), 2)
        self.assertEqual(int(next_token[0, 1]), 3)

    def test_zero_bubble_defaults_off_and_pads_seed(self):
        self.assertIs(envs.SGLANG_SPEC_V2_ZERO_BUBBLE.default, False)
        with envs.SGLANG_SPEC_V2_ZERO_BUBBLE.override(False):
            self.assertFalse(envs.SGLANG_SPEC_V2_ZERO_BUBBLE.get())

        probabilities = torch.ones((2, 1))
        indices = torch.tensor([[5], [7]])
        padded_probabilities, padded_indices = pad_zero_bubble_seed(
            topk_p=probabilities,
            topk_index=indices,
            num_steps=4,
            topk=1,
        )
        self.assertEqual(tuple(padded_indices.shape), (2, 4))
        self.assertEqual(padded_indices.tolist(), [[5, 0, 0, 0], [7, 0, 0, 0]])
        self.assertEqual(padded_probabilities[:, 1:].count_nonzero().item(), 0)
        validate_prefetched_topk1(
            topk_index=padded_indices,
            batch_size=2,
            num_steps=4,
        )

    def test_zero_bubble_rejects_unsupported_modes(self):
        valid = dict(
            enabled=True,
            topk=1,
            num_steps=4,
            enable_multi_layer_eagle=False,
            is_eagle3=False,
            use_rejection_sampling=False,
            speculative_adaptive=False,
        )
        validate_zero_bubble_config(**valid)
        for key, value in (
            ("topk", 2),
            ("use_rejection_sampling", True),
            ("speculative_adaptive", True),
        ):
            invalid = valid | {key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_zero_bubble_config(**invalid)

    def test_target_only_sampling_validation(self):
        valid = dict(
            tree_topk=1,
            num_draft_tokens=4,
            max_tree_depth=4,
            retrieve_index_shape=(2, 4),
            logits_shape=(8, 32000),
            batch_size=2,
            use_rejection_sampling=False,
            threshold_single=1.0,
            threshold_acc=1.0,
            sampling_backend="ascend",
        )
        validate_npu_target_only_sampling(**valid)
        for key, value, error_type in (
            ("tree_topk", 2, NotImplementedError),
            ("use_rejection_sampling", True, NotImplementedError),
            ("threshold_single", 0.9, ValueError),
            ("retrieve_index_shape", (2, 3), ValueError),
        ):
            invalid = valid | {key: value}
            with self.subTest(key=key), self.assertRaises(error_type):
                validate_npu_target_only_sampling(**invalid)


if __name__ == "__main__":
    unittest.main()
