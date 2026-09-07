"""CPU contracts for UNO tree compact target sampling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_utils import (
    _can_use_sparse_uno_tree_target_sampling,
)
from sglang.srt.speculative.uno_utils import sample_uno_tree_target_tokens
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestUnoTreeSparseSampling(CustomTestCase):
    def test_sparse_dispatch_guard(self):
        sampling_info = SimpleNamespace(
            sampling_seed=None,
            need_min_p_sampling=False,
        )
        spec_config = SimpleNamespace(
            speculative_use_rejection_sampling=False,
        )
        with (
            patch("sglang.srt.speculative.eagle_utils._is_cuda", True),
            patch(
                "sglang.srt.speculative.eagle_utils.get_spec",
                return_value=spec_config,
            ),
        ):
            self.assertTrue(
                _can_use_sparse_uno_tree_target_sampling(128, sampling_info)
            )
            self.assertFalse(
                _can_use_sparse_uno_tree_target_sampling(None, sampling_info)
            )
            self.assertFalse(
                _can_use_sparse_uno_tree_target_sampling(129, sampling_info)
            )

            sampling_info.sampling_seed = torch.tensor([1])
            self.assertFalse(
                _can_use_sparse_uno_tree_target_sampling(128, sampling_info)
            )
            sampling_info.sampling_seed = None
            sampling_info.need_min_p_sampling = True
            self.assertFalse(
                _can_use_sparse_uno_tree_target_sampling(128, sampling_info)
            )
            sampling_info.need_min_p_sampling = False
            spec_config.speculative_use_rejection_sampling = True
            self.assertFalse(
                _can_use_sparse_uno_tree_target_sampling(128, sampling_info)
            )

    def test_targets_are_sampled_from_compact_support(self):
        support_ids = torch.tensor(
            [
                [[10, 11], [20, 21], [30, 31]],
                [[40, 41], [50, 51], [60, 61]],
            ],
            dtype=torch.int64,
        )
        support_probs = torch.full((2, 3, 2), 0.5)
        sampled_offsets = torch.tensor(
            [[0], [1], [0], [1], [0], [1]],
            dtype=torch.long,
        )
        sampling_info = SimpleNamespace()
        next_token_logits = torch.empty((6, 100))

        with (
            patch(
                "sglang.srt.speculative.uno_utils._build_sparse_target_support",
                return_value=(support_ids, support_probs),
            ) as build_support,
            patch(
                "sglang.srt.speculative.uno_utils.fast_sample",
                return_value=(torch.empty((6, 1)), sampled_offsets),
            ),
        ):
            targets = sample_uno_tree_target_tokens(
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                batch_size=2,
                verify_width=3,
                max_top_k=2,
            )

        self.assertEqual(targets.tolist(), [[10, 21, 30], [41, 50, 61]])
        build_support.assert_called_once_with(
            next_token_logits=next_token_logits,
            sampling_info=sampling_info,
            batch_size=2,
            forward_width=3,
            max_top_k=2,
        )


if __name__ == "__main__":
    unittest.main()
