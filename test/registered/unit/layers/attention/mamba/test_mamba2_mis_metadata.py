import unittest

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    Mamba2Metadata,
    Mamba2MISMetadata,
    build_mamba2_mis_metadata,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMamba2MISMetadata(CustomTestCase):
    def test_single_request_segments_delimiters_and_chunk_alignment(self):
        mis_metadata = _build_metadata_for_single_req()
        self.assertIsInstance(mis_metadata, Mamba2MISMetadata)
        self.assertEqual(mis_metadata.chunk_size, 4)

        # Delimiters at 3, 10, 16 must be excluded from both query and item tokens.
        self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [7])
        self.assertEqual(mis_metadata.query_token_indices.tolist(), [0, 1, 2])
        self.assertEqual(mis_metadata.query_cu_seqlens.tolist(), [0, 3])
        self.assertEqual(mis_metadata.query_seq_lens_cpu, [3])
        self.assertEqual(mis_metadata.query_seq_idx.reshape(-1).tolist(), [0, 0, 0])
        self.assertEqual(mis_metadata.query_cache_indices.tolist(), [7])
        self.assertEqual(mis_metadata.query_cache_indices_i64.tolist(), [7])
        self.assertEqual(mis_metadata.query_has_initial_states.tolist(), [False])

        expected_item_tokens = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]
        self.assertEqual(mis_metadata.item_token_indices.tolist(), expected_item_tokens)
        self.assertEqual(mis_metadata.item_cu_seqlens.tolist(), [0, 6, 11])
        self.assertEqual(mis_metadata.item_seq_lens_cpu, [6, 5])
        self.assertEqual(
            mis_metadata.item_seq_idx.reshape(-1).tolist(),
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        )
        self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0, 1])
        self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [7, 7])
        self.assertEqual(mis_metadata.item_has_initial_states.tolist(), [True, True])

        # With chunk_size=4, Item 0 covers [0, 6) -> chunks (0, 0) and (1, 0),
        # and Item 1 covers [6, 11) -> chunks (1, 2) and (2, 0).
        self.assertEqual(mis_metadata.item_chunk_indices.tolist(), [0, 1, 1, 2])
        self.assertEqual(mis_metadata.item_chunk_offsets.tolist(), [0, 0, 2, 0])

        for delimiter_idx in (3, 10, 16):
            self.assertNotIn(delimiter_idx, mis_metadata.query_token_indices.tolist())
            self.assertNotIn(delimiter_idx, mis_metadata.item_token_indices.tolist())

    def test_mixed_batch_empty_query_and_dp_padding(self):
        # Req 0: seq_len=8, delimiters=[0, 2, 7] -> empty query, item 0 [1], item 1 [3..6]
        # Req 1: seq_len=6, delimiters=[2, 5] -> query [8, 9], item 2 [11, 12]
        # num_input_tokens=20 leaves trailing DP padding [14..19] that must be ignored.
        mis_metadata = build_mamba2_mis_metadata(
            is_prefill_only=True,
            extend_prefix_lens_cpu=[0, 0],
            extend_seq_lens_cpu=[8, 6],
            multi_item_delimiter_indices=[
                torch.tensor([0, 2, 7], dtype=torch.int64),
                torch.tensor([2, 5], dtype=torch.int64),
            ],
            num_input_tokens=20,
            cache_indices=torch.tensor([4, 9], dtype=torch.int32),
            chunk_size=4,
            device=torch.device("cpu"),
        )

        # Every request slot is cleared up front before segment execution.
        self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [4, 9])

        self.assertEqual(mis_metadata.query_token_indices.tolist(), [8, 9])
        self.assertEqual(mis_metadata.query_cu_seqlens.tolist(), [0, 2])
        self.assertEqual(mis_metadata.query_seq_lens_cpu, [2])
        self.assertEqual(mis_metadata.query_cache_indices.tolist(), [9])
        self.assertEqual(mis_metadata.query_has_initial_states.tolist(), [False])

        self.assertEqual(
            mis_metadata.item_token_indices.tolist(), [1, 3, 4, 5, 6, 11, 12]
        )
        self.assertEqual(mis_metadata.item_cu_seqlens.tolist(), [0, 1, 5, 7])
        self.assertEqual(mis_metadata.item_seq_lens_cpu, [1, 4, 2])
        self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0, 1, 2])
        self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [4, 4, 9])
        self.assertEqual(
            mis_metadata.item_has_initial_states.tolist(), [True, True, True]
        )
        self.assertEqual(mis_metadata.item_chunk_indices.tolist(), [0, 0, 1, 1])
        self.assertEqual(mis_metadata.item_chunk_offsets.tolist(), [0, 1, 0, 1])

        all_used = set(mis_metadata.query_token_indices.tolist()) | set(
            mis_metadata.item_token_indices.tolist()
        )
        for padded_pos in range(14, 20):
            self.assertNotIn(padded_pos, all_used)

    def test_empty_query_single_token_item_batch(self):
        mis_metadata = build_mamba2_mis_metadata(
            is_prefill_only=True,
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[3],
            multi_item_delimiter_indices=[torch.tensor([0, 2], dtype=torch.int64)],
            num_input_tokens=3,
            cache_indices=torch.tensor([3], dtype=torch.int32),
            chunk_size=4,
            device=torch.device("cpu"),
        )
        self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [3])
        self.assertEqual(mis_metadata.query_token_indices.numel(), 0)
        self.assertEqual(mis_metadata.query_cu_seqlens.tolist(), [0])
        self.assertEqual(mis_metadata.item_token_indices.tolist(), [1])
        self.assertEqual(mis_metadata.item_cu_seqlens.tolist(), [0, 1])
        self.assertEqual(mis_metadata.item_seq_lens_cpu, [1])
        self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0])
        self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [3])
        self.assertEqual(mis_metadata.item_has_initial_states.tolist(), [True])
        self.assertEqual(mis_metadata.item_chunk_indices.tolist(), [0])
        self.assertEqual(mis_metadata.item_chunk_offsets.tolist(), [0])

    def test_cache_indices_dtypes_and_caller_mutation_isolation(self):
        for cache_dtype in (torch.int32, torch.int64):
            with self.subTest(cache_dtype=cache_dtype):
                cache_indices = torch.tensor([11], dtype=cache_dtype)
                delims = [torch.tensor([2, 5], dtype=torch.int64)]
                mis_metadata = build_mamba2_mis_metadata(
                    is_prefill_only=True,
                    extend_prefix_lens_cpu=[0],
                    extend_seq_lens_cpu=[6],
                    multi_item_delimiter_indices=delims,
                    num_input_tokens=6,
                    cache_indices=cache_indices,
                    chunk_size=4,
                    device=torch.device("cpu"),
                )
                self.assertEqual(mis_metadata.query_cache_indices.dtype, torch.int32)
                self.assertEqual(mis_metadata.item_conv_indices.dtype, torch.int32)
                self.assertEqual(
                    mis_metadata.query_cache_indices_i64.dtype, torch.int64
                )
                self.assertEqual(mis_metadata.item_ssm_indices_i64.dtype, torch.int64)
                self.assertEqual(mis_metadata.item_conv_indices.tolist(), [0])

                # Caller mutations after metadata build must not alias stored metadata tensors.
                cache_indices.fill_(999)
                delims[0].fill_(0)
                self.assertEqual(mis_metadata.reset_cache_indices.tolist(), [11])
                self.assertEqual(mis_metadata.query_cache_indices.tolist(), [11])
                self.assertEqual(mis_metadata.query_cache_indices_i64.tolist(), [11])
                self.assertEqual(mis_metadata.item_ssm_indices_i64.tolist(), [11])
                self.assertEqual(mis_metadata.item_token_indices.tolist(), [3, 4])

    def test_batch_validation_errors(self):
        invalid_cases = [
            ("non_prefill_only", {"is_prefill_only": False}, "prefill-only requests"),
            ("non_zero_prefix", {"extend_prefix_lens_cpu": [1]}, "cached prefixes"),
            (
                "missing_delims",
                {"multi_item_delimiter_indices": None},
                "delimiter indices for every request",
            ),
            (
                "wrong_delims_len",
                {"multi_item_delimiter_indices": []},
                "delimiter indices for every request",
            ),
            (
                "tokens_exceed_input",
                {"num_input_tokens": 5},
                "exceed the input tokens",
            ),
            ("non_positive_chunk_size", {"chunk_size": 0}, "positive chunk_size"),
            ("negative_chunk_size", {"chunk_size": -4}, "positive chunk_size"),
            (
                "2d_cache_indices",
                {"cache_indices": torch.tensor([[1]], dtype=torch.int32)},
                "1D integer cache_indices",
            ),
            (
                "wrong_len_cache_indices",
                {"cache_indices": torch.tensor([1, 2], dtype=torch.int32)},
                "1D integer cache_indices",
            ),
            (
                "float_cache_indices",
                {"cache_indices": torch.tensor([1.0], dtype=torch.float32)},
                "1D integer cache_indices",
            ),
            (
                "meta_cache_indices",
                {"cache_indices": torch.empty(1, dtype=torch.int32, device="meta")},
                "1D integer cache_indices",
            ),
        ]
        _assert_invalid_mis_cases(testcase=self, invalid_cases=invalid_cases)

    def test_delimiter_validation_errors(self):
        invalid_cases = [
            (
                "too_few_delims",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([5], dtype=torch.int64)
                    ]
                },
                "at least two delimiters",
            ),
            (
                "duplicate_delims",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 2, 5], dtype=torch.int64)
                    ]
                },
                "strictly increasing",
            ),
            (
                "decreasing_delims",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([3, 2, 5], dtype=torch.int64)
                    ]
                },
                "strictly increasing",
            ),
            (
                "negative_delim",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([-1, 5], dtype=torch.int64)
                    ]
                },
                "outside the request",
            ),
            (
                "overflow_delim",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 6], dtype=torch.int64)
                    ]
                },
                "outside the request",
            ),
            (
                "final_delim_not_last",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 4], dtype=torch.int64)
                    ]
                },
                "final delimiter must be the last request token",
            ),
            (
                "empty_single_item",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 3, 5], dtype=torch.int64)
                    ]
                },
                "does not support empty items",
            ),
            (
                "trailing_empty_item",
                {
                    "multi_item_delimiter_indices": [
                        torch.tensor([2, 4, 5], dtype=torch.int64)
                    ]
                },
                "does not support empty items",
            ),
            (
                "empty_item_after_empty_query",
                {
                    "extend_seq_lens_cpu": [2],
                    "num_input_tokens": 2,
                    "multi_item_delimiter_indices": [
                        torch.tensor([0, 1], dtype=torch.int64)
                    ],
                },
                "does not support empty items",
            ),
        ]
        _assert_invalid_mis_cases(testcase=self, invalid_cases=invalid_cases)

    def test_prepare_mis_validates_radix_and_mode_guards(self):
        delimiters = [torch.tensor([2, 5], dtype=torch.int64)]
        md = Mamba2Metadata.prepare_mis(
            cache_indices=torch.tensor([7, 9], dtype=torch.int32),
            replayssm_ring_indices=torch.tensor([11, 13], dtype=torch.int32),
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[6],
            multi_item_delimiter_indices=delimiters,
            num_input_tokens=6,
            chunk_size=4,
            device=torch.device("cpu"),
            is_prefill_only=True,
            is_extend=True,
            is_target_verify=False,
            mamba_prefill_track_mask_cpu=[False],
            has_mamba_track_mask=False,
            has_mamba_track_indices=False,
        )
        self.assertIsNone(md.mixed_metadata)
        self.assertIsInstance(md.mis_metadata, Mamba2MISMetadata)
        self.assertEqual(md.mis_metadata.chunk_size, 4)
        self.assertEqual(md.mis_metadata.query_token_indices.tolist(), [0, 1])
        self.assertEqual(md.mis_metadata.item_token_indices.tolist(), [3, 4])
        self.assertEqual(md.mamba_cache_indices.tolist(), [7])
        self.assertEqual(md.replayssm_ring_indices.tolist(), [11])
        self.assertEqual(md.num_prefills, 1)
        self.assertEqual(md.num_prefill_tokens, 6)
        self.assertEqual(md.num_decodes, 0)

        cases = [
            (
                "cpu_track_mask",
                {"mamba_prefill_track_mask_cpu": [True]},
                "disable-radix-cache",
            ),
            (
                "gpu_track_mask_without_cpu",
                {
                    "mamba_prefill_track_mask_cpu": None,
                    "has_mamba_track_mask": True,
                },
                "disable-radix-cache",
            ),
            (
                "track_indices_present",
                {"has_mamba_track_indices": True},
                "disable-radix-cache",
            ),
            ("non_prefill_only", {"is_prefill_only": False}, "prefill-only"),
            ("not_extend", {"is_extend": False}, "prefill-only"),
            ("target_verify", {"is_target_verify": True}, "prefill-only"),
            (
                "missing_extend_seq_lens",
                {"extend_seq_lens_cpu": None},
                "extend_seq_lens_cpu",
            ),
            (
                "missing_extend_prefix_lens",
                {"extend_prefix_lens_cpu": None},
                "extend_prefix_lens_cpu",
            ),
        ]
        base_kwargs = {
            "cache_indices": torch.tensor([7], dtype=torch.int32),
            "replayssm_ring_indices": None,
            "extend_prefix_lens_cpu": [0],
            "extend_seq_lens_cpu": [6],
            "multi_item_delimiter_indices": delimiters,
            "num_input_tokens": 6,
            "chunk_size": 4,
            "device": torch.device("cpu"),
            "is_prefill_only": True,
            "is_extend": True,
            "is_target_verify": False,
            "mamba_prefill_track_mask_cpu": [False],
            "has_mamba_track_mask": False,
            "has_mamba_track_indices": False,
        }
        for name, overrides, expected_msg in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, expected_msg):
                    Mamba2Metadata.prepare_mis(**dict(base_kwargs, **overrides))


def _assert_invalid_mis_cases(*, testcase: CustomTestCase, invalid_cases: list):
    base_kwargs = {
        "is_prefill_only": True,
        "extend_prefix_lens_cpu": [0],
        "extend_seq_lens_cpu": [6],
        "multi_item_delimiter_indices": [torch.tensor([2, 5], dtype=torch.int64)],
        "num_input_tokens": 6,
        "cache_indices": torch.tensor([1], dtype=torch.int32),
        "chunk_size": 4,
        "device": torch.device("cpu"),
    }
    for name, overrides, expected_msg in invalid_cases:
        with testcase.subTest(name=name):
            kwargs = dict(base_kwargs, **overrides)
            with testcase.assertRaisesRegex(ValueError, expected_msg):
                build_mamba2_mis_metadata(**kwargs)


def _build_metadata_for_single_req():
    return build_mamba2_mis_metadata(
        is_prefill_only=True,
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[17],
        multi_item_delimiter_indices=[torch.tensor([3, 10, 16], dtype=torch.int64)],
        num_input_tokens=17,
        cache_indices=torch.tensor([7], dtype=torch.int32),
        chunk_size=4,
        device=torch.device("cpu"),
    )


if __name__ == "__main__":
    unittest.main()
