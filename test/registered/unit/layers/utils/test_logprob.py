"""Unit tests for srt/layers/utils/logprob.py — no server, no model loading."""

import math
import unittest
from unittest import mock

import torch

from sglang.srt.layers.utils.logprob import (
    LogprobStage,
    compute_temp_top_p_normalized_logprobs,
    get_token_ids_logprobs_chunk,
    get_token_ids_logprobs_raw,
    get_top_logprobs_chunk,
    get_top_logprobs_raw,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestGetTopLogprobs(CustomTestCase):
    """Tests functions that get topk logprobs and corresponding indices"""

    def test_get_top_logprobs_raw_no_copy_to_cpu(self):
        """
        Test correct behavior for no_copy_to_cpu for get_top_logprobs_raw.

        (Arbitrarily set logprobs for one sequence expecting top-1.)

        Check that:
            - vals and idxs are torch.Tensor if no_copy_to_cpu (reduce GPU->CPU overhead), else list.
        """
        logprobs = torch.full((1, 10), -10.0)
        logprobs[0, 3] = -0.1
        top_logprobs_nums = [1]

        vals, idxs = get_top_logprobs_raw(
            logprobs,
            top_logprobs_nums,
            LogprobStage.DECODE,
            no_copy_to_cpu=True,
        )

        self.assertIsInstance(vals[0], torch.Tensor)
        self.assertIsInstance(idxs[0], torch.Tensor)

        vals, idxs = get_top_logprobs_raw(
            logprobs,
            top_logprobs_nums,
            LogprobStage.DECODE,
            no_copy_to_cpu=False,
        )

        self.assertIsInstance(vals[0], list)
        self.assertIsInstance(idxs[0], list)

    def test_get_top_logprobs_raw_decode(self):
        """
        Test correctness of DECODE phase for get_top_logprobs_raw:

        Multiple sequences requesting varying top-k over their logprobs.

        Check that:
            - Each sequence receives the correct top-k indices and logprob vals.
        """

        num_rows, vocab_size = 3, 20

        logprobs = torch.full((num_rows, vocab_size), -10.0)

        # seq_0
        logprobs[0, 3] = -0.1

        # seq_1 (top-2 over logprobs)
        logprobs[1, 10] = -0.1
        logprobs[1, 8] = -0.2

        # seq_2 (top-3 over logprobs)
        logprobs[2, 9] = -0.1
        logprobs[2, 7] = -0.2
        logprobs[2, 12] = -0.3

        top_logprobs_nums = [1, 2, 3]

        vals, idxs = get_top_logprobs_raw(
            logprobs, top_logprobs_nums, LogprobStage.DECODE
        )

        self.assertEqual(idxs, [[3], [10, 8], [9, 7, 12]])
        torch.testing.assert_close(vals, [[-0.1], [-0.1, -0.2], [-0.1, -0.2, -0.3]])

    def test_get_top_logprobs_raw_prefill(self):
        """
        Test PREFILL phase for get_top_logprobs_raw:

        seq_1 has pruned_len = 0, so pt does not advance for seq_1.

        Check that:
            - seq_2 receives correct top-k indices and logprob vals.
            - Same for seq_0.
        """

        num_rows, vocab_size = 5, 10
        logprobs = torch.full((num_rows, vocab_size), -10.0)

        # seq_0 expects top-1 at idxs 3,6 for tokens 0,1
        logprobs[0, 3] = logprobs[1, 6] = -0.1

        # seq_2 expects top-2 for tokens 0,1,2 (2,3,4 in flattened logprobs tensor)
        logprobs[2, 2] = logprobs[3, 7] = logprobs[4, 3] = -0.1
        logprobs[2, 4] = logprobs[3, 9] = logprobs[4, 5] = -0.2

        top_logprobs_nums = [1, 1, 2]
        pruned_lens = [2, 0, 3]

        vals, idxs = get_top_logprobs_raw(
            logprobs,
            top_logprobs_nums,
            LogprobStage.PREFILL,
            extend_logprob_pruned_lens_cpu=pruned_lens,
        )

        torch.testing.assert_close(vals[0], [[-0.1], [-0.1]])
        self.assertEqual(vals[1], [])
        torch.testing.assert_close(vals[2], [[-0.1, -0.2], [-0.1, -0.2], [-0.1, -0.2]])
        self.assertEqual(idxs[0], [[3], [6]])
        self.assertEqual(idxs[1], [])
        self.assertEqual(idxs[2], [[2, 4], [7, 9], [3, 5]])

    def test_get_top_logprobs_chunk_split_boundary(self):
        """
        Test for correctness of get_top_logprobs_chunk:

        seq_0, seq_1, seq_2 all have pruned_len > 0.
        Chunk size is 3.

        Check that:
            - sequences split across chunk boundaries are correctly handled.
            - top-k indices and values are correct for all three sequences and their tokens.

        Changes to this test should only involve changing setup dict.
        """

        # seq_idx: (pruned_len (num tokens to process), top_k over tokens, logprobs per token position))
        setup = {
            0: {
                "pruned_len": 2,
                "top_k": 1,
                "token_probs": [
                    [(3, -0.1)],
                    [(6, -0.1)],
                ],
            },
            1: {
                "pruned_len": 3,
                "top_k": 1,
                "token_probs": [
                    [(2, -0.1)],
                    [(7, -0.1)],
                    [(9, -0.1)],
                ],
            },
            2: {
                "pruned_len": 6,
                "top_k": 2,
                "token_probs": [
                    [(1, -0.1), (2, -0.2)],
                    [(3, -0.1), (5, -0.2)],
                    [(7, -0.1), (9, -0.2)],
                    [(2, -0.1), (4, -0.2)],
                    [(6, -0.1), (9, -0.2)],
                    [(4, -0.1), (6, -0.2)],
                ],
            },
        }

        total_rows = sum(len(seq["token_probs"]) for seq in setup.values())
        max_token_id = max(
            token_id
            for seq in setup.values()
            for token_probs in seq["token_probs"]
            for token_id, _ in token_probs
        )

        logprobs = torch.full((total_rows, max_token_id + 1), -10.0)

        sorted_setup = sorted(setup.items())
        row = 0
        top_k_nums = []
        pruned_lens = []
        token_to_seq_idx = []
        logits_metadata = mock.MagicMock()

        # init logprobs with top-k probs and set inputs for get_top_logprobs_chunk
        # also: set up token_to_seq_idx for slicing top_k_nums and pruned_lens
        # (maps flattened first dimension of logprobs, to seq. idx)
        for seq_idx, seq_config in sorted_setup:
            top_k_nums.append(seq_config["top_k"])
            pruned_lens.append(seq_config["pruned_len"])
            token_to_seq_idx.extend([seq_idx] * max(seq_config["pruned_len"], 1))
            for token_probs in seq_config["token_probs"]:
                for token_id, lprob in token_probs:
                    logprobs[row, token_id] = lprob
                row += 1
        token_to_seq_idx.append(token_to_seq_idx[-1])

        logits_metadata.top_logprobs_nums = top_k_nums
        chunk_size = 3
        n = sum(pruned_lens)
        split_pruned_len = 0
        input_top_logprobs_val = []
        input_top_logprobs_idx = []

        # iterate over logprobs in chunks, update input_top_logprobs_val/idx
        for i in range(math.ceil(n / chunk_size)):
            start, end = i * chunk_size, min((i + 1) * chunk_size, n)
            chunk_slice = slice(
                token_to_seq_idx[start], token_to_seq_idx[end] + 1
            )  # +1 to include boundary seqs
            split_pruned_len = get_top_logprobs_chunk(
                logprobs[start:end],
                logits_metadata,
                top_k_nums[chunk_slice],
                pruned_lens[chunk_slice],
                input_top_logprobs_val,
                input_top_logprobs_idx,
                split_pruned_len,
            )

        # check input_top_logprobs_idx/val:
        # indices and values corresponding to top-k over seq_0,1,2 match with expected
        for seq_idx, seq_config in sorted_setup:
            self.assertEqual(
                len(input_top_logprobs_val[seq_idx]), seq_config["pruned_len"]
            )
            for token_pos, token_probs in enumerate(seq_config["token_probs"]):
                sorted_probs = sorted(token_probs, key=lambda x: -x[1])
                self.assertEqual(
                    input_top_logprobs_idx[seq_idx][token_pos],
                    [t for t, _ in sorted_probs],
                )
                torch.testing.assert_close(
                    input_top_logprobs_val[seq_idx][token_pos],
                    [lp for _, lp in sorted_probs],
                )

    def test_get_top_logprobs_chunk_zero_pruned_len_middle(self):
        """
        Test for correctness of get_top_logprobs_chunk:

        seq_1 has pruned_len = 0 (i.e. no tokens to process).
        Chunk size is 3.

        Check that:
            - seq_2 still receives correct top-2 ordering and logprob values for each token.
            - Check the same for seq_0, but top-1 instead.

        Changes to this test should only involve changing setup dict.
        """

        # seq_idx: (pruned_len (num tokens to process), top_k over tokens, logprobs per token position))
        setup = {
            0: {"pruned_len": 2, "top_k": 1, "token_probs": [[(3, -0.1)], [(6, -0.1)]]},
            1: {"pruned_len": 0, "top_k": 1, "token_probs": []},
            2: {
                "pruned_len": 6,
                "top_k": 2,
                "token_probs": [
                    [(1, -0.1), (2, -0.2)],
                    [(3, -0.1), (6, -0.2)],
                    [(8, -0.1), (9, -0.2)],
                    [(2, -0.1), (4, -0.2)],
                    [(7, -0.1), (8, -0.2)],
                    [(4, -0.1), (5, -0.2)],
                ],
            },
        }

        total_rows = sum(len(seq["token_probs"]) for seq in setup.values())
        max_token_id = max(
            token_id
            for seq in setup.values()
            for token_probs in seq["token_probs"]
            for token_id, _ in token_probs
        )
        vocab_size = max_token_id + 1
        logprobs = torch.full((total_rows, vocab_size), -10.0)

        sorted_setup = sorted(setup.items())
        row = 0
        top_k_nums = []
        pruned_lens = []
        token_to_seq_idx = []
        logits_metadata = mock.MagicMock()

        # init logprobs with top-k probs and set inputs for get_top_logprobs_chunk
        # also: set up token_to_seq_idx for slicing top_k_nums and pruned_lens
        # (maps flattened first dimension of logprobs, to seq. idx)
        for seq_idx, seq_config in sorted_setup:
            top_k_nums.append(seq_config["top_k"])
            pruned_lens.append(seq_config["pruned_len"])
            token_to_seq_idx.extend([seq_idx] * max(seq_config["pruned_len"], 1))
            for token_probs in seq_config["token_probs"]:
                for token_id, lprob in token_probs:
                    logprobs[row, token_id] = lprob
                row += 1
        token_to_seq_idx.append(token_to_seq_idx[-1])

        logits_metadata.top_logprobs_nums = top_k_nums
        chunk_size = 3
        n = sum(pruned_lens)

        split_pruned_len = 0
        input_top_logprobs_val = []
        input_top_logprobs_idx = []

        # iterate over logprobs in chunks, update input_top_logprobs_val/idx
        for i in range(math.ceil(n / chunk_size)):
            start, end = i * chunk_size, min((i + 1) * chunk_size, n)
            chunk_slice = slice(token_to_seq_idx[start], token_to_seq_idx[end] + 1)
            split_pruned_len = get_top_logprobs_chunk(
                logprobs[start:end],
                logits_metadata,
                top_k_nums[chunk_slice],
                pruned_lens[chunk_slice],
                input_top_logprobs_val,
                input_top_logprobs_idx,
                split_pruned_len,
            )

        # check input_top_logprobs_idx/val for correctness:
        # seq_1 empty, seq_0 and seq_2 have correct tokens
        for seq_idx, seq_config in sorted_setup:
            if not seq_config["token_probs"]:
                self.assertEqual(input_top_logprobs_idx[seq_idx], [])
                self.assertEqual(input_top_logprobs_val[seq_idx], [])
                continue

            for token_pos, token_probs in enumerate(seq_config["token_probs"]):
                sorted_probs = sorted(token_probs, key=lambda x: -x[1])
                self.assertEqual(
                    input_top_logprobs_idx[seq_idx][token_pos],
                    [t for t, _ in sorted_probs],
                )
                torch.testing.assert_close(
                    input_top_logprobs_val[seq_idx][token_pos],
                    [lp for _, lp in sorted_probs],
                )


class TestGetTokenIdsLogprobs(CustomTestCase):
    """Tests functions that perform look-up of logprobs for requested token IDs"""

    def test_get_token_ids_logprobs_raw_no_copy_to_cpu(self):
        """
        Test correct behavior for no_copy_to_cpu for get_token_ids_logprobs_raw.

        (Arbitrarily set logprobs for one sequence expecting one token ID.)

        Check that:
            - vals[0] is torch.Tensor if no_copy_to_cpu (reduce GPU->CPU overhead), else list.
        """

        logprobs = torch.full((1, 10), -10.0)
        logprobs[0, 3] = -0.1
        token_ids = [[3]]

        vals, idxs = get_token_ids_logprobs_raw(
            logprobs,
            token_ids,
            LogprobStage.DECODE,
            no_copy_to_cpu=True,
        )

        self.assertIsInstance(vals[0], torch.Tensor)

        vals, idxs = get_token_ids_logprobs_raw(
            logprobs,
            token_ids,
            LogprobStage.DECODE,
            no_copy_to_cpu=False,
        )

        self.assertIsInstance(vals[0], list)

    def test_get_token_ids_logprobs_raw_decode(self):
        """
        Test correctness of DECODE phase for get_token_ids_logprobs_raw:

        Multiple sequences requesting varying token_ids over their logprobs.

        Check that:
            - Each sequence receives the indices and logprob vals consistent with token_ids passed in.
        """

        num_rows, vocab_size = 3, 20

        logprobs = torch.full((num_rows, vocab_size), -10.0)

        logprobs[0, 3] = logprobs[1, 8] = logprobs[2, 7] = -0.1
        logprobs[1, 10] = logprobs[2, 9] = -0.2
        logprobs[2, 12] = -0.3

        token_ids = [[3], [8, 10], [7, 9, 12]]

        vals, idxs = get_token_ids_logprobs_raw(
            logprobs, token_ids, LogprobStage.DECODE
        )

        self.assertEqual(idxs, [[3], [8, 10], [7, 9, 12]])
        torch.testing.assert_close(vals, [[-0.1], [-0.1, -0.2], [-0.1, -0.2, -0.3]])

    def test_get_token_ids_logprobs_raw_prefill(self):
        """
        Test PREFILL phase for get_token_ids_logprobs_raw:

        seq_1 has pruned_len = 0, so pt does not advance for seq_1.

        Check that:
            - seq_2 receives indices and logprob vals consistent with token_ids passed in.
            - Same for seq_0.
        """

        num_rows, vocab_size = 5, 10
        logprobs = torch.full((num_rows, vocab_size), -10.0)

        # Look up seq_0 at idxs 3,6 for tokens 0,1
        logprobs[0, 3] = logprobs[0, 6] = logprobs[1, 3] = logprobs[1, 6] = -0.1

        # Look up seq_2 for tokens 0,1,2 (2,3,4 in flattened logprobs tensor)
        logprobs[2, 2] = logprobs[3, 2] = logprobs[4, 2] = -0.1
        logprobs[2, 4] = logprobs[3, 4] = logprobs[4, 4] = -0.2

        token_ids = [[3, 6], [], [2, 4]]
        pruned_lens = [2, 0, 3]

        vals, idxs = get_token_ids_logprobs_raw(
            logprobs,
            token_ids,
            LogprobStage.PREFILL,
            extend_logprob_pruned_lens_cpu=pruned_lens,
        )

        torch.testing.assert_close(vals[0], [[-0.1, -0.1], [-0.1, -0.1]])
        self.assertEqual(vals[1], [])
        torch.testing.assert_close(vals[2], [[-0.1, -0.2], [-0.1, -0.2], [-0.1, -0.2]])
        self.assertEqual(idxs[0], [[3, 6], [3, 6]])
        self.assertEqual(idxs[1], [])
        self.assertEqual(idxs[2], [[2, 4], [2, 4], [2, 4]])

    def test_get_token_ids_logprobs_chunk_split_boundary(self):
        """
        Test for correctness of get_token_ids_logprobs_chunk:

        seq_0, seq_1, seq_2 all have pruned_len > 0.
        Chunk size is 3.

        Check that:
            - sequences split across chunk boundaries are correctly handled.
            - indices and values correspond to token_ids for all three sequences and their tokens.

        Changes to this test should only involve changing setup dict.
        """

        # seq_idx: (pruned_len, token_ids to look up at every pos in the seq)
        setup = {
            0: {"pruned_len": 2, "token_ids": [3, 6]},
            1: {"pruned_len": 3, "token_ids": [2, 7]},
            2: {"pruned_len": 6, "token_ids": [1, 4]},
        }

        sorted_setup = sorted(setup.items())
        total_rows = sum(seq["pruned_len"] for seq in setup.values())
        max_token_id = max(tid for seq in setup.values() for tid in seq["token_ids"])
        vocab_size = max_token_id + 1
        logprobs = torch.full((total_rows, vocab_size), -10.0)

        row = 0
        pruned_lens = []
        token_ids_logprobs = []
        token_to_seq_idx = []

        # init logprobs with probs at token_ids, set inputs for get_top_logprobs_chunk, init token_to_seq_idx
        for seq_idx, seq_config in sorted_setup:
            pruned_lens.append(seq_config["pruned_len"])
            token_ids_logprobs.append(seq_config["token_ids"])
            token_to_seq_idx.extend([seq_idx] * max(seq_config["pruned_len"], 1))
            for _ in range(seq_config["pruned_len"]):
                for i, tid in enumerate(seq_config["token_ids"]):
                    logprobs[row, tid] = -0.1 * (i + 1)
                row += 1
        token_to_seq_idx.append(token_to_seq_idx[-1])

        chunk_size = 3
        n = sum(pruned_lens)
        split_pruned_len = 0
        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []

        # iterate over logprobs in chunks, update input_top_logprobs_val/idx
        for i in range(math.ceil(n / chunk_size)):
            start, end = i * chunk_size, min((i + 1) * chunk_size, n)
            chunk_slice = slice(token_to_seq_idx[start], token_to_seq_idx[end] + 1)
            split_pruned_len = get_token_ids_logprobs_chunk(
                logprobs[start:end],
                token_ids_logprobs[chunk_slice],
                pruned_lens[chunk_slice],
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
                split_pruned_len,
            )

        # check input_top_logprobs_idx/val:
        # indices and values corresponding to token_ids over seq_0,1,2 match with expected
        for seq_idx, seq_config in sorted_setup:
            expected_vals = [
                [-0.1 * (i + 1) for i in range(len(seq_config["token_ids"]))]
                for _ in range(seq_config["pruned_len"])
            ]
            expected_idxs = [seq_config["token_ids"]] * seq_config["pruned_len"]
            self.assertEqual(input_token_ids_logprobs_idx[seq_idx], expected_idxs)
            torch.testing.assert_close(
                input_token_ids_logprobs_val[seq_idx], expected_vals
            )

    def test_get_token_ids_logprobs_chunk_zero_pruned_len(self):
        """
        Test for correctness of get_token_ids_logprobs_chunk:

        seq_1 has pruned_len = 0 (i.e. no tokens to process).
        Chunk size is 3.

        Check that:
            - seq_2 still receives correct logprob values and indices corresponding to token_ids.
            - Check the same for seq_0.

        Changes to this test should only involve changing setup dict.
        """
        # seq_idx: (pruned_len, token_ids to look up at every pos in the seq)
        setup = {
            0: {"pruned_len": 2, "token_ids": [3, 6]},
            1: {"pruned_len": 0, "token_ids": []},
            2: {"pruned_len": 6, "token_ids": [1, 4]},
        }

        sorted_setup = sorted(setup.items())
        total_rows = sum(seq["pruned_len"] for seq in setup.values())
        max_token_id = max(tid for seq in setup.values() for tid in seq["token_ids"])
        vocab_size = max_token_id + 1
        logprobs = torch.full((total_rows, vocab_size), -10.0)

        row = 0
        pruned_lens = []
        token_ids_logprobs = []
        token_to_seq_idx = []

        # init logprobs with probs at token_ids, set inputs for get_top_logprobs_chunk, init token_to_seq_idx
        for seq_idx, seq_config in sorted_setup:
            pruned_lens.append(seq_config["pruned_len"])
            token_ids_logprobs.append(seq_config["token_ids"])
            token_to_seq_idx.extend([seq_idx] * max(seq_config["pruned_len"], 1))
            for _ in range(seq_config["pruned_len"]):
                for i, tid in enumerate(seq_config["token_ids"]):
                    logprobs[row, tid] = -0.1 * (i + 1)
                row += 1
        token_to_seq_idx.append(token_to_seq_idx[-1])

        chunk_size = 3
        n = sum(pruned_lens)
        split_pruned_len = 0
        input_token_ids_logprobs_val = []
        input_token_ids_logprobs_idx = []

        # iterate over logprobs in chunks, update input_top_logprobs_val/idx
        for i in range(math.ceil(n / chunk_size)):
            start, end = i * chunk_size, min((i + 1) * chunk_size, n)
            chunk_slice = slice(token_to_seq_idx[start], token_to_seq_idx[end] + 1)
            split_pruned_len = get_token_ids_logprobs_chunk(
                logprobs[start:end],
                token_ids_logprobs[chunk_slice],
                pruned_lens[chunk_slice],
                input_token_ids_logprobs_val,
                input_token_ids_logprobs_idx,
                split_pruned_len,
            )

        # check input_top_logprobs_idx/val:
        # indices and values corresponding to token_ids over seq_0,1,2 match with expected
        for seq_idx, seq_config in sorted_setup:
            if not seq_config["token_ids"]:
                self.assertEqual(input_token_ids_logprobs_idx[seq_idx], [])
                self.assertEqual(input_token_ids_logprobs_val[seq_idx], [])
                continue

            expected_vals = [
                [-0.1 * (i + 1) for i in range(len(seq_config["token_ids"]))]
                for _ in range(seq_config["pruned_len"])
            ]
            expected_idxs = [seq_config["token_ids"]] * seq_config["pruned_len"]
            self.assertEqual(input_token_ids_logprobs_idx[seq_idx], expected_idxs)
            torch.testing.assert_close(
                input_token_ids_logprobs_val[seq_idx], expected_vals
            )


class TestComputeLogprobs(CustomTestCase):
    """Tests functions that compute logprobs from raw logits"""

    def test_compute_temp_top_p_normalized_logprobs_temp_scaled(self):
        """
        Test temperature scaling branch of compute_temp_top_p_normalized_logprobs.

        Check that:
            - output equals log_softmax(logits / temperature) when temp_scaled_logprobs is True.
        """
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        temperature = torch.tensor([2.0])

        logits_metadata = mock.MagicMock()
        logits_metadata.temp_scaled_logprobs = True
        logits_metadata.top_p_normalized_logprobs = False

        vals = compute_temp_top_p_normalized_logprobs(
            logits, logits_metadata, temperature=temperature
        )

        expected = torch.nn.functional.log_softmax(logits / temperature, dim=-1)
        torch.testing.assert_close(vals, expected)

    def test_compute_temp_top_p_normalized_logprobs_top_p_normalized(self):
        """
        Test top_p normalization branch of compute_temp_top_p_normalized_logprobs.

        Check that:
            - top_p_normalize_probs_torch is called when top_p_normalized_logprobs is True and top_p != 1.0.
        """
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_p = torch.tensor([0.9])

        logits_metadata = mock.MagicMock()
        logits_metadata.temp_scaled_logprobs = False
        logits_metadata.top_p_normalized_logprobs = True

        with mock.patch(
            "sglang.srt.layers.sampler.top_p_normalize_probs_torch"
        ) as mock_normalize:
            mock_normalize.return_value = torch.softmax(logits, dim=-1)
            compute_temp_top_p_normalized_logprobs(logits, logits_metadata, top_p=top_p)
            mock_normalize.assert_called_once()

    def test_compute_temp_top_p_normalized_logprobs_top_p_all_one(self):
        """
        Test top_p == 1.0 fallback of compute_temp_top_p_normalized_logprobs.

        Check that:
            - top_p_normalize_probs_torch is NOT called when all top_p == 1.0.
            - output equals log_softmax of logits.
        """
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        top_p = torch.tensor([1.0])

        logits_metadata = mock.MagicMock()
        logits_metadata.temp_scaled_logprobs = False
        logits_metadata.top_p_normalized_logprobs = True

        with mock.patch(
            "sglang.srt.layers.sampler.top_p_normalize_probs_torch"
        ) as mock_normalize:
            vals = compute_temp_top_p_normalized_logprobs(
                logits, logits_metadata, top_p=top_p
            )
            mock_normalize.assert_not_called()

        expected = torch.nn.functional.log_softmax(logits, dim=-1)
        torch.testing.assert_close(vals, expected)


if __name__ == "__main__":
    unittest.main()
