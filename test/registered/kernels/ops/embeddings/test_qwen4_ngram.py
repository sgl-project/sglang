"""Correctness coverage for Qwen4 packed-prefill N-gram hashing."""

import unittest
from itertools import pairwise

import torch

from sglang.kernels.ops.embeddings import qwen4_ngram
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
EOS = 151645
HEADS = 16


def _metadata(lengths):
    lengths = torch.tensor(lengths, device="cuda", dtype=torch.long)
    return torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])


def _hash_parameters():
    multipliers = torch.tensor(
        [0x3FFFFFFFFFFF0001, 0x4FFFFFFFFFFF0003, 0x5FFFFFFFFFFF0005],
        device="cuda",
        dtype=torch.long,
    )
    vocab_sizes = torch.tensor(
        [
            1000003,
            1000033,
            1000037,
            1000039,
            1000081,
            1000099,
            1000117,
            1000121,
            1000133,
            1000151,
            1000159,
            1000171,
            1000183,
            1000187,
            1000193,
            1000199,
        ],
        device="cuda",
        dtype=torch.long,
    )
    offsets = torch.arange(HEADS, device="cuda", dtype=torch.long) * 1000200
    return multipliers, vocab_sizes, offsets


def _reference(input_ids, query_start_loc, history, multipliers, sizes, offsets):
    contexts = []
    starts = query_start_loc.cpu().tolist()
    for row, (start, end) in enumerate(pairwise(starts)):
        joined = torch.cat([history[row], input_ids[start:end]])
        for token in range(end - start):
            contexts.append(joined[token : token + 3])
    if not contexts:
        return input_ids.new_empty((0, HEADS))
    contexts = torch.stack(contexts)
    previous_1 = torch.where(contexts[:, 1] == EOS, EOS, contexts[:, 1])
    previous_2 = torch.where(
        (contexts[:, 0] == EOS) | (contexts[:, 1] == EOS),
        EOS,
        contexts[:, 0],
    )
    mixed_2 = contexts[:, 2] * multipliers[0]
    mixed_2 = torch.bitwise_xor(mixed_2, previous_1 * multipliers[1])
    mixed_3 = torch.bitwise_xor(mixed_2, previous_2 * multipliers[2])
    mixed = torch.cat(
        [
            mixed_2[:, None].expand(-1, HEADS // 2),
            mixed_3[:, None].expand(-1, HEADS // 2),
        ],
        dim=1,
    )
    return torch.remainder(mixed, sizes) + offsets


def _fused(input_ids, query_start_loc, history):
    multipliers, sizes, offsets = _hash_parameters()
    return qwen4_ngram.fused_qwen4_packed_ngram_hash(
        input_ids,
        query_start_loc,
        history,
        multipliers,
        sizes,
        offsets,
        EOS,
    )


@requires_cuda
class TestQwen4PackedNGram(CustomTestCase):
    def test_api_exists(self):
        self.assertTrue(
            hasattr(qwen4_ngram, "fused_qwen4_packed_ngram_hash"),
            "packed prefill N-gram fusion is not implemented",
        )

    def test_empty_rows_and_eos_match_reference(self):
        input_ids = torch.tensor(
            [7, EOS, 9, 10, EOS, 12], device="cuda", dtype=torch.long
        )
        query_start_loc = _metadata([0, 3, 0, 2, 1, 0])
        history = torch.tensor(
            [[1, 2], [3, EOS], [4, 5], [EOS, 6], [7, 8], [9, 10]],
            device="cuda",
            dtype=torch.long,
        )
        multipliers, sizes, offsets = _hash_parameters()
        expected = _reference(
            input_ids, query_start_loc, history, multipliers, sizes, offsets
        )
        actual = _fused(input_ids, query_start_loc, history)
        self.assertTrue(torch.equal(actual, expected))

    def test_empty_batch(self):
        input_ids = torch.empty(0, device="cuda", dtype=torch.long)
        query_start_loc = _metadata([0, 0, 0])
        history = torch.tensor(
            [[1, 2], [3, 4], [5, 6]], device="cuda", dtype=torch.long
        )
        self.assertEqual(
            tuple(_fused(input_ids, query_start_loc, history).shape), (0, 16)
        )

    def test_history_update_api_exists(self):
        self.assertTrue(
            hasattr(qwen4_ngram, "fused_qwen4_packed_ngram_update"),
            "packed N-gram history update is not implemented",
        )

    def test_history_writeback_handles_empty_rows_and_track_boundaries(self):
        input_ids = torch.tensor([10, 11, 12, 20], device="cuda")
        query_start_loc = _metadata([0, 3, 0, 1])
        history = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], device="cuda")
        context_pool = torch.full((10, 2), EOS, device="cuda")
        initial_pool = context_pool.clone()
        state_indices = torch.tensor([0, 2, 3, 4], device="cuda")
        track_indices = torch.tensor([0, 7, 8, 9], device="cuda")
        track_offsets = torch.tensor([0, 1, 0, 1], device="cuda")

        qwen4_ngram.fused_qwen4_packed_ngram_update(
            input_ids,
            query_start_loc,
            history,
            context_pool,
            state_indices,
            track_indices=track_indices,
            track_offsets=track_offsets,
        )
        self.assertTrue(torch.equal(context_pool[0], initial_pool[0]))
        self.assertTrue(
            torch.equal(context_pool[2], torch.tensor([11, 12], device="cuda"))
        )
        self.assertTrue(
            torch.equal(context_pool[4], torch.tensor([8, 20], device="cuda"))
        )
        self.assertTrue(
            torch.equal(context_pool[7], torch.tensor([4, 10], device="cuda"))
        )
        self.assertTrue(
            torch.equal(context_pool[9], torch.tensor([8, 20], device="cuda"))
        )
        self.assertTrue(torch.equal(context_pool[3], initial_pool[3]))
        self.assertTrue(torch.equal(context_pool[8], initial_pool[8]))

    def test_chunked_history_matches_one_shot(self):
        tokens = torch.tensor([11, 12, 13, 14, 15, 16], device="cuda")
        history = torch.tensor([[8, 9]], device="cuda")
        state_indices = torch.tensor([2], device="cuda")
        one_pool = torch.full((4, 2), EOS, device="cuda")
        qwen4_ngram.fused_qwen4_packed_ngram_update(
            tokens, _metadata([6]), history, one_pool, state_indices
        )

        chunk_pool = torch.full((4, 2), EOS, device="cuda")
        qwen4_ngram.fused_qwen4_packed_ngram_update(
            tokens[:2], _metadata([2]), history, chunk_pool, state_indices
        )
        qwen4_ngram.fused_qwen4_packed_ngram_update(
            tokens[2:],
            _metadata([4]),
            chunk_pool.index_select(0, state_indices),
            chunk_pool,
            state_indices,
        )
        self.assertTrue(torch.equal(chunk_pool, one_pool))

    def test_chunked_ids_match_one_shot(self):
        tokens = torch.tensor([11, 12, 13, 14, 15, 16], device="cuda")
        initial_history = torch.tensor([[8, 9]], device="cuda")
        one_shot = _fused(tokens, _metadata([6]), initial_history)

        first = _fused(tokens[:2], _metadata([2]), initial_history)
        second_history = tokens[:2].reshape(1, 2)
        second = _fused(tokens[2:], _metadata([4]), second_history)
        self.assertTrue(torch.equal(torch.cat([first, second]), one_shot))

    def test_signed_int64_overflow_matches_torch_remainder(self):
        input_ids = torch.tensor(
            [(1 << 62) + 17, (1 << 61) + 9, 123], device="cuda", dtype=torch.long
        )
        query_start_loc = _metadata([3])
        history = torch.tensor(
            [[(1 << 62) + 3, (1 << 62) + 5]], device="cuda", dtype=torch.long
        )
        multipliers, sizes, offsets = _hash_parameters()
        expected = _reference(
            input_ids, query_start_loc, history, multipliers, sizes, offsets
        )
        actual = _fused(input_ids, query_start_loc, history)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(bool((actual >= offsets).all()))

    def test_cuda_graph_replay_reads_live_layout_and_history(self):
        input_ids = torch.tensor([1, 2, 3, 4, 5], device="cuda")
        query_start_loc = _metadata([2, 3])
        history = torch.tensor([[EOS, 8], [9, 10]], device="cuda")
        multipliers, sizes, offsets = _hash_parameters()
        qwen4_ngram.fused_qwen4_packed_ngram_hash(
            input_ids,
            query_start_loc,
            history,
            multipliers,
            sizes,
            offsets,
            EOS,
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = qwen4_ngram.fused_qwen4_packed_ngram_hash(
                input_ids,
                query_start_loc,
                history,
                multipliers,
                sizes,
                offsets,
                EOS,
            )

        input_ids.copy_(torch.tensor([21, EOS, 23, 24, 25], device="cuda"))
        query_start_loc.copy_(_metadata([1, 4]))
        history.copy_(torch.tensor([[31, 32], [EOS, 34]], device="cuda"))
        expected = _reference(
            input_ids, query_start_loc, history, multipliers, sizes, offsets
        )
        graph.replay()
        self.assertTrue(torch.equal(actual, expected))

    def test_metadata_contract_rejects_fallback_inputs(self):
        input_ids = torch.tensor([1, 2, 3], device="cuda")
        query_start_loc = _metadata([1, 2])
        history = torch.tensor([[4, 5], [6, 7]], device="cuda")
        multipliers, sizes, offsets = _hash_parameters()
        can_fuse = qwen4_ngram.can_fuse_qwen4_packed_ngram_hash
        self.assertTrue(
            can_fuse(
                input_ids,
                query_start_loc,
                history,
                multipliers,
                sizes,
                offsets,
            )
        )
        self.assertFalse(
            can_fuse(
                input_ids.int(),
                query_start_loc,
                history,
                multipliers,
                sizes,
                offsets,
            )
        )

        empty_history = torch.empty((0, 2), device="cuda", dtype=torch.long)
        empty_starts = torch.zeros(1, device="cuda", dtype=torch.long)
        self.assertFalse(
            can_fuse(
                input_ids,
                empty_starts,
                empty_history,
                multipliers,
                sizes,
                offsets,
            )
        )
        self.assertFalse(
            can_fuse(
                input_ids,
                query_start_loc.int(),
                history,
                multipliers,
                sizes,
                offsets,
            )
        )

    def test_dispatch_keeps_nonordinary_paths_on_fallback(self):
        self.assertTrue(
            hasattr(qwen4_exp, "_use_qwen4_packed_ngram_prefill"),
            "packed N-gram dispatch is not implemented",
        )
        input_ids = torch.tensor([1, 2, 3], device="cuda")
        query_start_loc = _metadata([1, 2])
        history = torch.tensor([[4, 5], [6, 7]], device="cuda")
        multipliers, sizes, offsets = _hash_parameters()
        args = (
            query_start_loc,
            input_ids,
            history,
            multipliers,
            sizes,
            offsets,
        )
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.SPLIT_PREFILL,
        ):
            self.assertTrue(
                qwen4_exp._use_qwen4_packed_ngram_prefill(
                    mode, *args, dp_attention_enabled=False, offloaded=False
                )
            )
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            self.assertFalse(
                qwen4_exp._use_qwen4_packed_ngram_prefill(
                    mode, *args, dp_attention_enabled=False, offloaded=False
                )
            )
        self.assertFalse(
            qwen4_exp._use_qwen4_packed_ngram_prefill(
                ForwardMode.EXTEND,
                *args,
                dp_attention_enabled=True,
                offloaded=False,
            )
        )
        self.assertFalse(
            qwen4_exp._use_qwen4_packed_ngram_prefill(
                ForwardMode.EXTEND,
                *args,
                dp_attention_enabled=False,
                offloaded=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
