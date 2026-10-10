"""Captured Mamba2 prefill metadata for the breakable prefill CUDA graph.

Capture lays out the batch's requests plus one pad sequence; replay refreshes
the same tensors in place with the live requests first, the pad sequence
absorbing the bucket's padding tokens and zero-length sequences filling the
remaining request slots, all pads on the reserved padding slot.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    Mamba2AttnBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAD_SLOT = 10


class _Pool:
    def get_mamba_indices(self, rows):
        return (rows + PAD_SLOT).to(torch.int32)

    def translate_mamba_indices(self, indices):
        return indices


def _backend():
    backend = object.__new__(Mamba2AttnBackend)
    backend.device = "cpu"
    backend._mamba_chunk_size = 256
    backend.req_to_token_pool = _Pool()
    backend._pad_mamba_index = None
    backend.use_captured_forward_metadata_for_breakable_cuda_graph = True
    return backend


def _capture_batch(num_tokens, seq_lens):
    return SimpleNamespace(
        input_ids=torch.zeros(num_tokens, dtype=torch.int64),
        batch_size=len(seq_lens),
        req_pool_indices=torch.arange(len(seq_lens)),
        extend_seq_lens_cpu=list(seq_lens),
    )


def _live_batch(req_pool_indices, seq_lens):
    starts = [0]
    for length in seq_lens[:-1]:
        starts.append(starts[-1] + length)
    return SimpleNamespace(
        batch_size=len(seq_lens),
        req_pool_indices=torch.tensor(req_pool_indices),
        extend_seq_lens=torch.tensor(seq_lens),
        extend_seq_lens_cpu=list(seq_lens),
        extend_start_loc=torch.tensor(starts),
    )


class TestMamba2BreakableCudaGraphMetadata(unittest.TestCase):
    def test_capture_adds_one_pad_sequence_and_covers_the_bucket(self):
        backend = _backend()
        metadata = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
            _capture_batch(16, [13, 1, 1, 1])
        )
        self.assertIs(backend.forward_metadata, metadata)
        self.assertEqual((metadata.num_prefills, metadata.num_decodes), (5, 0))
        self.assertEqual(metadata.num_prefill_tokens, 16)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 13, 14, 15, 16, 16])
        self.assertEqual(
            metadata.mamba_cache_indices.tolist(), [10, 11, 12, 13, PAD_SLOT]
        )
        mixed = metadata.mixed_metadata
        self.assertFalse(mixed.prep_initial_states)
        self.assertFalse(mixed.has_initial_states.any())
        self.assertEqual(mixed.seq_idx.tolist(), [[0] * 13 + [1, 2, 3]])
        # The causal-conv grid must reach the longest sequence a replay
        # can bring, and keep one program row per captured sequence.
        self.assertEqual(mixed.extend_seq_lens_cpu, [16, 0, 0, 0, 0])

    def test_replay_refreshes_the_captured_tensors_in_place(self):
        backend = _backend()
        metadata = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
            _capture_batch(16, [13, 1, 1, 1])
        )
        tensors = (
            metadata.query_start_loc,
            metadata.mamba_cache_indices,
            metadata.mixed_metadata.seq_idx,
        )
        backend.forward_metadata = None
        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            metadata, _live_batch([3, 7], [5, 4])
        )
        self.assertIs(backend.forward_metadata, metadata)
        for before, after in zip(
            tensors,
            (
                metadata.query_start_loc,
                metadata.mamba_cache_indices,
                metadata.mixed_metadata.seq_idx,
            ),
        ):
            self.assertIs(before, after)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 5, 9, 16, 16, 16])
        self.assertEqual(
            metadata.mamba_cache_indices.tolist(),
            [13, 17, PAD_SLOT, PAD_SLOT, PAD_SLOT],
        )
        self.assertEqual(
            metadata.mixed_metadata.seq_idx.tolist(), [[0] * 5 + [1] * 4 + [2] * 7]
        )

    def test_replay_of_an_exact_bucket_leaves_the_pad_sequence_empty(self):
        backend = _backend()
        metadata = backend.init_forward_metadata_for_breakable_cuda_graph_capture(
            _capture_batch(8, [7, 1])
        )
        backend.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            metadata, _live_batch([1, 2], [6, 2])
        )
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 6, 8, 8])
        self.assertEqual(metadata.mixed_metadata.seq_idx.tolist(), [[0] * 6 + [1] * 2])

    def test_request_slots_are_a_geometric_ladder_up_to_the_pool(self):
        backend = _backend()
        self.assertEqual(
            backend.breakable_cuda_graph_request_slots(64), (1, 2, 4, 8, 16, 32, 64)
        )
        self.assertEqual(
            backend.breakable_cuda_graph_request_slots(48), (1, 2, 4, 8, 16, 32, 48)
        )
        self.assertEqual(backend.breakable_cuda_graph_request_slots(1), (1,))

    def test_prefix_states_cannot_replay(self):
        backend = _backend()
        for prefix_lens, expected in (
            (None, True),
            ([0, 0], True),
            ([0, 3], False),
        ):
            with self.subTest(prefix_lens=prefix_lens):
                self.assertEqual(
                    backend.can_replay_breakable_cuda_graph(
                        batch_size=2, prefix_lens=prefix_lens
                    ),
                    expected,
                )


class TestHybridLinearAttnBackendCapturedMetadata(unittest.TestCase):
    def test_full_attention_stays_eager_while_mamba_is_captured(self):
        calls = []
        full = SimpleNamespace(
            init_forward_metadata=lambda batch: calls.append(("full", batch))
        )
        hybrid = object.__new__(HybridLinearAttnBackend)
        hybrid.full_attn_backend = full
        hybrid.linear_attn_backend = _backend()
        self.assertTrue(hybrid.use_captured_forward_metadata_for_breakable_cuda_graph)

        capture = _capture_batch(8, [7, 1])
        metadata = hybrid.init_forward_metadata_for_breakable_cuda_graph_capture(
            capture
        )
        live = _live_batch([4], [6])
        hybrid.prepare_forward_metadata_for_breakable_cuda_graph_replay(
            metadata,
            live,
            static_forward_batch=SimpleNamespace(input_ids=capture.input_ids),
        )
        self.assertEqual(calls, [("full", capture), ("full", live)])
        self.assertIs(hybrid.linear_attn_backend.forward_metadata, metadata)
        self.assertEqual(metadata.query_start_loc.tolist(), [0, 6, 8, 8])


if __name__ == "__main__":
    unittest.main()
