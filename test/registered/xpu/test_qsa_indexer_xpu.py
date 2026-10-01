"""XPU unit tests for the Qwen3.8-Flash-Next QSA (sparse-attention) indexer.

Mirrors test/registered/kernel/qsa/test_qsa.py's dispatch coverage, plus a
real-hardware regression test for the compress-group OOB gather (#38346).

A full E2E integration test would need real model weights or the reduced
dummy checkpoint used for local validation (neither is published).
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")


class _ForwardMode:
    def __init__(self, decode):
        self.decode = decode

    def is_decode(self):
        return self.decode


class _DispatchIndexer:
    """Mirrors test_qsa.py's _DispatchIndexer; used to prove forward_xpu
    dispatches prefill/decode identically to QSAIndexer's behavior."""

    layer_id = 3
    index_n_heads = 4
    compress_ratio = 4
    _pending_ring_slots = QSAIndexer._pending_ring_slots
    _forward_impl = QSAIndexer._forward_impl

    def __init__(self):
        self.selected = None
        self.logical_positions = None

    @staticmethod
    def project_qk(hidden_states, positions, **kwargs):
        rows = hidden_states.shape[0]
        return torch.zeros(rows, 4, 128), torch.zeros(rows, 1, 128), False

    def select_prefill_tokens(self, *args):
        self.selected = "prefill"
        return torch.tensor([1])

    def select_decode_tokens(self, *args):
        self.selected = "decode"
        return torch.tensor([2])

    def update_key_state_and_compress(
        self, token_k, logical_positions, rope_positions, metadata, **kwargs
    ):
        self.logical_positions = logical_positions.clone()


class _DispatchMetadata:
    token_to_kv_pool = None
    out_cache_loc = None
    compress_member_rows = None
    decode_logical_positions = None
    pending_ring_slots = None
    # Consumed by the real _pending_ring_slots helper the dispatch indexer
    # borrows: one token row owned by request slot 1.
    token_to_batch_idx = torch.zeros(2, dtype=torch.int32)
    req_pool_indices = torch.ones(2, dtype=torch.int32)
    sequence_lengths = torch.ones(2, dtype=torch.int32)

    @staticmethod
    def get_decode_mqa_inputs(layer_id):
        return (
            torch.zeros(2, 1, 1, 128),
            torch.zeros(1, 1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            1,
        )

    @staticmethod
    def get_prefill_mqa_inputs(layer_id, positions):
        return (
            torch.zeros(1, 1, 128),
            torch.zeros(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
            torch.ones(1, dtype=torch.int32),
        )

    @staticmethod
    def get_token_to_batch_idx():
        return torch.zeros(1, dtype=torch.int32)

    @staticmethod
    def get_seqlens_int32():
        return torch.ones(1, dtype=torch.int32)

    @staticmethod
    def get_seqlens_expanded():
        return torch.tensor([7], dtype=torch.int32)


class TestQSAIndexerForwardXpuDispatch(unittest.TestCase):
    """forward_xpu must route prefill/decode identically to QSAIndexer"""

    def test_forward_xpu_dispatches_prefill_and_decode_mqa(self):
        indexer = _DispatchIndexer()
        metadata = _DispatchMetadata()
        inputs = torch.zeros(1, 16)
        positions = torch.zeros(1, dtype=torch.int32)

        prefill_result = QSAIndexer.forward_xpu(
            indexer,
            inputs,
            positions,
            SimpleNamespace(forward_mode=_ForwardMode(False)),
            metadata,
        )
        self.assertEqual(indexer.selected, "prefill")
        self.assertEqual(indexer.logical_positions.tolist(), [0])
        self.assertEqual(prefill_result.item(), 1)

        positions.fill_(99)  # an arbitrary position for decode mode
        decode_result = QSAIndexer.forward_xpu(
            indexer,
            inputs,
            positions,
            SimpleNamespace(forward_mode=_ForwardMode(True)),
            metadata,
        )
        self.assertEqual(indexer.selected, "decode")
        # It should be _DispatchMetadata.get_seqlens_expanded() - 1
        self.assertEqual(indexer.logical_positions.tolist(), [6])
        # It should be _DispatchIndexer.select_decode_tokens(...)
        self.assertEqual(decode_result.item(), 2)


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestQSACompressGroupOOBClampOnRealXpu(unittest.TestCase):
    """Regression test for the compress-group OOB gather (#38346): calls
    ``QSAIndexer.update_key_state_and_compress`` with a short extend
    chunk that forces ``group_locs`` past the end of ``token_k``.
    Unclamped, XPU's bounds-checked SYCL gather aborts the process."""

    def _build_indexer(self):
        from sglang.srt.layers.rotary_embedding import get_rope

        config = SimpleNamespace(
            hidden_size=32,
            indexer_n_heads=2,
            indexer_kv_heads=1,
            indexer_head_dim=8,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            rms_norm_eps=1e-6,
        )
        rotary_emb = get_rope(
            head_size=8,
            rotary_dim=8,
            max_position=64,
            base=10000,
            is_neox_style=True,
            dtype=torch.float32,
        )
        indexer = QSAIndexer(config, layer_id=0, rotary_emb=rotary_emb)
        return indexer.to("xpu")

    def test_update_key_state_and_compress_clamps_oob_group_locs(self):
        indexer = self._build_indexer()
        num_rows = 6  # token_k has only 6 valid rows in this extend chunk.
        token_k = torch.randn(num_rows, 1, 8, dtype=torch.float32, device="xpu")

        # A worst-case capacity bound can reserve more groups than this short
        # chunk actually has complete members for; the last group's starting
        # row, uncompensated, walks the real gather past num_rows - 1.
        compressed_k = torch.zeros(3, 1, 8, dtype=torch.float32, device="xpu")
        metadata = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                set_qsa_compressed_k_buffer=lambda layer_id, loc, val: (
                    compressed_k.__setitem__(loc.long(), val)
                ),
            ),
            compress_member_rows=torch.tensor(
                [0, 4, 5], dtype=torch.long, device="xpu"
            ),
            is_cuda_graph=False,
            write_locs=torch.arange(3, dtype=torch.long, device="xpu"),
            compress_group_positions=torch.zeros(3, dtype=torch.long, device="xpu"),
            extend_rope_matrix=None,
        )

        # Must not raise/abort (this is the real production code path).
        indexer.update_key_state_and_compress(
            token_k,
            logical_positions=torch.arange(num_rows, device="xpu"),
            rope_positions=torch.arange(num_rows, device="xpu"),
            metadata=metadata,
            state_slots=None,
            state_stored=True,  # skip the pending-ring write; irrelevant here.
        )

        # The clamped tail group repeats row 5 (the last valid row) instead
        # of reading out of bounds. Build an unclamped equivalent -- a chunk
        # padded with copies of row 5 so the same group [5, 5, 5, 5] is
        # reached without any clamping -- and check both compress to the
        # same result.
        padded_token_k = torch.cat([token_k, token_k[-1:].expand(3, -1, -1)], dim=0)
        reference_k = torch.zeros(1, 1, 8, dtype=torch.float32, device="xpu")
        reference_metadata = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(
                set_qsa_compressed_k_buffer=lambda layer_id, loc, val: (
                    reference_k.__setitem__(loc.long(), val)
                ),
            ),
            compress_member_rows=torch.tensor([5], dtype=torch.long, device="xpu"),
            is_cuda_graph=False,
            write_locs=torch.zeros(1, dtype=torch.long, device="xpu"),
            compress_group_positions=torch.zeros(1, dtype=torch.long, device="xpu"),
            extend_rope_matrix=None,
        )
        indexer.update_key_state_and_compress(
            padded_token_k,
            logical_positions=torch.arange(padded_token_k.shape[0], device="xpu"),
            rope_positions=torch.arange(padded_token_k.shape[0], device="xpu"),
            metadata=reference_metadata,
            state_slots=None,
            state_stored=True,
        )
        torch.testing.assert_close(compressed_k[2], reference_k[0])


if __name__ == "__main__":
    unittest.main()
