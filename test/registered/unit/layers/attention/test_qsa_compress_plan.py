"""The QSA extend compression plan against chunks shorter than one group.

An extend chunk can carry fewer token rows than the compression ratio of 4. The
write plan is padded to a shape-derived capacity and a padding entry names token
row 0, whose group window spans rows [0, ratio) of this forward's packed tensors
-- rows a short chunk does not have. Gathering them killed the engine with a
device-side assert on every rank.

Two independent ways to get such a chunk, both measured on the pre-fix engine:
a repeated prompt whose page-granular prefix hit leaves a 1-3 token tail, and a
single cold prompt whose length leaves a 1-3 token remainder past the
chunked-prefill boundary (4097 tokens at chunked_prefill_size 4096 is enough).
The second needs no cache, no repetition and no concurrency, so flushing the
prefix cache is not a workaround.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.runtime_context import publish
from sglang.srt.server_args import ServerArgs

RATIO = 4
HEAD_DIM = 128
HIDDEN = 256
PAGE_SIZE = 16
# Ratio-aligned and non-zero, so each row's compressed slots are disjoint and
# none of them collides with the plan's reserved padding slot 0.
SLOT_BASE_STRIDE = 1024


class _FakePool:
    """The three QSA pool buffers the compression step touches."""

    def __init__(self, num_slots: int, num_compressed: int):
        self.key_state = torch.zeros(num_slots, 1, HEAD_DIM)
        self.qsa_rope_position_buffer = torch.zeros(num_slots, 3, dtype=torch.int64)
        self.compressed = torch.zeros(num_compressed, 1, HEAD_DIM)

    def get_qsa_key_state_buffer(self, layer_id):
        return self.key_state

    def set_qsa_key_state_buffer(self, layer_id, loc, token_k):
        self.key_state[loc.long()] = token_k.to(self.key_state.dtype)

    def set_qsa_rope_position_buffer(self, loc, positions):
        positions = positions.long()
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(3, -1)
        self.qsa_rope_position_buffer[loc.long()] = positions.transpose(0, 1)

    def get_qsa_compressed_k_buffer(self, layer_id):
        return self.compressed

    def set_qsa_compressed_k_buffer(self, layer_id, loc, compressed_k):
        self.compressed[loc.long()] = compressed_k.to(self.compressed.dtype)


def _make_indexer():
    publish(ServerArgs(model_path="dummy"), role="test")
    config = SimpleNamespace(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=2048,
        indexer_compress_ratio=RATIO,
        hidden_size=HIDDEN,
        rms_norm_eps=1e-6,
    )
    rotary = MRotaryEmbedding(
        head_size=HEAD_DIM,
        rotary_dim=HEAD_DIM,
        max_position_embeddings=32768,
        base=1000000,
        is_neox_style=True,
        dtype=torch.float32,
        mrope_section=None,
        mrope_interleaved=False,
    )
    return QSAIndexer(config, layer_id=0, quant_config=None, rotary_emb=rotary)


def _extend_metadata(pool, prefix_len: int, extend_len: int):
    """One request resuming from ``prefix_len`` cached tokens, via the real plan."""
    seq_len = prefix_len + extend_len
    lengths = torch.tensor([seq_len])
    prefix_lens = torch.tensor([prefix_len])
    # Position i holds raw slot i + 1, keeping compressed slot 0 free for the
    # plan's inert padding write.
    token_slot_table = torch.arange(1, seq_len + 1, dtype=torch.int32).unsqueeze(0)
    write_locs, group_positions, sequence_ids, member_rows = (
        QwenSparseAttnBackend._qsa_write_plan(
            token_slot_table=token_slot_table,
            start_blocks=prefix_lens // RATIO,
            end_blocks=lengths // RATIO,
            capacity=extend_len // RATIO + 1,
            compress_ratio=RATIO,
            row_token_starts=torch.tensor([0]),
            prefix_lens=prefix_lens,
        )
    )
    return SimpleNamespace(
        token_to_kv_pool=pool,
        is_cuda_graph=False,
        write_locs=write_locs,
        compress_group_positions=group_positions,
        compress_sequence_ids=sequence_ids,
        compress_member_rows=member_rows,
        extend_rope_matrix=None,
        token_slot_table=token_slot_table,
        sequence_lengths=lengths,
    )


def _extend_metadata_rows(pool, rows):
    """Several (prefix_len, extend_len) requests packed into one forward.

    Row r takes raw slots from its own ratio-aligned base so no two rows can
    land on the same compressed slot, and every base is non-zero so a write to
    the plan's reserved padding slot 0 stays distinguishable from a real one.
    """
    prefix_lens = torch.tensor([p for p, _ in rows])
    extend_lens = torch.tensor([e for _, e in rows])
    lengths = prefix_lens + extend_lens
    table = torch.zeros(len(rows), int(lengths.max()), dtype=torch.int32)
    for r, seq_len in enumerate(lengths.tolist()):
        base = (r + 1) * SLOT_BASE_STRIDE
        table[r, :seq_len] = torch.arange(base, base + seq_len, dtype=torch.int32)
    write_locs, group_positions, sequence_ids, member_rows = (
        QwenSparseAttnBackend._qsa_write_plan(
            token_slot_table=table,
            start_blocks=prefix_lens // RATIO,
            end_blocks=lengths // RATIO,
            capacity=int(extend_lens.sum()) // RATIO + len(rows),
            compress_ratio=RATIO,
            row_token_starts=torch.cumsum(extend_lens, 0) - extend_lens,
            prefix_lens=prefix_lens,
        )
    )
    return SimpleNamespace(
        token_to_kv_pool=pool,
        is_cuda_graph=False,
        write_locs=write_locs,
        compress_group_positions=group_positions,
        compress_sequence_ids=sequence_ids,
        compress_member_rows=member_rows,
        extend_rope_matrix=None,
        token_slot_table=table,
        sequence_lengths=lengths,
    )


def _compressed_slot(row: int, block: int) -> int:
    """The compressed slot row ``row``'s block ``block`` writes, per the plan."""
    return ((row + 1) * SLOT_BASE_STRIDE) // RATIO + block


class TestQsaExtendCompressPlan(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.indexer = _make_indexer()
        # Wide enough for the multi-row cases, whose per-row slot bases reach
        # into the hundreds.
        self.pool = _FakePool(num_slots=2048, num_compressed=2048)

    def _compress(self, prefix_len: int, extend_len: int) -> torch.Tensor:
        # gemma_rmsnorm and the RoPE tables are CUDA-only and covered by the
        # kernel tests; this one is about which rows the plan gathers.
        self.indexer.normalize_compressed_keys = lambda pooled, positions: pooled
        metadata = _extend_metadata(self.pool, prefix_len, extend_len)
        token_k = torch.randn(extend_len, 1, HEAD_DIM)
        positions = torch.arange(prefix_len, prefix_len + extend_len)
        self.indexer.update_key_state_and_compress(
            token_k,
            positions,
            positions,
            metadata,
            state_stored=True,
        )
        return token_k

    def test_chunk_shorter_than_a_group_writes_nothing(self):
        for extend_len in range(1, RATIO):
            with self.subTest(extend_len=extend_len):
                self.pool.compressed.zero_()
                self._compress(prefix_len=PAGE_SIZE * 4, extend_len=extend_len)
                self.assertEqual(int(self.pool.compressed.count_nonzero()), 0)

    def test_chunk_of_exactly_one_group_still_compresses_it(self):
        token_k = self._compress(prefix_len=PAGE_SIZE * 4, extend_len=RATIO)
        # The one planned group covers tokens 64-67, whose first raw slot is 65;
        # slot 0 is the padding entry's reserved inert write.
        written = self.pool.compressed.reshape(self.pool.compressed.shape[0], -1)
        self.assertEqual(
            written.count_nonzero(dim=1).nonzero().flatten().tolist(), [0, 65 // RATIO]
        )
        torch.testing.assert_close(
            self.pool.compressed[65 // RATIO], token_k.mean(dim=0)
        )

    def _compress_rows(self, rows) -> torch.Tensor:
        self.indexer.normalize_compressed_keys = lambda pooled, positions: pooled
        metadata = _extend_metadata_rows(self.pool, rows)
        token_k = torch.randn(sum(e for _, e in rows), 1, HEAD_DIM)
        positions = torch.cat([torch.arange(p, p + e) for p, e in rows])
        self.indexer.update_key_state_and_compress(
            token_k,
            positions,
            positions,
            metadata,
            state_stored=True,
        )
        return token_k

    def _written_slots(self) -> list:
        flat = self.pool.compressed.reshape(self.pool.compressed.shape[0], -1)
        return flat.count_nonzero(dim=1).nonzero().flatten().tolist()

    def test_partial_trailing_group_is_not_compressed(self):
        """A chunk of ratio+1..ratio+3 compresses its whole group and no more."""
        first_block = PAGE_SIZE * 4 // RATIO
        for extend_len in (RATIO + 1, RATIO + 2, RATIO + 3):
            with self.subTest(extend_len=extend_len):
                self.pool.compressed.zero_()
                token_k = self._compress_rows([(PAGE_SIZE * 4, extend_len)])
                self.assertEqual(
                    self._written_slots(), [0, _compressed_slot(0, first_block)]
                )
                torch.testing.assert_close(
                    self.pool.compressed[_compressed_slot(0, first_block)],
                    token_k[:RATIO].mean(dim=0),
                )

    def test_short_row_beside_a_long_row_stays_inert(self):
        """A row too short to fill a group must not disturb a row that fills two.

        This is the shape that hid the bug in production: the guard keys off the
        packed row count, so a lone short row beside a long one does not take
        the early return, and its padding entry has to stay harmless on its own.
        """
        token_k = self._compress_rows([(PAGE_SIZE * 4, 1), (PAGE_SIZE * 4, 2 * RATIO)])
        first_block = PAGE_SIZE * 4 // RATIO
        long_slots = [
            _compressed_slot(1, first_block),
            _compressed_slot(1, first_block + 1),
        ]
        self.assertEqual(self._written_slots(), [0] + long_slots)
        torch.testing.assert_close(
            self.pool.compressed[long_slots[0]], token_k[1 : 1 + RATIO].mean(dim=0)
        )
        torch.testing.assert_close(
            self.pool.compressed[long_slots[1]],
            token_k[1 + RATIO : 1 + 2 * RATIO].mean(dim=0),
        )

    def test_many_short_rows_write_only_the_padding_slot(self):
        """Enough short rows to clear the guard still complete no group."""
        self._compress_rows([(PAGE_SIZE * 4, 1)] * RATIO)
        self.assertEqual(self._written_slots(), [0])

    def test_cold_start_short_chunk_writes_nothing(self):
        """The guard is about chunk length, not about having a cached prefix."""
        for extend_len in range(1, RATIO):
            with self.subTest(extend_len=extend_len):
                self.pool.compressed.zero_()
                self._compress_rows([(0, extend_len)])
                self.assertEqual(self._written_slots(), [])

    def test_cold_start_compresses_every_whole_group(self):
        token_k = self._compress_rows([(0, 2 * RATIO)])
        slots = [_compressed_slot(0, 0), _compressed_slot(0, 1)]
        self.assertEqual(self._written_slots(), [0] + slots)
        torch.testing.assert_close(
            self.pool.compressed[slots[0]], token_k[:RATIO].mean(dim=0)
        )
        torch.testing.assert_close(
            self.pool.compressed[slots[1]], token_k[RATIO : 2 * RATIO].mean(dim=0)
        )


if __name__ == "__main__":
    unittest.main()
