"""The DCP owner rule partitions the sequence, and the write path obeys it.

[Test Category] Memory
[Test Target] NPUMLATokenToKVPool._resolve_dcp_write / set_index_k_buffer

Under decode context parallelism the allocator hands every rank the same
*virtual* locations, spanning the whole sequence. Two different things then
happen to them, and the pool has to keep them apart:

  latent KV   sharded    keep `loc % c == rank`, then store at `loc // c`
  index-K     replicated store at the raw `loc`, no filter, no divide

On CUDA both halves of the latent-KV rule live inside a Triton kernel
(kernels/ops/kvcache/mla_buffer.py:42). On NPU the store is a fused vendor
operator, so they run as tensor ops beforehand -- which is exactly the seam
where an owner filter can end up applied without its matching divide, a bug
that looks like sharding is working until something is read back.

These tests pin the partition property itself (every position owned once, by
one rank, at the physical row its owner expects) rather than any one rank's
behaviour. Simulating c ranks on one device is the point: the invariant is a
statement about the ranks *together*.

One NPU, no weights, no server.
"""

import unittest

import torch

from sglang.srt.layers.dcp.layout import get_dcp_lens
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=90, suite="full-1-npu-a3")

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128
PAGE_SIZE = 128
LAYER_NUM = 4
SIZE = 2048
DCP_SIZE = 4
DEVICE = "npu:0"


class _FakeLayer:
    """set_kv_buffer reads only layer_id off the layer it is handed."""

    def __init__(self, layer_id: int):
        self.layer_id = layer_id


def _build(**overrides):
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool

    kwargs = dict(
        size=SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=LAYER_NUM,
        device=DEVICE,
        enable_memory_saver=False,
        index_head_dim=INDEX_HEAD_DIM,
        start_layer=0,
        end_layer=LAYER_NUM,
    )
    kwargs.update(overrides)
    return NPUMLATokenToKVPool(**kwargs)


class TestNpuMlaDcpShardedWrite(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)

    def tearDown(self):
        torch.npu.empty_cache()

    def _resolve(self, pool, loc, rank, dcp_size=DCP_SIZE):
        """Run the write resolution as `rank` would see it."""
        k = torch.zeros(
            (loc.numel(), 1, KV_LORA_RANK), dtype=torch.bfloat16, device=DEVICE
        )
        v = torch.zeros(
            (loc.numel(), 1, QK_ROPE_HEAD_DIM), dtype=torch.bfloat16, device=DEVICE
        )
        with get_parallel().override(attn_dcp_size=dcp_size, attn_dcp_rank=rank):
            local_loc, _, _ = pool._resolve_dcp_write(loc, k, v)
        return local_loc

    def test_every_position_has_exactly_one_owner(self):
        pool = _build()
        # Real locs start at one widened page: free_pages is seeded from 1.
        first = PAGE_SIZE * DCP_SIZE
        loc = torch.arange(first, first + 1024, dtype=torch.int64, device=DEVICE)

        claimed = []
        for rank in range(DCP_SIZE):
            with get_parallel().override(attn_dcp_size=DCP_SIZE, attn_dcp_rank=rank):
                owned = (loc % DCP_SIZE) == get_parallel().attn_dcp_rank
            claimed.append(loc[owned])

        all_claimed = torch.cat(claimed).sort().values
        self.assertEqual(all_claimed.numel(), loc.numel(), "positions lost or doubled")
        self.assertTrue(torch.equal(all_claimed, loc), "the union is not the sequence")

    def test_the_resolved_shape_never_depends_on_the_data(self):
        """Why this resolves by redirect instead of by selection.

        A boolean index has a data-dependent output shape, so torch_npu answers
        it with aclnnNonzeroV2 and synchronizes the stream to learn how many
        rows survived -- which a captured stream refuses outright. Decode graph
        capture died here on every rank. The shape being a pure function of the
        input shape is therefore a hard requirement, not a nicety, and it is the
        one property no accuracy assertion below would notice the loss of.
        """
        pool = _build()
        first = PAGE_SIZE * DCP_SIZE
        for seq_len in (1, 7, 128, 1000, 1024):
            loc = torch.arange(first, first + seq_len, dtype=torch.int64, device=DEVICE)
            for rank in range(DCP_SIZE):
                with self.subTest(seq_len=seq_len, rank=rank):
                    self.assertEqual(self._resolve(pool, loc, rank).shape, loc.shape)

    def test_per_rank_ownership_matches_get_dcp_lens(self):
        """The pool's owner rule and the layout module's closed form must agree,
        or the attention path will size a page table the storage cannot fill.

        Counted as destinations that are not the padding row rather than as
        surviving rows: the resolution keeps every row now and aims the
        non-owned ones at row 0. The fixture starts at the first real loc, so
        no owned position can legitimately resolve to 0 and the count is exact.
        """
        pool = _build()
        first = PAGE_SIZE * DCP_SIZE
        for seq_len in (1, 7, 128, 1000, 1024):
            loc = torch.arange(first, first + seq_len, dtype=torch.int64, device=DEVICE)
            lens = torch.tensor([seq_len], device=DEVICE)
            for rank in range(DCP_SIZE):
                with self.subTest(seq_len=seq_len, rank=rank):
                    local = self._resolve(pool, loc, rank)
                    kept = int((local != 0).sum().item())
                    # get_dcp_lens counts positions 0..seq_len-1; the fixture is
                    # offset by a whole number of widened pages, which preserves
                    # every position's owner.
                    expected = get_dcp_lens(lens, DCP_SIZE, rank).item()
                    self.assertEqual(kept, expected)

    def test_a_virtual_page_maps_onto_the_physical_page_of_the_same_index(self):
        """Why the pool keeps an unscaled page size while the allocator widens
        its own: rank r's members of virtual page k are exactly physical page k.
        If this ever stops holding, the radix cache and the pool disagree."""
        pool = _build()
        for page in (1, 2, 15):
            base = page * PAGE_SIZE * DCP_SIZE
            loc = torch.arange(
                base, base + PAGE_SIZE * DCP_SIZE, dtype=torch.int64, device=DEVICE
            )
            for rank in range(DCP_SIZE):
                with self.subTest(page=page, rank=rank):
                    local = self._resolve(pool, loc, rank)
                    owned = (loc % DCP_SIZE) == rank
                    mine = local[owned]
                    self.assertEqual(mine.numel(), PAGE_SIZE)
                    self.assertEqual(mine.min().item(), page * PAGE_SIZE)
                    self.assertEqual(mine.max().item(), (page + 1) * PAGE_SIZE - 1)

    def test_every_non_owned_row_is_aimed_at_the_reserved_padding_row(self):
        """Physical page 0 is the padding page on every rank: the allocator
        never issues a virtual loc below one widened page, so nothing real can
        divide down into it. That is what makes row 0 a safe destination for the
        rows this rank does not own -- exactly what CUDA reserves it for with
        `reserved_skip_index`. Both halves are asserted, because sending a real
        token there and leaving a non-owned row pointing somewhere real are
        opposite failures with the same silent shape: plausible data in the
        wrong place."""
        pool = _build()
        first = PAGE_SIZE * DCP_SIZE
        loc = torch.arange(first, first + 512, dtype=torch.int64, device=DEVICE)

        for rank in range(DCP_SIZE):
            with self.subTest(rank=rank):
                local = self._resolve(pool, loc, rank)
                owned = (loc % DCP_SIZE) == rank
                self.assertGreaterEqual(local[owned].min().item(), PAGE_SIZE)
                self.assertTrue(torch.all(local[~owned] == 0))

    def test_padding_at_virtual_zero_stays_harmless(self):
        """A padding write lands on the padding row on every rank -- kept there
        by rank 0 because that is where it divides to, redirected there by the
        others because they do not own it. Never anywhere else."""
        pool = _build()
        loc = torch.zeros(4, dtype=torch.int64, device=DEVICE)

        for rank in range(DCP_SIZE):
            with self.subTest(rank=rank):
                local = self._resolve(pool, loc, rank)
                self.assertEqual(local.numel(), loc.numel())
                self.assertTrue(torch.all(local == 0))

    def test_the_write_round_trips_to_the_owning_rank_only(self):
        """The end-to-end shape of the bug this guards: an owner filter without
        its divide would still write plausible-looking data, c times too high."""
        first = PAGE_SIZE * DCP_SIZE
        n = 64
        loc = torch.arange(first, first + n, dtype=torch.int64, device=DEVICE)
        # Each virtual position carries its own value, so a misplaced row is
        # identifiable rather than merely wrong. Kept under 128 so every marker
        # is exact in bfloat16, which carries only 8 bits of significand.
        values = (loc % 100).to(torch.bfloat16)

        for rank in range(DCP_SIZE):
            pool = _build()
            cache_k = values.view(n, 1, 1).expand(n, 1, KV_LORA_RANK).contiguous()
            cache_v = values.view(n, 1, 1).expand(n, 1, QK_ROPE_HEAD_DIM).contiguous()
            with get_parallel().override(attn_dcp_size=DCP_SIZE, attn_dcp_rank=rank):
                pool.set_kv_buffer(_FakeLayer(0), loc, cache_k.clone(), cache_v.clone())

            k_rows = pool.get_key_buffer(0).view(-1, 1, KV_LORA_RANK)
            for i in range(n):
                virtual = first + i
                if virtual % DCP_SIZE != rank:
                    continue
                with self.subTest(rank=rank, virtual=virtual):
                    got = k_rows[virtual // DCP_SIZE, 0, 0].item()
                    self.assertEqual(got, float(virtual % 100))
            del pool
            torch.npu.empty_cache()

    def test_the_indexer_is_not_translated(self):
        """The replicated half: index-K addresses global positions raw. If the
        latent-KV rule ever leaks into this path, the indexer loses c-1 of every
        c positions it is supposed to see."""
        pool = _build(index_buf_size=SIZE * DCP_SIZE)
        virtual = SIZE * DCP_SIZE - 3
        loc = torch.tensor([virtual], dtype=torch.int32, device=DEVICE)
        value = torch.full(
            (1, INDEX_HEAD_DIM), 7.0, dtype=torch.bfloat16, device=DEVICE
        )

        # Rank 3 does not own this position under the latent-KV rule, and must
        # write it anyway.
        with get_parallel().override(attn_dcp_size=DCP_SIZE, attn_dcp_rank=3):
            pool.set_index_k_buffer(0, loc, value)

        buffer = pool.get_index_k_buffer(0).view(-1, 1, INDEX_HEAD_DIM)
        self.assertEqual(buffer[virtual, 0, 0].item(), 7.0)

    def test_dcp_size_one_touches_nothing(self):
        pool = _build()
        loc = torch.arange(0, 32, dtype=torch.int64, device=DEVICE)

        local = self._resolve(pool, loc, rank=0, dcp_size=1)

        self.assertTrue(torch.equal(local, loc))


if __name__ == "__main__":
    unittest.main()
