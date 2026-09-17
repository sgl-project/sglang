"""Unit tests for host-pool allocation and free-list bookkeeping."""

import threading
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import MambaPool, MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.ple_state_pool import (
    PLE_NGRAM_STATE_LAYER_ID,
    NGramPool,
    ShortConvPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry, base
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.ple import (
    PleStatePoolHost,
    collect_ple_state_regions,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=17, suite="base-a-test-cpu")


class TestHostKVCache(CustomTestCase):
    def setUp(self):
        self.page_size = 2
        # Small device pool is enough to construct the host pool.
        self.device_pool = MHATokenToKVPool(
            size=self.page_size * 2,
            page_size=self.page_size,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

    def test_double_alloc(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        # Mimic bookkeeping corruption: push an already-used slot back to the
        # head of free_slots so the next alloc would hand out an in-use slot.
        leak = torch.tensor([int(indices[0])])
        self.host_pool.free_slots = torch.cat([leak, self.host_pool.free_slots])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.alloc(4)
        msg = str(ctx.exception)
        self.assertIn("Double-alloc", msg)
        self.assertIn(f"[{int(leak[0])}]", msg)

    def test_double_free(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        self.host_pool.free(indices[:2])
        # indices[1] is double freed.
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices[1:])
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[1])}]", msg)

    def test_free_unallocated(self):
        indices = torch.tensor([1])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[0])}]", msg)

    def test_free_after_clear(self):
        indices = self.host_pool.alloc(4)
        self.host_pool.clear()
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(str(indices.tolist()), msg)

    def test_shm_allocator(self):
        shm_host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="shm",
        )
        self.assertIsNotNone(shm_host_pool.fd)
        self.assertGreaterEqual(shm_host_pool.fd, 0)

        indices = shm_host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        shm_host_pool.free(indices)

    def test_empty_free_keeps_release_list_empty(self):
        self.assertEqual(self.host_pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(self.host_pool.num_release_slots, 0)
        self.assertEqual(self.host_pool.release_slots, [])


class TestLazyHostPoolRelease(CustomTestCase):
    @staticmethod
    def _make_mamba_pool():
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_deepseek_v4_pool():
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.size = 8
        pool.slot_page_size = 2
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_logical_pool():
        return LogicalHostPool(size=8, page_size=2)

    @staticmethod
    def _make_transfer_pool(*, page_aligned_only):
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.pool_name = str(PoolName.DEEPSEEK_V4_C4_INDEXER)
        pool.slot_page_size = 4
        pool.layer_num = 1
        pool.page_aligned_only = page_aligned_only
        pool.device_ptrs = [0]
        pool.data_ptrs = [0]
        return pool

    def _assert_lazy_release(self, pool):
        self.assertEqual(pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

        allocated = pool.alloc(6)
        free_slots_before = pool.free_slots

        pool.free(allocated[:2])

        # free() should keep the primary free-list untouched and only record
        # the released chunk for a later merge.
        self.assertIs(pool.free_slots, free_slots_before)
        self.assertEqual(pool.num_release_slots, 2)
        self.assertEqual(len(pool.release_slots), 1)
        self.assertEqual(pool.available_size(), 4)

        # Consume the primary free-list first without merging pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([6, 7])))
        self.assertEqual(pool.num_release_slots, 2)

        # Once the primary free-list is exhausted, alloc() merges and reuses
        # the pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([0, 1])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 0)

        pool.free(torch.tensor([0, 1]))
        pool.clear()
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 8)

        # Exercise the general merge path with multiple released chunks.
        allocated = pool.alloc(8)
        pool.free(allocated[:2])
        pool.free(allocated[2:4])
        self.assertEqual(len(pool.release_slots), 2)
        self.assertTrue(torch.equal(pool.alloc(4), torch.tensor([0, 1, 2, 3])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

    def test_mamba_pool_lazy_release(self):
        self._assert_lazy_release(self._make_mamba_pool())

    def test_deepseek_v4_pool_lazy_release(self):
        pool = self._make_deepseek_v4_pool()
        self._assert_lazy_release(pool)

        # Preserve the pool's page-aligned allocation behavior.
        pool.clear()
        self.assertEqual(len(pool.alloc(1)), 2)

    def test_grouped_page_rows_reject_unaligned_transfers(self):
        # FP4 indexer rows group their slots, so a partial page has no
        # well-defined token-granular copy and must not silently fall back.
        pool = self._make_transfer_pool(page_aligned_only=True)
        unaligned = torch.arange(3, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            pool.backup_from_device_all_layer(None, unaligned, unaligned, "direct")
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            pool.load_to_device_per_layer(None, unaligned, unaligned, 0, "direct")

    def test_fused_page_rows_keep_token_granular_transfers(self):
        pool = self._make_transfer_pool(page_aligned_only=False)
        unaligned = torch.arange(3, dtype=torch.int64)
        with unittest.mock.patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_cache_dsv4_mla"
        ) as transfer:
            pool.backup_from_device_all_layer(None, unaligned, unaligned, "direct")
            pool.load_to_device_per_layer(None, unaligned, unaligned, 0, "direct")
        self.assertEqual(transfer.call_count, 2)

    def test_logical_pool_lazy_release(self):
        pool = self._make_logical_pool()
        self._assert_lazy_release(pool)

        # Preserve the logical pool's strict page-alignment checks.
        pool.clear()
        with self.assertRaises(ValueError):
            pool.alloc(1)
        with self.assertRaises(ValueError):
            pool.free(torch.tensor([0]))


class TestHostMemoryBudget(CustomTestCase):
    # Pinned so the two budget reads below see identical free memory; the real
    # psutil value drifts between calls and would flake the equality checks.
    _AVAILABLE = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3)

    def _budget_with_ranks(self, ranks):
        # Deliberate single-accessor stub: isolates the budget math from the
        # topology derivation, which the ranks_per_host case below covers.
        fake_mem = unittest.mock.Mock(available=self._AVAILABLE)
        with (
            unittest.mock.patch.object(base, "ranks_per_host", return_value=ranks),
            unittest.mock.patch.object(
                base.psutil, "virtual_memory", return_value=fake_mem
            ),
        ):
            return base.host_memory_budget_bytes()

    def test_budget_is_split_across_co_located_ranks(self):
        solo = self._budget_with_ranks(1)
        self.assertEqual(self._budget_with_ranks(4), solo // 4)

    def test_reserve_is_taken_before_the_split(self):
        # Each rank must not get its own copy of the reserve.
        budget = self._budget_with_ranks(8)
        self.assertLessEqual(
            budget * 8, self._AVAILABLE - base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        )

    def test_ranks_per_host_divides_world_size_by_nodes(self):
        # The launcher slices ranks uniformly across nodes, so the co-located
        # rank count is world_size // nnodes — no hostname collective.
        fake_group = unittest.mock.Mock(world_size=16)
        with (
            get_context().override_server_args(nnodes=2),
            unittest.mock.patch.object(
                torch.distributed, "is_initialized", return_value=True
            ),
            unittest.mock.patch.object(
                base, "get_world_group", return_value=fake_group
            ),
        ):
            self.assertEqual(base.ranks_per_host(), 8)


class TestHostPoolGroup(CustomTestCase):
    @staticmethod
    def _group(**sizes):
        return HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName(name),
                    host_pool=LogicalHostPool(size=size, page_size=1),
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=name == PoolName.KV.value,
                )
                for name, size in sizes.items()
            ]
        )

    def test_resolve_and_release_multi_pool_allocation(self):
        group = self._group(kv=4, swa=2)
        primary = group.alloc(2)
        transfers = [
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2)),
            PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.SWA),
        ]

        self.assertIsNotNone(
            group.resolve_host_transfers(
                transfers,
                primary_device_indices=torch.arange(2),
                primary_host_indices=primary,
            )
        )
        self.assertIs(transfers[1].host_indices, transfers[0].host_indices)
        group.free(primary)
        group.release_transfers(transfers)
        self.assertEqual(group.available_size(), 4)
        self.assertEqual(group.available_size(PoolName.SWA), 2)

    def test_resolve_rolls_back_partial_allocation(self):
        group = self._group(kv=4, swa=2, mamba=1)
        transfers = [
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2)),
            PoolTransfer(name=PoolName.MAMBA, device_indices=torch.arange(2)),
        ]

        self.assertIsNone(group.resolve_host_transfers(transfers))
        self.assertIsNone(transfers[0].host_indices)
        self.assertEqual(group.available_size(PoolName.SWA), 2)


# ===== Qwen4-Exp PLE state host pool =====

PLE_SLOTS = 8
PLE_HOST_SLOTS = 16
PLE_SHORT_CONV_LAYER_IDS = [1, 3]
PLE_CHANNELS = 6
PLE_STATE_LEN = 4
PLE_NGRAM_CONTEXT_LEN = 2

PLE_SHORT_CONV_SLOT_BYTES = PLE_CHANNELS * PLE_STATE_LEN * torch.bfloat16.itemsize
PLE_NGRAM_SLOT_BYTES = PLE_NGRAM_CONTEXT_LEN * torch.int64.itemsize


def _mamba_pool_with_ple(*, enable_ple: bool = True, spec_draft_tokens=None):
    """A MambaPool carrying only what the PLE host pool reads.

    `__new__` skips the mamba_cache allocation and keeps the real
    `register_slot_state` bookkeeping and sibling pools.
    """
    pool = MambaPool.__new__(MambaPool)
    pool.size = PLE_SLOTS
    pool.device = "cpu"
    if not enable_ple:
        return pool
    pool.register_slot_state(
        ShortConvPool(
            size=PLE_SLOTS,
            state_shape=(PLE_CHANNELS, PLE_STATE_LEN),
            layer_ids=PLE_SHORT_CONV_LAYER_IDS,
            dtype=torch.bfloat16,
            device="cpu",
            spec_state_size=PLE_SLOTS if spec_draft_tokens else 0,
            speculative_num_draft_tokens=spec_draft_tokens,
        )
    )
    pool.register_slot_state(
        NGramPool(
            size=PLE_SLOTS,
            context_len=PLE_NGRAM_CONTEXT_LEN,
            eos_token_id=0,
            device="cpu",
        )
    )
    return pool


def _ple_anchor_host(size: int = PLE_HOST_SLOTS) -> MambaPoolHost:
    anchor = MambaPoolHost.__new__(MambaPoolHost)
    anchor.size = size
    anchor.page_size = 1
    anchor.device = "cpu"
    anchor.lock = threading.RLock()
    anchor.clear()
    return anchor


def _ple_host(layout: str = "page_first", anchor_size: int = PLE_HOST_SLOTS):
    return PleStatePoolHost(
        _mamba_pool_with_ple(),
        _ple_anchor_host(anchor_size),
        layout=layout,
        pin_memory=False,
        device="cpu",
    )


class TestPleStateRegions(CustomTestCase):
    def test_slot_axis_is_normalized_to_dim_zero(self):
        """Every transfer assumes an entry is [slots, *state_shape]; short-conv
        keeps slots at dim 1 and N-gram has no layer dim at all."""
        for region in collect_ple_state_regions(_mamba_pool_with_ple()):
            for tensor in region.device_tensors:
                self.assertEqual(tensor.shape[0], PLE_SLOTS + 1)
                self.assertEqual(tuple(tensor.shape[1:]), region.state_shape)

    def test_regions_group_by_field_dtype_and_shape(self):
        regions = collect_ple_state_regions(_mamba_pool_with_ple())

        # Field names are the PD transfer protocol's, not this module's.
        self.assertEqual([r.field for r in regions], ["ple_short_conv", "ple_ngram"])
        short_conv, ngram = regions
        self.assertEqual(short_conv.dtype, torch.bfloat16)
        self.assertEqual(short_conv.state_shape, (PLE_CHANNELS, PLE_STATE_LEN))
        self.assertEqual(short_conv.layer_ids, PLE_SHORT_CONV_LAYER_IDS)
        self.assertEqual(ngram.dtype, torch.int64)
        self.assertEqual(ngram.state_shape, (PLE_NGRAM_CONTEXT_LEN,))
        self.assertEqual(ngram.layer_ids, [PLE_NGRAM_STATE_LAYER_ID])

    def test_intermediate_spec_scratch_is_excluded(self):
        """`intermediate_*` is per-draft-token scratch; mirroring it would
        restore a mid-verify window."""
        regions = collect_ple_state_regions(_mamba_pool_with_ple(spec_draft_tokens=3))

        short_conv = regions[0]
        self.assertEqual(len(short_conv.device_tensors), len(PLE_SHORT_CONV_LAYER_IDS))
        for tensor in short_conv.device_tensors:
            self.assertEqual(tuple(tensor.shape[1:]), (PLE_CHANNELS, PLE_STATE_LEN))

    def test_disabled_ple_yields_no_regions(self):
        """A hybrid model without PLE still builds both sibling pools, disabled;
        the assembler skips the PLE host pool on this."""
        self.assertEqual(
            collect_ple_state_regions(_mamba_pool_with_ple(enable_ple=False)), []
        )


class TestPleStatePoolHost(CustomTestCase):
    def test_slots_mirror_the_anchor_pool(self):
        """Transfers arrive with MAMBA's host indices, so sizing this pool
        independently would index out of range."""
        for anchor_size in (PLE_HOST_SLOTS, PLE_HOST_SLOTS * 3):
            with self.subTest(anchor_size=anchor_size):
                host = _ple_host(anchor_size=anchor_size)
                self.assertEqual(host.size, anchor_size)
                for buffer in host.region_buffers:
                    self.assertEqual(buffer.shape[0], anchor_size)

    def test_page_first_buffer_geometry(self):
        """One slot's whole region stays contiguous, giving the
        `slot_bytes * entries` row stride the transfer helpers assume."""
        expected = {
            "page_first": (
                (PLE_HOST_SLOTS, 2, PLE_CHANNELS, PLE_STATE_LEN),
                (PLE_HOST_SLOTS, 1, PLE_NGRAM_CONTEXT_LEN),
            ),
            "page_first_direct": (
                (PLE_HOST_SLOTS, 2, 1, PLE_CHANNELS, PLE_STATE_LEN),
                (PLE_HOST_SLOTS, 1, 1, PLE_NGRAM_CONTEXT_LEN),
            ),
        }
        for layout, shapes in expected.items():
            with self.subTest(layout=layout):
                host = _ple_host(layout)
                self.assertEqual(
                    [tuple(b.shape) for b in host.region_buffers], list(shapes)
                )

    def test_layer_first_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "page-first layout"):
            _ple_host("layer_first")

    def test_pool_without_ple_state_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no slot-sibling"):
            PleStatePoolHost(
                _mamba_pool_with_ple(enable_ple=False),
                _ple_anchor_host(),
                layout="page_first",
                pin_memory=False,
                device="cpu",
            )

    def test_size_per_token_matches_allocated_bytes(self):
        """HiCache budgets host pools as `size * size_per_token`, so a
        per-token cost that counts an entry twice over-reserves silently."""
        host = _ple_host()
        expected = (
            PLE_SHORT_CONV_SLOT_BYTES * len(PLE_SHORT_CONV_LAYER_IDS)
            + PLE_NGRAM_SLOT_BYTES
        )

        self.assertEqual(host.size_per_token, expected)
        self.assertEqual(
            sum(b.numel() * b.element_size() for b in host.region_buffers),
            expected * PLE_HOST_SLOTS,
        )

    def test_pool_is_a_single_transfer_unit(self):
        """The per-layer load loop resolves a global layer to one local index,
        and the N-gram context belongs to no model layer."""
        host = _ple_host()

        self.assertEqual(host.layer_num, 1)
        with self.assertRaisesRegex(ValueError, "single transfer unit"):
            host.load_to_device_per_layer(None, torch.tensor([0]), torch.tensor([0]), 1)

    def test_model_layer_ids_exclude_the_ngram_sentinel(self):
        """The assembler anchors the layer_mapping at min(model_layer_ids), and
        the sentinel is outside the transfer loop's range."""
        host = _ple_host()

        self.assertEqual(host.model_layer_ids, PLE_SHORT_CONV_LAYER_IDS)
        self.assertNotIn(PLE_NGRAM_STATE_LAYER_ID, host.model_layer_ids)

    def test_page_buffer_meta_strides_by_region_row(self):
        host = _ple_host()
        row_bytes = [
            PLE_SHORT_CONV_SLOT_BYTES * len(PLE_SHORT_CONV_LAYER_IDS),
            PLE_NGRAM_SLOT_BYTES,
        ]
        bases = [buffer.data_ptr() for buffer in host.region_buffers]

        ptrs, sizes = host.get_page_buffer_meta(torch.tensor([0, 2]))

        # Region-major within each page, matching get_data_page's byte order.
        self.assertEqual(
            ptrs,
            [
                bases[0],
                bases[1],
                bases[0] + 2 * row_bytes[0],
                bases[1] + 2 * row_bytes[1],
            ],
        )
        self.assertEqual(sizes, row_bytes * 2)

    def test_data_page_round_trips_through_flat_bytes(self):
        """The L3 serialization pair must agree on region order and per-region
        byte counts."""
        host = _ple_host()
        short_conv, ngram = host.region_buffers
        conv_value = torch.arange(short_conv[3].numel(), dtype=torch.bfloat16).reshape(
            short_conv[3].shape
        )
        ngram_value = torch.tensor([[7, 9]], dtype=torch.int64)
        short_conv[3].copy_(conv_value)
        ngram[3].copy_(ngram_value)

        page = host.get_data_page(3)
        self.assertEqual(page.numel(), host.size_per_token)

        short_conv[3].zero_()
        ngram[3].zero_()
        host.set_from_flat_data_page(3, page)

        self.assertTrue(torch.equal(short_conv[3], conv_value))
        self.assertTrue(torch.equal(ngram[3], ngram_value))


if __name__ == "__main__":
    unittest.main()
