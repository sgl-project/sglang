"""Striped HiCache host tier: owned-row narrowing and backup-admission consensus."""

import os
import tempfile
import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import CacheOperation
from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import build_pool_entry
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host import HostPoolGroup
from sglang.srt.mem_cache.pool_host.striped_host import (
    StripedHostPlacement,
    StripedLogicalHostPool,
    build_striped_load_plan,
    owned_rows_replicated,
    validate_striped_host_group,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE = 256
# DSv4-Pro per-page row bytes and layer counts (MI355X).
C4_LAYERS = 30
C128_LAYERS = 31
TRANSFER_LAYERS = C4_LAYERS + C128_LAYERS
DSV4_POOLS = (
    (PoolName.DEEPSEEK_V4_C4, 32768, "c4"),
    (PoolName.DEEPSEEK_V4_C4_INDEXER, 4096, "c4"),
    (PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE, 256, "c4"),
    (PoolName.DEEPSEEK_V4_C4_ROPE, 8192, "c4"),
    (PoolName.DEEPSEEK_V4_C128, 1024, "c128"),
    (PoolName.DEEPSEEK_V4_C128_ROPE, 256, "c128"),
)
# Interleaved compression ratios; the exact DSv4 order does not matter here.
C4_LAYER_MAP = {2 * i + 1: i for i in range(C4_LAYERS)}
C128_LAYER_MAP = {2 * i: i for i in range(C128_LAYERS)}


def _ref_transfer_kv_direct(
    src_layers, dst_layers, src_indices, dst_indices, page_size
):
    assert page_size == 1
    for src, dst in zip(src_layers, dst_layers, strict=True):
        dst[dst_indices] = src[src_indices]


def _ref_all_layer_direct_lf_pf(
    src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
):
    assert page_size == 1
    for layer_id, src in enumerate(src_ptrs):
        dst_ptrs[0][dst_indices, layer_id, 0] = src[src_indices]


def _ref_per_layer_direct_pf_lf(
    src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
):
    assert page_size == 1
    dst_ptrs[0][dst_indices] = src_ptrs[0][src_indices, layer_id, 0]


def _cpu_transfer_kernels():
    patches = [
        mock.patch.object(
            memory_pool_host, "transfer_kv_direct", _ref_transfer_kv_direct, create=True
        ),
        mock.patch.object(
            memory_pool_host,
            "transfer_kv_all_layer_direct_lf_pf",
            _ref_all_layer_direct_lf_pf,
            create=True,
        ),
        mock.patch.object(
            memory_pool_host,
            "transfer_kv_per_layer_direct_pf_lf",
            _ref_per_layer_direct_pf_lf,
            create=True,
        ),
        mock.patch.object(
            memory_pool_host, "host_memory_budget_bytes", return_value=1 << 40
        ),
    ]
    for patch in patches:
        patch.start()
    return patches


def _striped_anchor(placement, per_rank_pages, layout="layer_first"):
    return StripedLogicalHostPool(
        per_rank_size=per_rank_pages * placement.page_size,
        page_size=placement.page_size,
        placement=placement,
        layout=layout,
    )


def _dsv4_device_buffers(num_device_pages, seed):
    generator = torch.Generator().manual_seed(seed)
    buffers = {}
    for name, item_bytes, family in DSV4_POOLS:
        num_layers = C4_LAYERS if family == "c4" else C128_LAYERS
        buffers[name] = [
            torch.randint(
                0,
                256,
                (num_device_pages, item_bytes),
                dtype=torch.uint8,
                generator=generator,
            )
            for _ in range(num_layers)
        ]
    return buffers


def _dsv4_host_group(anchor, device_buffers, num_host_pages, layout):
    entries = [
        build_pool_entry(
            name=PoolName.KV,
            host_pool=anchor,
            device_pool=None,
            layer_mapping={i: i for i in range(TRANSFER_LAYERS)},
            transfer_layer_id_max=TRANSFER_LAYERS,
            is_anchor=True,
        )
    ]
    for name, item_bytes, family in DSV4_POOLS:
        host_pool = DeepSeekV4PagedHostPool(
            pool_name=str(name),
            device_buffers=device_buffers[name],
            item_bytes=item_bytes,
            num_host_pages=num_host_pages,
            slot_page_size=PAGE,
            layout=layout,
            device="cpu",
            pin_memory=False,
            page_aligned_only=True,
        )
        # Host buffers start uninitialized; zero them so unwritten rows are visible.
        for buffer in host_pool.get_hybrid_pool_buffer():
            buffer.zero_()
        entries.append(
            build_pool_entry(
                name=name,
                host_pool=host_pool,
                device_pool=None,
                layer_mapping=C4_LAYER_MAP if family == "c4" else C128_LAYER_MAP,
                transfer_layer_id_max=TRANSFER_LAYERS,
            )
        )
    return HostPoolGroup(entries)


def _controller(host_group, host_stripe, io_backend="direct"):
    controller = HybridCacheController.__new__(HybridCacheController)
    controller.mem_pool_host = host_group
    controller.io_backend = io_backend
    controller.device = "cpu"
    controller.host_stripe = host_stripe
    controller.tp_group = None
    controller.write_queue = []
    return controller


def _sidecars():
    return [
        PoolTransfer(name=name, indices_from_pool=PoolName.KV)
        for name, _, _ in DSV4_POOLS
    ]


def _op(host_indices, device_indices):
    return CacheOperation(
        host_indices,
        device_indices,
        node_id=0,
        pool_transfers=[
            PoolTransfer(
                name=name,
                host_indices=host_indices,
                device_indices=device_indices,
                indices_from_pool=PoolName.KV,
            )
            for name, _, _ in DSV4_POOLS
        ],
    )


def _run_backup(controller, op):
    host, device, transfers = controller._move_write_operation(op)
    for transfer in controller._l2_transfers(host, device, transfers):
        transfer.host_pool.backup_from_device_all_layer_physical(
            transfer.device_pool,
            transfer.host_indices,
            transfer.device_indices,
            controller.io_backend,
        )


def _host_page_row(pool, layer, page):
    if pool.layout == "layer_first":
        return pool.kv_buffer[layer][page]
    return pool.kv_buffer[page, layer].flatten()


def _random_page_op(anchor, num_pages, device_pages, seed):
    """Whole-page op: host pages from the striped allocator, shuffled device pages."""
    host_indices = anchor.alloc(num_pages * PAGE)
    generator = torch.Generator().manual_seed(seed)
    dev_pages = device_pages[torch.randperm(len(device_pages), generator=generator)]
    dev_pages = dev_pages[:num_pages]
    offsets = torch.arange(PAGE, dtype=torch.int64)
    device_indices = (dev_pages[:, None] * PAGE + offsets).flatten()
    return host_indices, device_indices


class TestStripedNarrowing(unittest.TestCase):
    def setUp(self):
        self._patches = _cpu_transfer_kernels()

    def tearDown(self):
        for patch in self._patches:
            patch.stop()

    def test_owned_rows_partition_the_op(self):
        num_ranks = 4
        anchor = _striped_anchor(
            StripedHostPlacement(num_ranks=num_ranks, rank=0, page_size=4), 8
        )
        anchor.alloc(3 * 4)
        host = anchor.alloc(13 * 4)
        device = torch.arange(1000, 1000 + host.numel(), dtype=torch.int64)
        covered = []
        for rank in range(num_ranks):
            placement = StripedHostPlacement(
                num_ranks=num_ranks, rank=rank, page_size=4
            )
            local, owned_dev = owned_rows_replicated(
                placement, host_indices=host, device_indices=device
            )
            self.assertEqual(local.numel(), owned_dev.numel())
            self.assertTrue(bool((local < 8 * 4).all()))
            # Each owned device slot still pairs with its logical host slot.
            logical = host[owned_dev - 1000]
            self.assertTrue(bool(((logical // 4) % num_ranks == rank).all()))
            self.assertTrue(torch.equal(placement.local(logical), local))
            covered.append(owned_dev)
        self.assertTrue(torch.equal(torch.sort(torch.cat(covered)).values, device))

    def test_partial_or_misaligned_pages_are_refused(self):
        placement = StripedHostPlacement(num_ranks=2, rank=0, page_size=4)
        for host in (torch.arange(6), torch.arange(2, 6), torch.tensor([0, 1, 3, 2])):
            with self.subTest(host=host.tolist()):
                with self.assertRaises(ValueError):
                    owned_rows_replicated(
                        placement, host_indices=host, device_indices=host.clone()
                    )
        # A split or misaligned device page aborts the load plan.
        host = torch.arange(8)
        for device in (torch.arange(2, 10), torch.tensor([0, 1, 3, 2, 4, 5, 6, 7])):
            with self.subTest(device=device.tolist()):
                with self.assertRaises(RuntimeError):
                    build_striped_load_plan(
                        placement, host_indices=host, device_indices=device, seq=0
                    )
        plan = build_striped_load_plan(
            placement, host_indices=host, device_indices=torch.arange(4, 12), seq=0
        )
        self.assertEqual([p.tolist() for p in plan.dev_pages_by_owner], [[1], [2]])

    def test_union_of_owned_backups_equals_stock_backup(self):
        num_ranks, per_rank_pages, device_pages = 8, 4, 40
        for layout in ("page_first_direct", "layer_first"):
            with self.subTest(layout=layout):
                device_buffers = _dsv4_device_buffers(device_pages, seed=7)
                logical = _striped_anchor(
                    StripedHostPlacement(num_ranks=num_ranks, rank=0, page_size=PAGE),
                    per_rank_pages,
                    layout=layout,
                )
                # Churn the allocator so ops mix owners unevenly.
                first = logical.alloc(5 * PAGE)
                logical.alloc(3 * PAGE)
                logical.free(first)
                ops = [
                    _random_page_op(logical, 9, torch.arange(device_pages), seed=1),
                    _random_page_op(logical, 1, torch.arange(device_pages), seed=2),
                    _random_page_op(logical, 6, torch.arange(device_pages), seed=3),
                ]

                stock_group = _dsv4_host_group(
                    logical, device_buffers, num_ranks * per_rank_pages, layout
                )
                stock = _controller(stock_group, host_stripe=None)
                for host, device in ops:
                    _run_backup(stock, _op(host, device))

                rank_groups = []
                for rank in range(num_ranks):
                    placement = StripedHostPlacement(
                        num_ranks=num_ranks, rank=rank, page_size=PAGE
                    )
                    anchor = _striped_anchor(placement, per_rank_pages, layout=layout)
                    group = _dsv4_host_group(
                        anchor, device_buffers, per_rank_pages, layout
                    )
                    validate_striped_host_group(group, placement)
                    controller = _controller(group, host_stripe=placement)
                    # Deferred backups reach start_writing as one merged op.
                    _run_backup(
                        controller,
                        CacheOperation.merge_ops([_op(h, d) for h, d in ops]),
                    )
                    rank_groups.append(group)

                self._assert_union_equals_stock(
                    stock_group, rank_groups, ops, num_ranks, per_rank_pages
                )

    def _assert_union_equals_stock(
        self, stock_group, rank_groups, ops, num_ranks, per_rank_pages
    ):
        backed_pages = {int(page) for host, _ in ops for page in host[::PAGE] // PAGE}
        for name, item_bytes, _ in DSV4_POOLS:
            stock_pool = stock_group.get_pool(name)
            for layer in range(stock_pool.layer_num):
                for page in range(num_ranks * per_rank_pages):
                    owner, local = page % num_ranks, page // num_ranks
                    striped = _host_page_row(
                        rank_groups[owner].get_pool(name), layer, local
                    )
                    self.assertEqual(striped.numel(), item_bytes)
                    self.assertTrue(
                        torch.equal(striped, _host_page_row(stock_pool, layer, page)),
                        (name, layer, page),
                    )
                    if page not in backed_pages:
                        self.assertFalse(bool(striped.any()), (name, layer, page))

    def test_validation_refuses_non_striped_groups(self):
        layout = "layer_first"
        device_buffers = _dsv4_device_buffers(4, seed=0)
        placement = StripedHostPlacement(num_ranks=2, rank=0, page_size=PAGE)
        anchor = _striped_anchor(placement, 4, layout=layout)
        with self.assertRaises(ValueError):
            validate_striped_host_group(
                _dsv4_host_group(anchor, device_buffers, 8, layout), placement
            )
        with self.assertRaises(ValueError):
            validate_striped_host_group(
                _dsv4_host_group(anchor, device_buffers, 4, layout),
                StripedHostPlacement(num_ranks=2, rank=1, page_size=PAGE),
            )
        group = _dsv4_host_group(anchor, device_buffers, 4, layout)
        validate_striped_host_group(group, placement)
        group.entries[1].host_pool.page_aligned_only = False
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            validate_striped_host_group(group, placement)

    def test_merged_op_is_narrowed_once_on_every_routing(self):
        num_ranks, rank, device_pages = 4, 1, 16
        device_buffers = _dsv4_device_buffers(device_pages, seed=1)
        for io_backend, layout in (
            ("kernel", "page_first"),
            ("direct", "page_first_direct"),
            ("direct", "layer_first"),
        ):
            with self.subTest(io_backend=io_backend, layout=layout):
                placement = StripedHostPlacement(
                    num_ranks=num_ranks, rank=rank, page_size=PAGE
                )
                anchor = _striped_anchor(placement, 4, layout=layout)
                group = _dsv4_host_group(anchor, device_buffers, 4, layout)
                controller = _controller(group, placement, io_backend=io_backend)
                merged = CacheOperation.merge_ops(
                    [
                        _op(*_random_page_op(anchor, k, torch.arange(16), seed=k))
                        for k in (3, 2, 4)
                    ]
                )
                expected_host, expected_dev = owned_rows_replicated(
                    placement,
                    host_indices=merged.host_indices,
                    device_indices=merged.device_indices,
                )
                self.assertGreater(expected_host.numel(), 0)
                with mock.patch.object(
                    hybrid_cache_controller,
                    "owned_rows_replicated",
                    wraps=owned_rows_replicated,
                ) as narrow:
                    host, device, transfers = controller._move_write_operation(merged)
                self.assertEqual(narrow.call_count, 1)
                # The data-less anchor keeps its logical ids.
                self.assertTrue(torch.equal(host, merged.host_indices))
                self.assertTrue(torch.equal(device, merged.device_indices))
                self.assertEqual(len(transfers), len(DSV4_POOLS))
                for transfer in transfers:
                    self.assertTrue(torch.equal(transfer.host_indices, expected_host))
                    self.assertTrue(torch.equal(transfer.device_indices, expected_dev))
                    self.assertTrue(bool((transfer.host_indices < 4 * PAGE).all()))

    def test_merge_shares_op_indices_only_for_riding_transfers(self):
        ops = [
            _op(torch.arange(4 * i, 4 * i + 4), torch.arange(40 + 4 * i, 44 + 4 * i))
            for i in range(3)
        ]
        own = torch.arange(100, 104)
        ops[1].pool_transfers.append(
            PoolTransfer(name=PoolName.SWA, host_indices=own, device_indices=own)
        )
        merged = CacheOperation.merge_ops(ops)
        riding, swa = merged.pool_transfers[0], merged.pool_transfers[-1]
        self.assertIs(riding.host_indices, merged.host_indices)
        self.assertIs(riding.device_indices, merged.device_indices)
        self.assertTrue(torch.equal(swa.host_indices, own))
        # A transfer with its own indices is concatenated, never aliased.
        ops[2].pool_transfers[0].host_indices = ops[2].host_indices.clone()
        merged = CacheOperation.merge_ops(ops)
        self.assertIsNot(merged.pool_transfers[0].host_indices, merged.host_indices)
        self.assertTrue(
            torch.equal(merged.pool_transfers[0].host_indices, merged.host_indices)
        )

    def test_striped_load_fails_closed(self):
        placement = StripedHostPlacement(num_ranks=2, rank=0, page_size=PAGE)
        anchor = _striped_anchor(placement, 4)
        group = _dsv4_host_group(
            anchor, _dsv4_device_buffers(4, seed=0), 4, "layer_first"
        )
        controller = _controller(group, placement)
        controller.load_queue = []
        controller.mem_pool_device_allocator = mock.MagicMock()
        with self.assertRaises(NotImplementedError):
            controller.load(torch.arange(PAGE), extra_pools=_sidecars())
        self.assertEqual(controller.load_queue, [])
        controller.mem_pool_device_allocator.alloc.assert_not_called()


class TestStripedConfigurationRefusals(unittest.TestCase):
    def setUp(self):
        self._patches = _cpu_transfer_kernels()
        self.placement = StripedHostPlacement(num_ranks=2, rank=1, page_size=PAGE)
        anchor = _striped_anchor(self.placement, 4)
        self.group = _dsv4_host_group(
            anchor, _dsv4_device_buffers(4, seed=0), 4, "layer_first"
        )

    def tearDown(self):
        for patch in self._patches:
            patch.stop()

    def _construct(self, world_size, rank, **kwargs):
        with (
            mock.patch.object(
                torch.distributed, "get_world_size", return_value=world_size
            ),
            mock.patch.object(torch.distributed, "get_rank", return_value=rank),
        ):
            return HybridCacheController(
                token_to_kv_pool_allocator=mock.MagicMock(),
                mem_pool_host=self.group,
                page_size=PAGE,
                tp_group=mock.sentinel.tp_group,
                load_cache_event=mock.MagicMock(),
                host_stripe=self.placement,
                **kwargs,
            )

    def test_constructor_refuses_placement_off_the_tp_group(self):
        for world_size, rank in ((4, 1), (2, 0)):
            with self.subTest(world_size=world_size, rank=rank):
                with self.assertRaisesRegex(ValueError, "TP group"):
                    self._construct(world_size, rank)

    def test_transfer_off_the_anchor_is_refused(self):
        controller = _controller(self.group, self.placement)
        for transfer in (
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4),
            PoolTransfer(name=PoolName.DEEPSEEK_V4_C4, indices_from_pool=PoolName.SWA),
        ):
            with self.subTest(indices_from_pool=transfer.indices_from_pool):
                with (
                    mock.patch.object(torch.distributed, "all_reduce") as all_reduce,
                    self.assertRaisesRegex(ValueError, "ride"),
                ):
                    controller.write(
                        torch.arange(PAGE), extra_pools=[transfer], flush=False
                    )
                all_reduce.assert_not_called()
        self.assertEqual(controller.write_queue, [])
        self.assertEqual(self.group.anchor_entry.host_pool.available_size(), 8 * PAGE)


class TestStripeOffIsStock(unittest.TestCase):
    def setUp(self):
        self._patches = _cpu_transfer_kernels()
        placement = StripedHostPlacement(num_ranks=2, rank=0, page_size=PAGE)
        self.anchor = _striped_anchor(placement, 4)
        group = _dsv4_host_group(
            self.anchor, _dsv4_device_buffers(4, seed=0), 8, "layer_first"
        )
        self.controller = _controller(group, host_stripe=None)

    def tearDown(self):
        for patch in self._patches:
            patch.stop()

    def test_stripe_off_moves_full_op(self):
        layout = "layer_first"
        device_buffers = _dsv4_device_buffers(8, seed=3)
        placement = StripedHostPlacement(num_ranks=2, rank=1, page_size=PAGE)
        anchor = _striped_anchor(placement, 4, layout=layout)
        group = _dsv4_host_group(anchor, device_buffers, 8, layout)
        host, device = _random_page_op(anchor, 4, torch.arange(8), seed=0)
        _, _, stock_transfers = _controller(group, None).move_hybrid_indices(
            _op(host, device)
        )
        _, _, striped_transfers = _controller(group, placement).move_hybrid_indices(
            _op(host, device)
        )
        self.assertEqual(stock_transfers[0].host_indices.numel(), 4 * PAGE)
        self.assertEqual(striped_transfers[0].host_indices.numel(), 2 * PAGE)

    def test_write_issues_no_collective(self):
        self.assertFalse(self.controller.backup_admission_is_collective())
        with mock.patch.object(torch.distributed, "all_reduce") as all_reduce:
            host = self.controller.write(
                torch.arange(2 * PAGE), extra_pools=_sidecars(), flush=False
            )
            with mock.patch.object(self.anchor, "alloc", return_value=None):
                failed = self.controller.write(
                    torch.arange(PAGE), extra_pools=_sidecars(), flush=False
                )
        all_reduce.assert_not_called()
        self.assertEqual(host.numel(), 2 * PAGE)
        self.assertIsNone(failed)
        (op,) = self.controller.write_queue
        self.assertEqual(
            [t.host_indices.numel() for t in op.pool_transfers],
            [2 * PAGE] * len(DSV4_POOLS),
        )

    def test_failed_host_precheck_skips_write(self):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = self.controller
        cache.evict_host = lambda num_tokens: 0
        with (
            mock.patch.object(self.anchor, "available_size", return_value=0),
            mock.patch.object(self.controller, "write") as write,
        ):
            self.assertIsNone(
                cache._execute_kv_backup(0, torch.arange(PAGE), {}, _sidecars())
            )
        write.assert_not_called()


NUM_GLOO_RANKS = 4
# (op index, failing rank) per scenario; the second injects at the pre-check.
_ALLOC_FAILURES = {3: 2, 6: 0}
_PRECHECK_FAILURE = (8, 1)


def _admission_worker(rank, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=NUM_GLOO_RANKS,
        timeout=timedelta(seconds=60),
    )
    patches = _cpu_transfer_kernels()
    # A distinct group object catches a collective issued on the default group.
    tp_group = torch.distributed.new_group(list(range(NUM_GLOO_RANKS)))
    try:
        _run_admission_trace(rank, tp_group)
    finally:
        for patch in patches:
            patch.stop()
        torch.distributed.destroy_process_group()


def _run_admission_trace(rank, tp_group):
    placement = StripedHostPlacement(
        num_ranks=NUM_GLOO_RANKS, rank=rank, page_size=PAGE
    )
    anchor = _striped_anchor(placement, 3)
    group = _dsv4_host_group(anchor, _dsv4_device_buffers(4, seed=0), 3, "layer_first")
    controller = _controller(group, host_stripe=placement)
    controller.tp_group = tp_group
    real_all_reduce = torch.distributed.all_reduce
    collectives = []

    def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, **kwargs):
        assert group is tp_group, group
        collectives.append(op)
        return real_all_reduce(tensor, op=op, group=group, **kwargs)

    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = controller
    cache.evict_host = lambda num_tokens: 0
    cache.tree_core = SimpleNamespace(rotation_phase_of=lambda node_id: None)

    real_alloc = anchor.alloc
    real_available = anchor.available_size
    trace = []
    for step in range(12):
        device = torch.arange(2 * PAGE, dtype=torch.int64)
        fail_alloc = _ALLOC_FAILURES.get(step) == rank
        fail_precheck = _PRECHECK_FAILURE == (step, rank)
        with (
            mock.patch.object(
                anchor,
                "alloc",
                side_effect=(lambda n: None)
                if fail_alloc or fail_precheck
                else real_alloc,
            ),
            mock.patch.object(
                anchor,
                "available_size",
                side_effect=(lambda: 0) if fail_precheck else real_available,
            ),
            mock.patch.object(torch.distributed, "all_reduce", all_reduce),
        ):
            host = cache._execute_kv_backup(step, device, {}, _sidecars())
        if host is not None and step % 3 == 2:
            anchor.free(host)
        snapshot = (
            None if host is None else host.tolist(),
            anchor.digest(),
            anchor.available_size(),
            len(controller.write_queue),
        )
        gathered = [None] * NUM_GLOO_RANKS
        torch.distributed.all_gather_object(gathered, snapshot)
        assert all(s == gathered[0] for s in gathered), (step, gathered)
        trace.append(host is None)

    # 12 logical pages, 2 per op, every third admitted op freed: step 10 fills
    # the tier, so step 11 is vetoed on every rank without any injection.
    vetoed = {step for step, skipped in enumerate(trace) if skipped}
    assert vetoed == {3, 6, 8, 11}, trace
    # One MIN admission collective per backup, on every rank.
    assert collectives == [torch.distributed.ReduceOp.MIN] * 12, collectives


class TestBackupAdmissionConsensus(unittest.TestCase):
    def test_injected_alloc_failure_vetoes_op_on_every_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            torch.multiprocessing.spawn(
                _admission_worker,
                args=(f"{directory}/store",),
                nprocs=NUM_GLOO_RANKS,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
