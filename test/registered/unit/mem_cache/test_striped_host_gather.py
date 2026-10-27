"""Striped HiCache host tier: padded chunked load gather."""

import ast
import inspect
import os
import tempfile
import unittest
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache import memory_pool_host
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host import striped_host
from sglang.srt.mem_cache.pool_host.striped_host import (
    PyncclStripedGatherTransport,
    StripedGatherError,
    StripedGatherExecutor,
    StripedGatherFatal,
    StripedHostPlacement,
    StripedLogicalHostPool,
    TorchDistStripedGatherTransport,
    _page_rows_u8,
    build_striped_load_plan,
    collect_striped_plan_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

PAGE = 4
DEVICE_PAGES = 48
# DSv4-Pro per-page row bytes per pool on MI355X; c4 layers carry four pools, c128 two.
C4_POOLS = (
    (32768, torch.uint8),
    (4096, torch.bfloat16),
    (256, torch.uint8),
    (8192, torch.float32),
)
C128_POOLS = ((1024, torch.bfloat16), (256, torch.uint8))
# Synthetic cases use a few layers; the DSv4 wiring case uses all of them.
NUM_C4_LAYERS = 3
NUM_C128_LAYERS = 2
ROW_BYTES = NUM_C4_LAYERS * sum(b for b, _ in C4_POOLS) + NUM_C128_LAYERS * sum(
    b for b, _ in C128_POOLS
)
DSV4_C4_LAYERS = 30
DSV4_C128_LAYERS = 31
DSV4_ROW_BYTES = 1_399_040
DSV4_DEVICE_PAGES = 16


def _rows(gen, num_pages, item_bytes, dtype, unified=False):
    """A device page buffer shaped like real pools: [pages, page_size, row / page_size]."""
    lead = 2 * num_pages if unified else num_pages
    raw = torch.randint(0, 256, (lead, item_bytes), generator=gen, dtype=torch.uint8)
    if unified:
        # unified_kv pools are a dim-0 narrow of a larger buffer.
        raw = raw.narrow(0, num_pages, num_pages)
    elem = torch.empty((), dtype=dtype).element_size()
    return raw.view(dtype).view(num_pages, PAGE, item_bytes // elem // PAGE)


def _device_buffers(seed, num_layers=None):
    """Replicated reference device pages: identical on every rank for a seed."""
    gen = torch.Generator().manual_seed(seed)
    pools = [C4_POOLS] * NUM_C4_LAYERS + [C128_POOLS] * NUM_C128_LAYERS
    return tuple(
        tuple((_rows(gen, DEVICE_PAGES, ib, dt), ib) for ib, dt in layer_pools)
        for layer_pools in pools[:num_layers]
    )


def _dsv4_layers(seed, unified):
    """Plan layers from collect_striped_plan_layers over real DSv4 host pools."""
    gen = torch.Generator().manual_seed(seed)
    c4_map = {2 * i + 1: i for i in range(DSV4_C4_LAYERS)}
    c128_map = {2 * i: i for i in range(DSV4_C128_LAYERS)}
    transfers = []
    with mock.patch.object(
        memory_pool_host, "host_memory_budget_bytes", return_value=1 << 40
    ):
        for pools, num_layers, mapping in (
            (C4_POOLS, DSV4_C4_LAYERS, c4_map),
            (C128_POOLS, DSV4_C128_LAYERS, c128_map),
        ):
            for item_bytes, dtype in pools:
                host_pool = DeepSeekV4PagedHostPool(
                    pool_name=f"pool{item_bytes}",
                    device_buffers=[
                        _rows(gen, DSV4_DEVICE_PAGES, item_bytes, dtype, unified)
                        for _ in range(num_layers)
                    ],
                    item_bytes=item_bytes,
                    num_host_pages=1,
                    slot_page_size=PAGE,
                    device="cpu",
                    pin_memory=False,
                    page_aligned_only=True,
                )
                transfers.append(
                    SimpleNamespace(host_pool=host_pool, layer_mapper=mapping.get)
                )
    return collect_striped_plan_layers(
        transfers, transfer_layer_id_max=DSV4_C4_LAYERS + DSV4_C128_LAYERS
    )


def _host_pages(case, num_ranks):
    """Logical host pages of a load, by scenario."""
    if case == "zero":
        return []
    if case == "single_owner":
        return [1 + num_ranks * i for i in range(5)]
    if case == "uneven":
        # Owner 0 gets 7 pages, the others 0-3: k_max is set by one owner.
        pages = [num_ranks * i for i in range(7)]
        pages += [num_ranks * i + c for c in range(1, num_ranks) for i in range(c % 4)]
        return pages[::-1]
    if case == "pool_alloc":
        placement = StripedHostPlacement(num_ranks=num_ranks, rank=0, page_size=PAGE)
        pool = StripedLogicalHostPool(
            per_rank_size=12 * PAGE, page_size=PAGE, placement=placement
        )
        held = [pool.alloc(PAGE * k) for k in (3, 5, 2, 7)]
        pool.free(held[1])
        pool.free(held[2])
        return (pool.alloc(PAGE * 9) // PAGE)[::PAGE].tolist()
    raise ValueError(case)


def _staging(num_ranks, chunk_pages, row_bytes=ROW_BYTES):
    """Total staging for chunks of ``chunk_pages`` rows; None = env default."""
    if chunk_pages is None:
        return None
    return num_ranks * chunk_pages * row_bytes + num_ranks - 1


CASES = (
    # (scenario, pages per chunk; None = the env default staging)
    ("uneven", None),
    ("uneven", 1),
    ("pool_alloc", 3),  # chunks of 3 pages with a partial tail
    ("single_owner", 2),
    ("zero", None),
)


class _LoggingTransport:
    def __init__(self, inner):
        self.inner = inner
        self.world_size = inner.world_size
        self.rank = inner.rank
        self.log = []

    def all_gather(self, recv, send):
        assert send.data_ptr() == recv.data_ptr() + self.rank * send.numel()
        self.log.append((recv.numel(), send.numel()))
        self.inner.all_gather(recv, send)


class _WaitEvent:
    def __init__(self):
        self.waited = []

    def wait(self, stream=None):
        self.waited.append(stream)


def _run_case(rank, num_ranks, group, case, executor, case_id, make_layers):
    placement = executor.placement
    reference = make_layers(11)
    # This rank's device KV: stale, rank-specific bytes in the load's pages.
    actual = make_layers(1000 + rank)
    num_device_pages = actual[0][0][0].shape[0]
    row_bytes = sum(ib for layer in actual for _, ib in layer)
    host_pages = torch.tensor(_host_pages(case, num_ranks), dtype=torch.int64)
    host = (host_pages[:, None] * PAGE + torch.arange(PAGE)).flatten()
    # Device pages are a rank-identical permutation unrelated to host pages.
    perm = torch.randperm(num_device_pages, generator=torch.Generator().manual_seed(7))
    dev_pages = perm[: host_pages.numel()]
    device = (dev_pages[:, None] * PAGE + torch.arange(PAGE)).flatten()

    plan = build_striped_load_plan(
        placement, host_indices=host, device_indices=device, seq=case_id, layers=actual
    )
    # Owned "H2D": the owner's host copy equals the reference rows.
    own_pages = plan.own_dev.reshape(-1, PAGE)[:, 0] // PAGE
    for ref_layer, act_layer in zip(reference, actual):
        for (ref, ib), (act, _) in zip(ref_layer, act_layer):
            _page_rows_u8(act, ib)[own_pages] = _page_rows_u8(ref, ib)[own_pages]
    untouched = [[_page_rows_u8(a, ib).clone() for a, ib in layer] for layer in actual]

    transport = executor.transport
    log_start = len(transport.log)
    gathers_start = executor.num_gathers
    bytes_start = executor.gathered_bytes
    signature = executor.signature(plan)
    h2d_done = _WaitEvent()
    event = executor.gather_plan(plan, stream=None, wait_events=(h2d_done,))
    assert event is plan.gather_done_event and event.query()
    assert h2d_done.waited == [None]

    loaded = set(dev_pages.tolist())
    other = torch.tensor(
        [p for p in range(num_device_pages) if p not in loaded], dtype=torch.int64
    )
    for li, (ref_layer, act_layer) in enumerate(zip(reference, actual)):
        for pi, ((ref, ib), (act, _)) in enumerate(zip(ref_layer, act_layer)):
            got = _page_rows_u8(act, ib)
            want = _page_rows_u8(ref, ib)
            assert torch.equal(got[dev_pages], want[dev_pages]), (case, rank, li, pi)
            assert torch.equal(got[other], untouched[li][pi][other]), (
                case,
                rank,
                li,
                pi,
            )

    if plan.k_max == 0:
        expected = 0
    else:
        expected = -(-plan.k_max // executor.chunk_pages(row_bytes))
    log = transport.log[log_start:]
    assert len(log) == expected, (case, log)
    assert executor.num_gathers - gathers_start == expected
    peer_pages = sum(plan.pages_per_owner) - plan.pages_per_owner[rank]
    assert executor.gathered_bytes - bytes_start == peer_pages * row_bytes
    assert signature.row_bytes == row_bytes and signature.k_max == plan.k_max
    shared = [None] * num_ranks
    torch.distributed.all_gather_object(shared, (log, signature), group=group)
    assert all(item == shared[0] for item in shared), shared
    try:
        executor.gather_plan(plan, stream=None)
    except StripedGatherError:
        pass
    else:
        raise AssertionError("a plan must not be gathered twice")


def _executor(rank, num_ranks, group, staging):
    placement = StripedHostPlacement(num_ranks=num_ranks, rank=rank, page_size=PAGE)
    transport = _LoggingTransport(TorchDistStripedGatherTransport(group))
    return StripedGatherExecutor(
        placement, transport, device=torch.device("cpu"), staging_bytes=staging
    )


def _gather_worker(rank, num_ranks, rendezvous):
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=num_ranks,
        timeout=timedelta(seconds=120),
    )
    # A distinct group catches a collective issued on the default group.
    group = torch.distributed.new_group(list(range(num_ranks)))
    real_all_gather = torch.distributed.all_gather

    def all_gather(outputs, tensor, group=None, **kwargs):
        assert group is not None and group is not torch.distributed.group.WORLD
        return real_all_gather(outputs, tensor, group=group, **kwargs)

    try:
        torch.distributed.all_gather = all_gather
        case_id = 0
        for case, chunk_pages in CASES:
            executor = _executor(
                rank, num_ranks, group, _staging(num_ranks, chunk_pages)
            )
            _run_case(rank, num_ranks, group, case, executor, case_id, _device_buffers)
            case_id += 1
        # One executor reused across plans of different row bytes: staging is
        # shared, never regrown, and chunking follows each plan's row.
        executor = _executor(rank, num_ranks, group, _staging(num_ranks, 3))
        for case, num_layers in (("uneven", None), ("pool_alloc", 2), ("uneven", 4)):
            _run_case(
                rank,
                num_ranks,
                group,
                case,
                executor,
                case_id,
                lambda seed, n=num_layers: _device_buffers(seed, n),
            )
            case_id += 1
        if num_ranks == 4:
            # Real DSv4 wiring: collect_striped_plan_layers over DSv4 host pools,
            # default and unified_kv-style (narrowed) device buffers.
            for unified in (False, True):
                executor = _executor(
                    rank, num_ranks, group, _staging(num_ranks, 3, DSV4_ROW_BYTES)
                )
                assert (
                    executor.validate_layers(_dsv4_layers(0, unified)) == DSV4_ROW_BYTES
                )
                _run_case(
                    rank,
                    num_ranks,
                    group,
                    "uneven",
                    executor,
                    case_id,
                    lambda seed, u=unified: _dsv4_layers(seed, u),
                )
                case_id += 1
    finally:
        torch.distributed.all_gather = real_all_gather
        torch.distributed.destroy_process_group()


class TestStripedGatherGloo(unittest.TestCase):
    def _spawn(self, num_ranks):
        with tempfile.TemporaryDirectory() as directory:
            os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
            torch.multiprocessing.spawn(
                _gather_worker,
                args=(num_ranks, f"{directory}/store"),
                nprocs=num_ranks,
                join=True,
            )

    def test_gather_n4(self):
        self._spawn(4)


class _FakePyncclComm:
    def __init__(self, world_size=4, rank=1, available=True, refuse=False):
        self.world_size = world_size
        self.rank = rank
        self.available = available
        self.device = torch.device("cpu")
        self.disabled = True
        self.refuse = refuse
        self.calls = []

    @contextmanager
    def change_state(self, enable=None):
        old = self.disabled
        self.disabled = self.refuse or not enable
        try:
            yield
        finally:
            self.disabled = old

    def all_gather(self, output, input, sizes=None):
        self.calls.append((self.disabled, output.numel(), input.numel(), sizes))


class _FakeGroup:
    def __init__(self, comm, world_size=4, rank=1):
        self.pynccl_comm = comm
        self.world_size = world_size
        self.rank_in_group = rank


class TestPyncclTransport(unittest.TestCase):
    def test_enables_pynccl_for_the_call_and_restores(self):
        comm = _FakePyncclComm()
        transport = PyncclStripedGatherTransport(_FakeGroup(comm))
        recv = torch.empty(4 * 8, dtype=torch.uint8)
        transport.all_gather(recv, recv[8:16])
        self.assertEqual(comm.calls, [(False, 32, 8, None)])
        self.assertTrue(comm.disabled)

    def test_refuses_missing_or_unavailable_or_mismatched_comm(self):
        for group in (
            _FakeGroup(None),
            _FakeGroup(_FakePyncclComm(available=False)),
            _FakeGroup(_FakePyncclComm(world_size=2)),
            _FakeGroup(_FakePyncclComm(rank=0)),
        ):
            with self.assertRaises(StripedGatherError):
                PyncclStripedGatherTransport(group)

    def test_refused_enable_is_fatal_not_silent(self):
        comm = _FakePyncclComm(refuse=True)
        transport = PyncclStripedGatherTransport(_FakeGroup(comm))
        recv = torch.empty(32, dtype=torch.uint8)
        with self.assertRaises(StripedGatherError):
            transport.all_gather(recv, recv[8:16])
        self.assertEqual(comm.calls, [])


class TestGatherStaticChecks(unittest.TestCase):
    """The prod gather never takes the ProcessGroupNCCL path."""

    BANNED = frozenset(
        {
            "all_gather_single",
            "all_gather_into_tensor",
            "_all_gather_into_tensor",
            "device_group",
        }
    )

    def test_module_never_names_the_pg_nccl_all_gather(self):
        tree = ast.parse(inspect.getsource(striped_host))
        names = {
            node.id if isinstance(node, ast.Name) else node.attr
            for node in ast.walk(tree)
            if isinstance(node, (ast.Name, ast.Attribute))
        }
        self.assertFalse(names & self.BANNED, names & self.BANNED)


class TestGatherLocal(unittest.TestCase):
    def test_refuses_non_contiguous_page_rows(self):
        buf = torch.zeros(8, 16, dtype=torch.uint8)
        with self.assertRaises(StripedGatherError):
            _page_rows_u8(buf.t(), 8)
        with self.assertRaises(StripedGatherError):
            _page_rows_u8(buf, 32)
        self.assertEqual(_page_rows_u8(buf.view(torch.bfloat16), 16).shape, (8, 16))

    def test_staging_is_allocated_at_init_and_never_grows(self):
        executor = _local_executor(staging_bytes=4 * ROW_BYTES)
        staging = executor._staging
        self.assertEqual(staging.numel(), 4 * ROW_BYTES)
        self.assertEqual(executor.chunk_pages(ROW_BYTES), 2)
        self.assertEqual(executor.validate_layers(_device_buffers(seed=0)), ROW_BYTES)
        self.assertIs(executor._staging, staging)
        too_small = _local_executor(staging_bytes=2 * ROW_BYTES - 1)
        with self.assertRaises(StripedGatherError):
            too_small.validate_layers(_device_buffers(seed=0))
        plan = _local_plan()
        with self.assertRaises(StripedGatherError):
            too_small.gather_plan(plan, stream=None)
        # A check failure precedes any collective, so the plan stays gatherable.
        self.assertFalse(plan.gather_started)

    def test_failure_after_start_is_fatal_and_not_retryable(self):
        class _BrokenEvent:
            def wait(self, stream=None):
                raise RuntimeError("H2D failed")

        cases = (
            (_RaisingTransport(2, 0), ()),  # the collective itself fails
            (_RaisingTransport(2, 0, fail=False), (_BrokenEvent(),)),
        )
        for transport, waits in cases:
            plan = _local_plan()
            executor = _local_executor(transport=transport)
            with self.assertRaises(StripedGatherFatal):
                executor.gather_plan(plan, stream=None, wait_events=waits)
            self.assertTrue(plan.gather_started)
            self.assertIsNone(plan.gather_done_event)
            with self.assertRaises(StripedGatherError) as ctx:
                executor.gather_plan(plan, stream=None)
            self.assertNotIsInstance(ctx.exception, StripedGatherFatal)


class _RaisingTransport:
    def __init__(self, world_size, rank, fail=True):
        self.world_size, self.rank, self.fail = world_size, rank, fail

    def all_gather(self, recv, send):
        if self.fail:
            raise RuntimeError("collective failed")


def _local_executor(staging_bytes=1 << 20, transport=None):
    transport = transport or _RaisingTransport(2, 0)
    placement = StripedHostPlacement(
        num_ranks=transport.world_size, rank=transport.rank, page_size=PAGE
    )
    return StripedGatherExecutor(
        placement, transport, device=torch.device("cpu"), staging_bytes=staging_bytes
    )


def _local_plan():
    placement = StripedHostPlacement(num_ranks=2, rank=0, page_size=PAGE)
    host = torch.arange(4 * PAGE, dtype=torch.int64)
    return build_striped_load_plan(
        placement,
        host_indices=host,
        device_indices=host,
        seq=3,
        layers=_device_buffers(seed=0),
    )


if __name__ == "__main__":
    unittest.main()
