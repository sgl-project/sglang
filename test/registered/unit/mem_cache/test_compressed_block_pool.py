"""r4 real fixed-budget block/storage tests, independent of fixture adapters."""

import concurrent.futures
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Imports the established CPU-only package bootstrap, never substitutes pool code.
from test_compressed_hicache import BlockPool, materialize_pages, payload

# isort: split
from sglang.srt.kv_compression.types import BufferDrainError, EncodedPage


def pool(raw=8192, bound=12288, verify=False):
    return BlockPool(
        raw, 16 * 1024**2, reservation_bytes=bound, verify=verify, pin_memory=False
    )


def store(p, handles, sizes, refs=None, checksum=None):
    refs = refs or [int(h) for h in handles]
    pages = [
        EncodedPage(
            torch.full((n,), i % 256, dtype=torch.uint8), "lz4", p.size_per_token
        )
        for i, n in enumerate(sizes)
    ]
    write = p.prepare_write(handles, sizes)
    try:
        with p.stage(read=False) as stage:
            for i, page in enumerate(pages):
                stage[i, : page.nbytes].copy_(page.data)
            write.scatter(stage)
            write.publish(refs, pages, [checksum] * len(pages))
    finally:
        write.close()
    return pages


def conserved(p):
    s = p.snapshot()
    assert (
        s["free_blocks"] + s["reserved_blocks"] + s["live_blocks"] + s["retired_blocks"]
        == p.block_count
    )
    assert (
        p.arena.numel() + p.metadata_bytes + p.staging_bytes + p.scratch_bytes
        <= p.budget_bytes
    )


@pytest.mark.parametrize("n", [1, 4095, 4096, 4097, 8192, 12288])
def test_lengths_and_noncontiguous_blocks(n):
    p = pool()
    h = p.alloc(64)
    p.free(h[::2])
    new = p.alloc(32)
    pages = store(p, new, [n] * 32)
    chains = p._chains(new)
    assert any(not np.all(np.diff(c) == 1) for c in chains if len(c) > 1) or n <= 4096
    leases = [p.acquire(x) for x in new]
    with materialize_pages([x.future.result() for x in leases]) as actual:
        assert all(torch.equal(a.data, b.data) for a, b in zip(actual, pages))
        # Physical block tail and next row must not leak through view/padding.
        assert not p.read_stage[0, n:].any()
    p.free(new)
    assert p.snapshot()["retired_blocks"] == 32 * ((n + 4095) // 4096)
    for lease in leases:
        lease.close()
    p.free(h[1::2])
    conserved(p)
    assert p.free_block_count == p.block_count


def test_shrink_returns_reusable_blocks_no_holes():
    p = BlockPool(147456, 64 * 1024**2, reservation_bytes=197632, pin_memory=False)
    h = p.alloc(2)
    before = p.free_block_count
    w = p.prepare_write(h, [148224] * 2)
    assert p.free_block_count - before == 24
    assert all(len(c) == 37 for c in w.blocks)
    p.free(h)
    assert p.active_writers == 2
    w.close()
    conserved(p)
    assert p.free_block_count == p.block_count


def test_mapping_is_frozen_and_cancel_late_write_cannot_publish():
    p = pool()
    h = p.alloc(2)
    w = p.prepare_write(h, [5000, 8000])
    before = p._chains(h)
    with pytest.raises(ValueError):
        p.prepare_write(h, [4096, 4096])
    p.free(h)
    assert p.retired_bytes == 4 * 4096
    other = p.alloc(2)
    assert not set(np.concatenate(before)).intersection(
        np.concatenate(p._chains(other))
    )
    with p.stage(read=False) as stage:
        stage[:2].fill_(17)
        w.scatter(stage)
        with pytest.raises(ValueError):
            w.publish([11, 12], [NS(encoding="lz4")] * 2, [None] * 2)
    w.close()
    p.free(other)
    conserved(p)
    assert p.free_block_count == p.block_count


def test_failed_copy_and_publish_roll_back_then_next_request():
    p = pool(verify=True)
    h = p.alloc(2)
    w = p.prepare_write(h, [1000, 5000])
    with p.stage(read=False) as stage:
        # A synchronous scatter failure may have partially touched owned bytes.
        with (
            patch.object(
                torch.Tensor, "index_copy_", side_effect=RuntimeError("partial scatter")
            ),
            pytest.raises(RuntimeError),
        ):
            w.scatter(stage)
        assert not w.written
        with pytest.raises(ValueError):
            w.publish([1, 2], [NS(encoding="lz4")] * 2, [bytes(32)] * 2)
    w.close()
    p.free(h)
    conserved(p)
    h = p.alloc(2)
    w = p.prepare_write(h, [1000, 5000])
    with p.stage(read=False) as stage:
        w.scatter(stage)
        with pytest.raises(ValueError, match="SHA256"):
            w.publish([1, 2], [NS(encoding="lz4")] * 2, [bytes(32), None])
    assert p.acquire_ref(1, "lz4") is None
    w.close()
    p.free(h)
    h = p.alloc(1)
    store(p, h, [5000], checksum=bytes(32))
    p.free(h)
    conserved(p)


def test_undrained_stage_and_write_stay_quarantined():
    p = pool()
    h = p.alloc(1)
    w = p.prepare_write(h, [4096])
    with pytest.raises(BufferDrainError), p.stage(read=False):
        w.quarantine()
        raise BufferDrainError("injected DMA uncertainty")
    w.close()
    p.free(h)
    assert p.active_writers == 1 and p.has_readers() and p.retired_bytes == 4096
    with pytest.raises(RuntimeError):
        p.clear()
    with pytest.raises(BufferDrainError), p.stage(read=False):
        pass


@pytest.mark.parametrize(
    "kind", ["chain", "owner", "cycle", "capacity", "free_stack", "negative_free"]
)
def test_corrupt_mapping_rejected_before_mutation(kind):
    p = pool()
    h = p.alloc(2)
    slot, r = p._record(h[0])
    first = int(r["first"])
    if kind == "chain":
        p.next_block[first] = -1
    if kind == "owner":
        p.block_owner[first] = slot + 1
    if kind == "cycle":
        p.next_block[first] = first
    if kind == "capacity":
        r["capacity"] = 2**50
    if kind == "free_stack":
        p.free_blocks[p.free_block_count - 1] = first
    if kind == "negative_free":
        p.free_blocks[p.free_block_count - 1] = -1
    before = p.snapshot()
    with pytest.raises(ValueError):
        if kind in ("free_stack", "negative_free"):
            p.alloc(1)
        else:
            p.free(h)
    assert before == p.snapshot()


def test_reader_lane_contention_and_pending_state():
    p = pool()
    h = p.alloc(2)
    expected = store(p, h, [4097, 8192])
    leases = [p.acquire(x) for x in h]
    entered = threading.Event()
    release = threading.Event()

    def first():
        with materialize_pages([leases[0].future.result()]) as pages:
            entered.set()
            release.wait(5)
            assert torch.equal(pages[0].data, expected[0].data)

    def second():
        with materialize_pages([leases[1].future.result()]) as pages:
            assert torch.equal(pages[0].data, expected[1].data)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        a = ex.submit(first)
        assert entered.wait(5)
        b = ex.submit(second)
        deadline = time.monotonic() + 5
        while not p.stage_waiters and time.monotonic() < deadline:
            time.sleep(0.001)
        assert p.stage_waiters == 1 and p.has_readers()
        # Write lane stays independent of blocked read users.
        with p.stage(read=False):
            pass
        release.set()
        a.result(5)
        b.result(5)
    for lease in leases:
        lease.close()
    p.free(h)
    assert not p.has_readers()
    conserved(p)


def test_duplicate_and_invalid_write_atomicity():
    p = pool()
    h = p.alloc(2)
    before = p.snapshot()
    for handles, sizes in [
        (h, [1]),
        ([h[0], h[0]], [1, 1]),
        (h, [1, 0]),
        (h, [1, 12289]),
    ]:
        with pytest.raises(ValueError):
            p.prepare_write(handles, sizes)
        assert p.snapshot() == before
    p.free(h)
    with pytest.raises(ValueError):
        p.free(h)


def test_mixed_raw_lz4_and_tail_window():
    p = pool()
    handles = p.alloc(65)
    for start in range(0, 65, 64):
        h = handles[start : start + 64]
        sizes = [8192 if i % 2 else 4097 for i in range(len(h))]
        w = p.prepare_write(h, sizes)
        pages = [
            EncodedPage(
                torch.full((n,), i, dtype=torch.uint8),
                "raw" if n == 8192 else "lz4",
                8192,
            )
            for i, n in enumerate(sizes)
        ]
        with p.stage(read=False) as stage:
            for i, page in enumerate(pages):
                stage[i, : page.nbytes].copy_(page.data)
            w.scatter(stage)
            w.publish([int(x) for x in h], pages, [None] * len(h))
        w.close()
        for handle, page in zip(h, pages):
            with p.acquire(handle) as lease:
                assert torch.equal(payload(lease.future.result()), page.data)
    p.free(handles)
    conserved(p)


def test_strict_budget_and_id_bounds():
    with pytest.raises(ValueError, match="budget"):
        BlockPool(4096, 65536, pin_memory=False)
    with pytest.raises(ValueError, match="int32"):
        BlockPool(4096, 2**60, pin_memory=False)


def test_fault_is_exact_single_operation_and_disabled_by_default(monkeypatch):
    import json

    from sglang.srt.kv_compression.faults import TestFault

    monkeypatch.delenv("SGLANG_KV_COMPRESSION_TEST_FAULT", raising=False)
    TestFault().check("write:1", "after_scatter")
    monkeypatch.setenv(
        "SGLANG_KV_COMPRESSION_TEST_FAULT",
        json.dumps(
            {"operation": "write:17", "point": "after_scatter", "kind": "error"}
        ),
    )
    f = TestFault()
    f.check("write:18", "after_scatter")
    f.check("write:17", "before_publish")
    assert not f.fired
    with pytest.raises(RuntimeError, match="write:17"):
        f.check("write:17", "after_scatter")
    f.check("write:17", "after_scatter")
    monkeypatch.setenv(
        "SGLANG_KV_COMPRESSION_TEST_FAULT",
        json.dumps({"operation": "write:*", "point": "after_scatter", "kind": "error"}),
    )
    with pytest.raises(ValueError):
        TestFault()


def test_concurrent_short_writes_late_readers_and_identity_reuse():
    p = pool()

    def write_turn(worker):
        for turn in range(8):
            # Includes 1-page nodes, tail windows and differently sized results.
            count = (worker + turn) % 7 + 1
            h = p.alloc(count)
            assert h is not None
            sizes = [1, 4096, 4097, 8000, 12288, 17, 9999][:count]
            pages = store(p, h, sizes)
            leases = [p.acquire(x) for x in h]
            p.free(h)
            with materialize_pages([l.future.result() for l in leases]) as actual:
                assert all(torch.equal(a.data, b.data) for a, b in zip(actual, pages))
            for l in reversed(leases):
                l.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for f in [ex.submit(write_turn, i) for i in range(4)]:
            f.result(30)
    conserved(p)
    assert not p.has_readers() and p.free_block_count == p.block_count


@pytest.mark.parametrize("undrained", [False, True])
def test_actual_pd_assembly_borrows_host_then_owns_wire(undrained):
    import contextlib
    import json
    import logging
    from unittest.mock import Mock

    from sglang.srt.kv_compression.provider import (
        HostEncodedKVProvider,
        RepresentationSpec,
    )
    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.types import align_bytes
    from test_compressed_hicache import layout, source_function

    model = layout()
    p = BlockPool(model.page_bytes, 8 * 1024**2, pin_memory=False)
    runtime = KVCompressionRuntime(model, "passthrough", 4 * 1024**2)
    runtime.provider = HostEncodedKVProvider(
        p, RepresentationSpec(model.tag, "passthrough")
    )
    h = p.alloc(2)
    pages = [
        EncodedPage(
            torch.full((model.page_bytes,), i + 4, dtype=torch.uint8),
            "raw",
            model.page_bytes,
        )
        for i in range(2)
    ]
    w = p.prepare_write(h, [model.page_bytes] * 2)
    with p.stage(read=False) as stage:
        for i, page in enumerate(pages):
            stage[i, : page.nbytes].copy_(page.data)
        w.scatter(stage)
        w.publish([101, 102], pages, [None, None])
    w.close()
    fake_torch = NS(
        empty=torch.empty,
        uint8=torch.uint8,
        cuda=NS(stream=lambda s: contextlib.nullcontext()),
    )
    method = source_function(
        "disaggregation/mooncake/compression.py",
        "encode_pages",
        "CompressionRuntime",
        torch=fake_torch,
        materialize_pages=materialize_pages,
        align_bytes=align_bytes,
        BufferDrainError=BufferDrainError,
        json=json,
        logger=logging.getLogger(__name__),
    )
    sender = NS(
        transport_failed=False,
        shared=runtime,
        local=threading.local(),
        wire_capacity=16384,
        device="cpu",
        outputs=[],
        manager=NS(_register_staging_memory=Mock()),
    )
    stream = NS(
        synchronize=Mock(side_effect=RuntimeError("uncertain") if undrained else None)
    )
    try:
        if undrained:
            with pytest.raises(BufferDrainError):
                method(sender, [0, 1], [101, 102], None, stream)
            assert p.active_readers == 2 and p.has_readers() and p.quarantined
            assert not runtime.idle()
        else:
            wire, desc, _ = method(sender, [0, 1], [101, 102], None, stream)
            assert p.active_readers == 0 and runtime.stats["encoded_pages"] == 0
            p.free(h)
            for (offset, n, encoding), page in zip(desc, pages):
                assert encoding == "raw" and torch.equal(
                    wire[offset : offset + n], page.data
                )
    finally:
        runtime.close()
