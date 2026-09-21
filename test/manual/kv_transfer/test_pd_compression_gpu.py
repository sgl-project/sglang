"""Run in the built GPU image: python -m pytest -q <this file>."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("tokens", [1, 31, 1023, 1024])
def test_fragmented_bf16_kv_roundtrip(tokens):
    from sglang.srt.disaggregation.common.staging_buffer import (
        StagingBuffer,
        gather_all_layers_to_staging,
        scatter_staging_to_kv,
    )
    from sglang.srt.disaggregation.compression.backend import NvcompLZ4Backend

    torch.manual_seed(42)
    device = torch.device("cuda", 0)
    # Small ordinary GQA pool; noncontiguous source/destination token rows.
    rows, layers, heads, head_dim = 2 * tokens + 3, 2, 8, 128
    k = [
        torch.randn(rows, heads, head_dim, device=device, dtype=torch.bfloat16)
        for _ in range(layers)
    ]
    v = [torch.randn_like(t) for t in k]
    dk, dv = [torch.zeros_like(t) for t in k], [torch.zeros_like(t) for t in v]
    source = torch.arange(tokens, device=device) * 2
    target = torch.flip(source, dims=[0]) + 1
    nbytes = tokens * layers * heads * head_dim * 4
    packed = StagingBuffer(nbytes, str(device), 0)
    gather_all_layers_to_staging(k, v, source.cpu().numpy(), packed, 0, heads, 1, 0)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    backend = NvcompLZ4Backend(device, stream)
    raw = packed.buffer[:nbytes]
    with torch.cuda.stream(stream):
        encoded = torch.empty(
            backend.max_output_bytes(raw), dtype=torch.uint8, device=device
        )
        decoded = torch.empty_like(raw)
        size = backend.compress_into(raw, encoded)
        backend.decompress_into(encoded[:size], decoded)
        assert torch.equal(raw, decoded)
        scatter_staging_to_kv(decoded, dk, dv, target, 1, 1, 1, 0, heads)
        stream.synchronize()
    for original, restored in zip(k + v, dk + dv):
        assert torch.equal(
            original[source].view(torch.uint8), restored[target].view(torch.uint8)
        )


def test_uncompressed_size_mismatch_is_rejected():
    from sglang.srt.disaggregation.compression.backend import NvcompLZ4Backend

    device = torch.device("cuda", 0)
    stream = torch.cuda.Stream(device=device)
    backend = NvcompLZ4Backend(device, stream)
    with torch.cuda.stream(stream):
        source = torch.zeros(4096, dtype=torch.uint8, device=device)
        encoded = torch.empty(
            backend.max_output_bytes(source), dtype=torch.uint8, device=device
        )
        size = backend.compress_into(source, encoded)
        assert 0 < size < source.numel(), "wire length must be actual encoded length"
        with pytest.raises(ValueError, match="destination length"):
            backend.decompress_into(
                encoded[:size], torch.empty(4095, dtype=torch.uint8, device=device)
            )


@pytest.mark.parametrize("tokens", [1, 31, 65, 1024])
def test_shared_page_runtime_l2_only_restore_and_reuse(tokens):
    from sglang.srt.kv_compression.layout import KVLayoutAdapter
    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.store import CompressedHostKVCache
    from sglang.srt.kv_compression.types import new_page_refs

    torch.manual_seed(42)
    device = torch.device("cuda", 0)
    buffers = [
        torch.randn(2 * tokens + 3, 8, 128, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    ]
    # Deliberately exercise lz4 and raw fallback. This is NOT a ratio benchmark.
    for tensor in buffers:
        tensor[::4].zero_()
    layout = KVLayoutAdapter(buffers[:2], buffers[2:])
    source = list(range(0, 2 * tokens, 2))
    target = list(range(1, 2 * tokens + 1, 2))
    expected = layout.pack_pages(source).cpu()
    runtime = KVCompressionRuntime(layout, "lz4", 128 * 1024**2)
    pool = CompressedHostKVCache(layout.page_bytes, 64 * 1024**2)
    from sglang.srt.kv_compression.provider import (
        HostEncodedKVProvider,
        RepresentationSpec,
    )

    runtime.provider = HostEncodedKVProvider(
        pool, RepresentationSpec(runtime.layout.tag, runtime.mode)
    )
    refs = new_page_refs(tokens)
    ready = torch.cuda.Event()
    ready.record()
    leases = runtime.acquire_pages(refs, source, ready)
    handles = pool.alloc(tokens)
    try:
        assert handles is not None
        encodings = []
        for h, ref, lease in zip(handles.tolist(), refs, leases):
            page = lease.future.result(timeout=120)
            encodings.append(page.encoding)
        from sglang.srt.kv_compression.host_io import CompressedHostIO

        CompressedHostIO(runtime, pool)._write(handles.tolist(), refs, source, ready)
        for lease in leases:
            lease.close()
        assert "lz4" in encodings, "GPU decompression must actually be exercised"
        for tensor in buffers:
            tensor.zero_()
        ready = torch.cuda.Event()
        ready.record()
        leases = runtime.acquire_pages(refs, source, ready)
        before = runtime.stats["encoded_pages"]

        def restore():
            runtime.stream.wait_event(ready)
            runtime.restore(leases, target)

        runtime.submit(restore, priority=0).result(timeout=120)
        assert torch.equal(expected, layout.pack_pages(target).cpu())
        assert runtime.stats["encoded_pages"] == before == tokens
        assert runtime.stats["reused_host_pages"] == tokens
        for lease in leases:
            lease.close()
        # A later transmission can read exactly the same object bytes.
        leases = runtime.acquire_pages(refs, target)
        assert runtime.stats["encoded_pages"] == tokens
    finally:
        for lease in leases:
            lease.close()
        runtime.close()
        pool.clear()


@pytest.mark.parametrize("tokens", [1, 65, 1025])
def test_forced_lz4_bit_exact_and_bounded(tokens):
    from sglang.srt.kv_compression.layout import KVLayoutAdapter
    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.types import new_page_refs

    device = torch.device("cuda", 0)
    torch.manual_seed(73)
    # Arbitrary bits deliberately exercise expansion without modifying the
    # FORCE policy. These cases validate mechanics, not model compressibility.
    buffers = [
        torch.randint(
            0, 256, (tokens * 2, 4, 256), dtype=torch.uint8, device=device
        ).view(torch.bfloat16)
        for _ in range(4)
    ]
    layout = KVLayoutAdapter(buffers[:2], buffers[2:])
    expected = layout.pack_pages(list(range(tokens))).cpu()
    ready = torch.cuda.Event()
    ready.record()
    runtime = KVCompressionRuntime(layout, "lz4", 256 * 1024**2, force=True)
    leases = runtime.acquire_pages(new_page_refs(tokens), list(range(tokens)), ready)
    try:
        pages = [lease.future.result(timeout=120) for lease in leases]
        assert all(page.encoding == "lz4" for page in pages)
        assert runtime.stats["attempted_encoded_bytes"] == sum(p.nbytes for p in pages)
        runtime.submit(
            lambda: runtime.restore(leases, list(range(tokens, tokens * 2))), 0
        ).result(timeout=120)
        assert torch.equal(
            expected, layout.pack_pages(list(range(tokens, tokens * 2))).cpu()
        )
        assert runtime.stats["peak_bytes"] <= runtime.budget_bytes
    finally:
        for lease in leases:
            lease.close()
        runtime.close()


@pytest.mark.parametrize("tokens", [1, 65, 1025])
def test_forced_lz4_verified_l2_eviction_restore(tokens):
    from sglang.srt.kv_compression.host_io import CompressedHostIO
    from sglang.srt.kv_compression.layout import KVLayoutAdapter
    from sglang.srt.kv_compression.provider import (
        HostEncodedKVProvider,
        RepresentationSpec,
    )
    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.store import CompressedHostKVCache
    from sglang.srt.kv_compression.types import KVVerificationError, new_page_refs
    from sglang.srt.mem_cache.l2_completion import AsyncL2State
    from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine

    torch.manual_seed(73)
    buffers = [
        torch.randint(
            0, 256, (tokens * 2, 8, 256), dtype=torch.uint8, device="cuda:0"
        ).view(torch.bfloat16)
        for _ in range(4)
    ]
    layout = KVLayoutAdapter(buffers[:2], buffers[2:])
    source = list(range(0, tokens * 2, 2))
    target = list(range(1, tokens * 2, 2))[::-1]
    expected = layout.pack_pages(source).cpu()
    runtime = KVCompressionRuntime(
        layout, "lz4", 512 * 1024**2, force=True, verify=True
    )
    pool = CompressedHostKVCache(
        layout.page_bytes,
        64 * 1024**2,
        reservation_bytes=runtime.output_bound,
        verify=True,
    )
    adapter = CompressedHostIO(runtime, pool)
    runtime.provider = HostEncodedKVProvider(
        pool, RepresentationSpec(runtime.layout.tag, runtime.mode)
    )
    pool.io = adapter
    state = AsyncL2State(runtime, pool, runtime.provider)
    engine = L2TransferEngine("kernel")
    handles, refs = pool.alloc(tokens), new_page_refs(tokens)
    leases = []
    ready = torch.cuda.Event()
    ready.record()
    try:
        assert handles is not None
        completion = engine.submit_device_to_host(
            [L2Transfer(pool, None, handles, torch.tensor(source, device="cuda:0"))],
            async_state=state,
            page_refs=refs,
        )
        completion.future.result(timeout=120)
        assert completion.query() and completion.actual_bytes > 0
        assert adapter.stats["published_lz4_pages"] == tokens
        assert adapter.stats["verified_backup_pages"] == tokens
        for tensor in buffers:
            tensor.zero_()
        zeroed = torch.cuda.Event()
        zeroed.record()
        leases = runtime.acquire_pages(refs, target)

        def restore():
            return engine.submit_async_restore(
                state, leases, torch.tensor(target, device="cuda:0")
            )

        restored = restore().future.result(timeout=120)
        assert restored.pages == tokens
        assert restored.actual_bytes == sum(l.future.result().nbytes for l in leases)
        assert restored.logical_bytes == tokens * layout.page_bytes
        assert restored.queue_seconds >= 0 and restored.execution_seconds >= 0
        assert restored.gpu_seconds is not None and restored.gpu_seconds >= 0
        assert torch.equal(expected, layout.pack_pages(target).cpu())
        assert runtime.stats["verified_restore_pages"] == tokens
        assert runtime.stats["encoded_pages"] == tokens
        for lease in leases:
            lease.close()
        # Acquiring for another send uses the same L2 objects, without encoding.
        leases = runtime.acquire_pages(refs, target)
        assert all(l.future.result().encoding == "lz4" for l in leases)
        assert runtime.stats["encoded_pages"] == tokens
        for lease in leases:
            lease.close()
        leases = []
        # A bad expected digest must reject actual restored GPU pages.
        pool.checksums[handles[0].item() & 0xFFFFFFFF, 0] ^= 1
        leases = runtime.acquire_pages(refs, target)
        with pytest.raises(KVVerificationError):
            restore().future.result(timeout=120)
        assert runtime.stats["peak_bytes"] <= runtime.budget_bytes
    finally:
        for lease in leases:
            lease.close()
        state.close()
        pool.clear()


@pytest.mark.parametrize("tokens", [1, 65])
def test_force_lz4_fixed_blocks_verified_restore(tokens):
    """Real nvCOMP, pinned D2H/scatter/gather/H2D and target byte validation."""
    from sglang.srt.kv_compression.host_io import CompressedHostIO
    from sglang.srt.kv_compression.layout import KVLayoutAdapter
    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.store import CompressedHostKVCache, materialize_pages
    from sglang.srt.kv_compression.types import new_page_refs

    device = torch.device("cuda", 0)
    buffers = [
        torch.randn(2 * tokens + 1, 8, 128, device=device, dtype=torch.bfloat16)
        for _ in range(4)
    ]
    layout = KVLayoutAdapter(buffers[:2], buffers[2:])
    runtime = KVCompressionRuntime(
        layout, "lz4", 128 * 1024**2, force=True, verify=True
    )
    pool = CompressedHostKVCache(
        layout.page_bytes,
        64 * 1024**2,
        reservation_bytes=runtime.output_bound,
        verify=True,
    )
    fragmented = pool.alloc(2 * tokens)
    pool.free(fragmented[::2])
    handles = pool.alloc(tokens)
    refs = new_page_refs(tokens)
    source, target = list(range(tokens)), list(range(tokens, 2 * tokens))
    expected = layout.pack_pages(source).cpu()
    ready = torch.cuda.Event()
    ready.record()
    leases = []
    try:
        io = CompressedHostIO(runtime, pool)
        io._write(handles.tolist(), refs, source, ready)
        assert io.stats["published_lz4_pages"] == tokens
        assert io.stats["verified_backup_pages"] == tokens
        for tensor in buffers:
            tensor.zero_()
        ready.record()
        ready.synchronize()
        leases = [pool.acquire(h) for h in handles]
        runtime.submit(lambda: runtime.restore(leases, target), 0).result(timeout=120)
        assert torch.equal(expected, layout.pack_pages(target).cpu())
        assert runtime.stats["verified_restore_pages"] == tokens
        # P/D assembly borrows the same storage and keeps the wire independent.
        stream = torch.cuda.Stream()
        for start in range(0, tokens, 64):
            with (
                torch.cuda.stream(stream),
                materialize_pages(
                    [l.future.result() for l in leases[start : start + 64]]
                ) as pages,
            ):
                wire = [p.data.to(device, non_blocking=True) for p in pages]
                stream.synchronize()
                assert all(torch.equal(w.cpu(), p.data) for w, p in zip(wire, pages))
        pool.free(handles)
        assert pool.retired_bytes > 0
        for l in leases:
            l.close()
        leases = []
        pool.free(fragmented[1::2])
        assert pool.free_block_count == pool.block_count and not pool.has_readers()
    finally:
        for l in leases:
            l.close()
        runtime.close()


def test_gpu_block_write_failure_drains_before_reclaim():
    from unittest.mock import patch

    from sglang.srt.kv_compression.host_io import CompressedHostIO
    from sglang.srt.kv_compression.layout import KVLayoutAdapter
    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.store import CompressedHostKVCache, WriteLease
    from sglang.srt.kv_compression.types import new_page_refs

    buffers = [
        torch.randn(4, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    layout = KVLayoutAdapter(buffers[:2], buffers[2:])
    runtime = KVCompressionRuntime(layout, "lz4", 64 * 1024**2, force=True, verify=True)
    pool = CompressedHostKVCache(
        layout.page_bytes,
        64 * 1024**2,
        reservation_bytes=runtime.output_bound,
        verify=True,
    )
    ready = torch.cuda.Event()
    ready.record()
    h = pool.alloc(2)
    io = CompressedHostIO(runtime, pool)
    try:
        with (
            patch.object(
                WriteLease, "scatter", side_effect=RuntimeError("injected after D2H")
            ),
            pytest.raises(RuntimeError, match="after D2H"),
        ):
            io._write(h.tolist(), new_page_refs(2), [0, 1], ready)
        assert not pool.active_writers and not pool.has_readers()
        pool.free(h)
        h = pool.alloc(2)
        io._write(h.tolist(), new_page_refs(2), [0, 1], ready)
        pool.free(h)
        assert pool.free_block_count == pool.block_count
    finally:
        runtime.close()


# Review regressions ported from the isolated H20 diagnostics.
def review_record(value):
    import json

    print(json.dumps(value))


def review_layout(count):
    from sglang.srt.kv_compression.layout import KVLayoutAdapter

    assert torch.cuda.is_available(), "A GPU skip cannot satisfy this diagnostic"
    # Full Qwen3-8B GQA page size: 36 layers * K/V * 8 * 128 * BF16.
    tensors = [
        torch.zeros((2 * count + 3, 8, 128), dtype=torch.bfloat16, device="cuda:0")
        for _ in range(72)
    ]
    return KVLayoutAdapter(tensors[:36], tensors[36:])


def review_encoded(runtime, indices):
    from sglang.srt.kv_compression.types import EncodedPage, new_page_refs
    from sglang.srt.kv_compression.verification import page_digests

    # The production Host I/O publisher adds these digests; the transient
    # encoder by itself intentionally does not attach L2 verification metadata.
    expected = page_digests(runtime.layout.pack_pages(indices))
    leases = runtime.acquire_pages(new_page_refs(len(indices)), indices)
    try:
        pages = [lease.future.result(timeout=120) for lease in leases]
        # Independent receiver input storage, outside decompression workspace.
        output = [
            EncodedPage(p.data.clone(), p.encoding, p.raw_bytes, digest)
            for p, digest in zip(pages, expected)
        ]
        torch.cuda.synchronize()
        return output
    finally:
        for lease in leases:
            lease.close()


@pytest.mark.parametrize("mode", ["passthrough", "lz4"])
@pytest.mark.parametrize("count", [1, 64, 65, 1024])
def test_decode_output_stays_budgeted_until_delayed_scatter(mode, count):
    import concurrent.futures
    import gc
    import threading

    from sglang.srt.kv_compression.runtime import KVCompressionRuntime

    model = review_layout(count)
    runtime = KVCompressionRuntime(
        model, mode, 512 * 1024**2, force=mode == "lz4", verify=True
    )
    pages = review_encoded(runtime, list(range(0, 2 * count, 2)))
    release, reached = threading.Event(), threading.Event()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    held = []
    owner = decoded = staging = None
    try:
        before = torch.cuda.memory_allocated()
        owner = runtime.submit(lambda: runtime.decode_pages(pages), priority=0).result(
            timeout=120
        )
        decoded = owner.raw
        torch.cuda.synchronize()
        staging = owner.to_staging_order()
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated()

        # A real consumer holds both results before writing non-contiguous KV.
        def scatter():
            held.extend([decoded, staging])
            reached.set()
            if not release.wait(timeout=30):
                raise TimeoutError("Test release signal missing")
            model.unpack_pages(decoded, list(range(1, 2 * count + 1, 2)))
            torch.cuda.synchronize()

        consumer = executor.submit(scatter)
        assert reached.wait(timeout=10)
        snapshot = runtime.snapshot()
        storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in held
        }
        live_result_bytes = sum(storages.values())
        review_record(
            dict(
                case="delayed-scatter",
                mode=mode,
                count=count,
                raw_bytes=count * model.page_bytes,
                held_storage_bytes=live_result_bytes,
                pytorch_allocated_delta=after - before,
                runtime=snapshot,
                encoding={
                    e: sum(p.encoding == e for p in pages) for e in ("raw", "lz4")
                },
                budget_covers_live_output=snapshot["resident_bytes"]
                >= live_result_bytes,
            )
        )
        release.set()
        consumer.result(timeout=30)
        assert torch.equal(model.pack_pages(list(range(1, 2 * count + 1, 2))), decoded)
        assert snapshot["resident_bytes"] >= live_result_bytes, (
            "Decode output/layout storage is live but uncharged"
        )
    finally:
        release.set()
        executor.shutdown(wait=True)
        del held[:]
        decoded = staging = None
        if owner is not None:
            owner.close(torch.cuda.current_stream())
            assert runtime.snapshot()["workspace_bytes"] == 0
        runtime.close()
        gc.collect()
        torch.cuda.empty_cache()


def test_decode_rejects_task_larger_than_workspace():

    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.types import EncodedPage

    model = review_layout(1)
    raw = torch.zeros(model.page_bytes, dtype=torch.uint8, device="cuda:0")
    runtime = KVCompressionRuntime(
        model, "passthrough", model.page_bytes - 1, verify=True
    )
    rejected = False
    try:
        try:
            runtime.submit(
                lambda: runtime.decode_pages([EncodedPage(raw, "raw", raw.numel())]),
                priority=0,
            ).result(timeout=30)
        except Exception as exc:
            from sglang.srt.kv_compression.types import CompressionCapacityError

            if not isinstance(exc, CompressionCapacityError):
                raise
            rejected = True
        review_record(
            dict(
                case="impossible-decode-budget",
                budget=runtime.budget_bytes,
                output_bytes=model.page_bytes,
                rejected=rejected,
                runtime=runtime.snapshot(),
            )
        )
        assert rejected, "Impossible decode task was accepted"
    finally:
        runtime.close()


@pytest.mark.parametrize("mode", ["passthrough", "lz4"])
@pytest.mark.parametrize("where", ["payload", "target"])
def test_actual_byte_corruption_detected_and_next_restore_succeeds(mode, where):
    import concurrent.futures
    from unittest.mock import patch

    from sglang.srt.kv_compression.runtime import KVCompressionRuntime
    from sglang.srt.kv_compression.types import EncodedPage, KVVerificationError, Lease
    from sglang.srt.kv_compression.verification import page_digests

    model = review_layout(1)
    runtime = KVCompressionRuntime(
        model, mode, 512 * 1024**2, force=mode == "lz4", verify=True
    )
    pages = review_encoded(runtime, [0])
    future = concurrent.futures.Future()
    future.set_result(pages[0])
    leases = [Lease(future, lambda: None)]
    triggered = 0
    rejected = False
    try:
        if where == "payload":
            # Mutate bytes without risking malformed nvCOMP metadata: encode
            # changed source bytes but retain the original source digest.
            model.buffers[0][0].view(torch.uint8).flatten()[0] ^= 1
            changed = review_encoded(runtime, [0])[0]
            assert not torch.equal(changed.data, pages[0].data)
            bad = EncodedPage(
                changed.data, changed.encoding, changed.raw_bytes, pages[0].raw_sha256
            )
            bad_future = concurrent.futures.Future()
            bad_future.set_result(bad)
            bad_leases = [Lease(bad_future, lambda: None)]
            triggered += 1
            try:
                runtime.submit(
                    lambda: runtime.restore(bad_leases, [1]), priority=0
                ).result(timeout=30)
            except KVVerificationError:
                rejected = True
            model.buffers[0][0].view(torch.uint8).flatten()[0] ^= 1
            bad_leases[0].close()
        else:
            original = model.unpack_pages

            def corrupt(raw, indices):
                nonlocal triggered
                original(raw, indices)
                model.buffers[0][indices[0]].view(torch.uint8).flatten()[0] ^= 1
                triggered += 1

            with patch.object(model, "unpack_pages", side_effect=corrupt):
                try:
                    runtime.submit(
                        lambda: runtime.restore(leases, [1]), priority=0
                    ).result(timeout=30)
                except KVVerificationError:
                    rejected = True
        failed_snapshot = runtime.snapshot()
        assert rejected and triggered == 1
        assert failed_snapshot["verified_restore_pages"] == 0
        assert failed_snapshot["resident_bytes"] == 0
        runtime.submit(lambda: runtime.restore(leases, [1]), priority=0).result(
            timeout=30
        )
        assert page_digests(model.pack_pages([1])) == [pages[0].raw_sha256]
        review_record(
            dict(
                case="real-byte-corruption",
                mode=mode,
                where=where,
                triggered=triggered,
                rejected=rejected,
                next_restore_passed=True,
                after_failure=failed_snapshot,
                boundary="Isolated CUDA restore; no scheduler/attention or RDMA",
            )
        )
    finally:
        for lease in leases:
            lease.close()
        runtime.close()
