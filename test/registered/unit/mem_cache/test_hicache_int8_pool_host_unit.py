"""Unit tests for the INT8 HiCache host pool (``MHATokenToKVPoolHostINT8``).

Requires CUDA/ROCm: the pool moves bytes through SGLang's JIT HiCache kernels and
pins its arena with the CUDA driver. Run on the pod:

    python -m pytest test/registered/unit/mem_cache/test_hicache_int8_pool_host_unit.py -v

The parts that need no GPU (record codec, staging geometry, growth policy) are
covered by ``test_hicache_int8_codec.py`` and run anywhere.
"""

import os
import unittest
from unittest import mock

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.pool_host import int8_codec as codec
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.mha_int8 import MHATokenToKVPoolHostINT8
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or is_npu()
    or is_xpu()
    or not (is_cuda() or is_hip()),
    reason="INT8 HiCache host pool requires CUDA/ROCm.",
)

DEVICE = "cuda"
LAYER_NUM = 2
HEAD_NUM = 8  # must satisfy codec.check_layout: head_num * head_dim == 1024
HEAD_DIM = 128
POOL_SIZE = 64
PAGE_SIZE = 1


#: The exact bound the format guarantees (see int8_codec docstring).
def error_bound(restored, scales):
    s = scales.float().unsqueeze(-1)
    return (0.5 + 2**-8) * s + 2**-8 * restored.float().abs()


def _make_device_pool(layer_num=LAYER_NUM, size=POOL_SIZE):
    return MHATokenToKVPool(
        size=size,
        page_size=PAGE_SIZE,
        head_num=HEAD_NUM,
        head_dim=HEAD_DIM,
        dtype=torch.bfloat16,
        layer_num=layer_num,
        device=DEVICE,
        enable_memory_saver=False,
    )


#: Pools created during the current test, torn down by _Int8PoolTestCase.
_LIVE_POOLS: list = []


class _Int8PoolTestCase(unittest.TestCase):
    """Base that unregisters every host arena a test created.

    The arena is pinned with cudaHostRegister. Dropping the Python reference is
    not enough: the CUDA registration survives until the buffer is unregistered,
    so when a later test's allocator is handed the same address range,
    cudaHostRegister fails with "part or all of the requested memory range is
    already mapped". That cascaded into 20 spurious failures across the suite,
    masking whatever real problems existed underneath.

    destroy() is idempotent and unregisters kv_buffer, so tearing down here is
    safe even when a test already destroyed its pool explicitly.
    """

    def setUp(self):
        _LIVE_POOLS.clear()

    def tearDown(self):
        for pool in _LIVE_POOLS:
            try:
                pool.destroy()
            except Exception:  # noqa: BLE001 - teardown must not mask a failure
                pass
        _LIVE_POOLS.clear()


def _make_host_pool(device_pool, *, host_size=0, ratio=2.0, layout="layer_first", **kw):
    pool = MHATokenToKVPoolHostINT8(
        device_pool,
        host_to_device_ratio=ratio,
        host_size=host_size,
        page_size=PAGE_SIZE,
        layout=layout,
        pin_memory=True,
        device="cpu",
        allocator_type="default",
        **kw,
    )
    _LIVE_POOLS.append(pool)
    return pool


def _fill(device_pool, *, seed=0, num_layers=None):
    """Fill device K/V with distinguishable, non-degenerate values."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    layers = num_layers if num_layers is not None else device_pool.layer_num
    for layer in range(layers):
        for buf, tag in (
            (device_pool.k_buffer[layer], 0),
            (device_pool.v_buffer[layer], 1),
        ):
            shape = buf.shape
            values = torch.randn(shape, generator=g) * (1.0 + layer) + tag
            buf.copy_(values.to(torch.bfloat16))


class TestSizing(_Int8PoolTestCase):
    def test_size_per_token_is_the_encoded_size(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        self.assertEqual(host_pool.size_per_token, codec.bytes_per_token(LAYER_NUM))
        self.assertEqual(host_pool.size_per_token, 2 * LAYER_NUM * 1152)
        # Baseline for this geometry: 2 (K,V) * layers * heads * dim * 2 bytes.
        baseline = 2 * LAYER_NUM * HEAD_NUM * HEAD_DIM * 2
        self.assertEqual(baseline, 4096 * LAYER_NUM)
        self.assertLess(host_pool.size_per_token, baseline)
        self.assertAlmostEqual(baseline / host_pool.size_per_token, 1.7777778, places=6)
        host_pool.destroy()

    def test_fixed_host_size_yields_more_tokens_than_bf16(self):
        """The headline claim: same --hicache-size, ~1.78x the token capacity."""
        device_pool = _make_device_pool()
        int8_pool = _make_host_pool(device_pool, host_size=1)
        bf16_pool = MHATokenToKVPoolHost(
            device_pool,
            host_to_device_ratio=2.0,
            host_size=1,
            page_size=PAGE_SIZE,
            layout="layer_first",
            pin_memory=True,
            device="cpu",
        )
        # The baseline pool pins its arena too, so it needs the same teardown.
        _LIVE_POOLS.append(bf16_pool)
        self.assertEqual(bf16_pool.size_per_token, 4096 * LAYER_NUM)
        self.assertEqual(int8_pool.size_per_token, 2304 * LAYER_NUM)
        # Ratio holds to within the +1 page slack.
        ratio = int8_pool.size / bf16_pool.size
        self.assertAlmostEqual(ratio, 4096 / 2304, delta=0.01)
        int8_pool.destroy()
        bf16_pool.destroy()

    def test_arena_shape_is_layer_first_encoded(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        self.assertEqual(
            host_pool.kv_buffer.shape,
            (2, LAYER_NUM, host_pool.size, codec.ROW_BYTES),
        )
        self.assertEqual(host_pool.kv_buffer.dtype, torch.uint8)
        # k_buffer/v_buffer are the two halves.
        self.assertEqual(
            host_pool.k_buffer.shape, (LAYER_NUM, host_pool.size, codec.ROW_BYTES)
        )
        host_pool.destroy()

    def test_layer_refs_are_contiguous_encoded_rows(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        for layer in range(LAYER_NUM):
            ref = host_pool.k_data_refs[layer]
            self.assertTrue(ref.is_contiguous())
            self.assertEqual(ref.shape[-1], codec.ROW_BYTES)
            self.assertEqual(ref.data_ptr() % codec.ALIGNMENT_BYTES, 0)
        host_pool.destroy()


class TestTransferRoundtrip(_Int8PoolTestCase):
    def test_d2h_then_h2d_reconstructs_within_bound(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)

        origin = {
            layer: (
                device_pool.k_buffer[layer].clone(),
                device_pool.v_buffer[layer].clone(),
            )
            for layer in range(LAYER_NUM)
        }

        src = torch.tensor([1, 5, 9, 20, 33], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1, 2, 3, 4], device=DEVICE, dtype=torch.int64)

        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        # Wipe the device rows so a stale read cannot pass the test.
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
            device_pool.v_buffer[layer].zero_()

        for layer in range(LAYER_NUM):
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()

        for layer in range(LAYER_NUM):
            for buf, orig, label in (
                (device_pool.k_buffer[layer], origin[layer][0], "K"),
                (device_pool.v_buffer[layer], origin[layer][1], "V"),
            ):
                got = buf[src].float()
                want = orig[src].float()
                scales = codec.compute_scales(
                    orig[src].reshape(len(src), HEAD_NUM, HEAD_DIM)
                )
                bound = error_bound(got.reshape(len(src), HEAD_NUM, HEAD_DIM), scales)
                err = (
                    got.reshape(len(src), HEAD_NUM, HEAD_DIM)
                    - want.reshape(len(src), HEAD_NUM, HEAD_DIM)
                ).abs()
                violations = int((err > bound).sum())
                self.assertEqual(
                    violations, 0, f"layer {layer} {label}: {violations} exceed bound"
                )

    def test_encoded_bytes_actually_shrink(self):
        """Prove the host arena holds compressed data, not raw BF16."""
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([2, 4, 6], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([10, 11, 12], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        # k_data_refs are views of the CPU host arena, so they must be indexed
        # with CPU indices. `src`/`dst` live on the GPU because the mover
        # requires device index tensors.
        raw = device_pool.k_buffer[0][src]
        stored = host_pool.k_data_refs[0][dst.cpu()]
        self.assertEqual(stored.dtype, torch.uint8)
        self.assertEqual(stored.numel(), len(src) * codec.ROW_BYTES)

        # Compare BYTES, not element counts. `raw` is bf16, so its element count
        # is half its byte count; comparing stored.numel() (uint8 bytes) against
        # raw.numel() (bf16 elements) mixed units and made the encoded arena look
        # larger than the raw one.
        stored_bytes = stored.numel() * stored.element_size()
        raw_bytes = raw.numel() * raw.element_size()
        self.assertEqual(raw_bytes, len(src) * HEAD_NUM * HEAD_DIM * 2)
        self.assertLess(
            stored_bytes,
            raw_bytes,
            f"encoded {stored_bytes} B must be smaller than raw {raw_bytes} B",
        )
        self.assertAlmostEqual(
            raw_bytes / stored_bytes,
            2048 / 1152,
            places=6,
            msg="compression per row should be exactly one bf16 row over one record",
        )
        # Decoding the arena must reproduce the device row within bound, which
        # proves it really is an encoding rather than padding.
        #
        # The comparison happens entirely on CPU. `stored` is a view of the CPU
        # host arena, so `decoded` is CPU, while `raw` was gathered from the GPU
        # device pool -- folding them together directly raises "Expected all
        # tensors to be on the same device, but found at least two devices,
        # cuda:0 and cpu!". Moving `raw` once keeps the arithmetic on one device.
        raw_cpu = raw.cpu()
        decoded = codec.decode_records(
            stored, head_num=HEAD_NUM, head_dim=HEAD_DIM, dtype=torch.bfloat16
        )
        self.assertEqual(decoded.device.type, "cpu")
        scales = codec.compute_scales(raw_cpu)
        bound = error_bound(decoded, scales)
        err = (decoded.float() - raw_cpu.float()).abs()
        self.assertEqual(int((err > bound).sum()), 0)

    def test_padding_bytes_are_zero_in_the_arena(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([1, 2], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        # Host arena is on CPU; index with CPU indices.
        padding = host_pool.k_data_refs[0][dst.cpu()][
            :, codec.PAYLOAD_BYTES + codec.SCALE_BYTES :
        ]
        self.assertTrue(bool((padding == 0).all()))

    def test_all_zero_kv_round_trips_exactly(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
            device_pool.v_buffer[layer].zero_()
        src = torch.tensor([3, 7], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].fill_(float("nan"))
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        for layer in range(LAYER_NUM):
            got = device_pool.k_buffer[layer][src]
            self.assertTrue(bool((got == 0).all()), f"layer {layer} not exactly zero")

    def test_multi_layer_pool_round_trips(self):
        """The all-layer mover must handle more than the 2-layer unit case."""
        device_pool = _make_device_pool(layer_num=36, size=128)
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool, num_layers=36, seed=3)
        src = torch.tensor([1, 2, 3, 4], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0, 1, 2, 3], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        for layer in range(36):
            device_pool.k_buffer[layer].zero_()
        for layer in range(36):
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        for layer in range(36):
            row = device_pool.k_buffer[layer][src].float()
            self.assertTrue(bool(torch.isfinite(row).all()), f"layer {layer}")
            self.assertGreater(float(row.abs().max()), 0.0, f"layer {layer} is empty")

    def test_d2h_then_h2d_with_disjoint_index_ranges(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([0, 1, 2], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([40, 41, 42], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
        for layer in range(LAYER_NUM):
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        self.assertGreater(float(device_pool.k_buffer[0][src].float().abs().max()), 0.0)


class TestStagingGrowth(_Int8PoolTestCase):
    def test_large_transfer_grows_staging_and_still_round_trips(self):
        device_pool = _make_device_pool(size=4096)
        host_pool = _make_host_pool(device_pool)
        initial_capacity = host_pool._d2h.capacity
        _fill(device_pool)
        count = initial_capacity + 100  # force a growth
        src = torch.arange(count, device=DEVICE, dtype=torch.int64)
        dst = torch.arange(count, device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        self.assertGreater(host_pool._d2h.capacity, initial_capacity)
        for layer in range(LAYER_NUM):
            device_pool.k_buffer[layer].zero_()
        for layer in range(LAYER_NUM):
            host_pool.load_to_device_per_layer(device_pool, dst, src, layer, "kernel")
        torch.cuda.synchronize()
        self.assertGreater(float(device_pool.k_buffer[0][src].float().abs().max()), 0.0)

    def test_d2h_growth_does_not_reallocate_h2d_staging(self):
        """Regression: a backup must not free storage another stream is reading.

        The two directions run on independent streams and can be in flight at
        once. If a D2H-triggered growth reallocated the H2D buffers, an in-flight
        load would read freed memory. Each direction may grow only for its own
        transfers.
        """
        device_pool = _make_device_pool(size=4096)
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)

        h2d_k_ptr = host_pool._h2d.k.data_ptr()
        h2d_v_ptr = host_pool._h2d.v.data_ptr()
        h2d_capacity = host_pool._h2d.capacity

        count = host_pool._d2h.capacity + 100  # force D2H growth
        src = torch.arange(count, device=DEVICE, dtype=torch.int64)
        dst = torch.arange(count, device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        self.assertGreater(host_pool._d2h.capacity, h2d_capacity)
        self.assertEqual(
            host_pool._h2d.k.data_ptr(),
            h2d_k_ptr,
            "D2H growth must not move the H2D K staging buffer",
        )
        self.assertEqual(
            host_pool._h2d.v.data_ptr(),
            h2d_v_ptr,
            "D2H growth must not move the H2D V staging buffer",
        )
        self.assertEqual(host_pool._h2d.capacity, h2d_capacity)

    def test_h2d_growth_does_not_reallocate_d2h_staging(self):
        """The symmetric guard: a load must not free a backup's staging."""
        device_pool = _make_device_pool(size=8192)
        host_pool = _make_host_pool(device_pool)

        d2h_k_ptr = host_pool._d2h.k.data_ptr()
        d2h_capacity = host_pool._d2h.capacity
        d2h_tables = [int(v) for v in host_pool._d2h_k_src_ptrs]

        count = host_pool._h2d.capacity + 100
        host_pool._ensure_h2d_capacity(count)

        self.assertGreater(host_pool._h2d.capacity, count - 1)
        self.assertEqual(
            host_pool._d2h.k.data_ptr(),
            d2h_k_ptr,
            "H2D growth must not move the D2H K staging buffer",
        )
        self.assertEqual(host_pool._d2h.capacity, d2h_capacity)
        self.assertEqual(
            [int(v) for v in host_pool._d2h_k_src_ptrs],
            d2h_tables,
            "H2D growth must not invalidate the D2H staging pointer tables",
        )

    def test_index_lists_are_per_direction(self):
        """Each direction owns its index list, so growth cannot strand the other.

        Regression: a single shared list meant an H2D-triggered extension could
        drop the old tensor while the D2H stream still had it referenced. Separate
        lists remove the question entirely.
        """
        device_pool = _make_device_pool(size=8192)
        host_pool = _make_host_pool(device_pool)
        self.assertIsNot(host_pool._d2h_indices, host_pool._h2d_indices)

        d2h_before = host_pool._d2h_indices
        big = host_pool._h2d.capacity + 500
        host_pool._ensure_h2d_capacity(big)

        # The D2H list is untouched, and the H2D list covers the new size.
        self.assertIs(host_pool._d2h_indices, d2h_before)
        view = host_pool._h2d_index_view(big)
        self.assertGreaterEqual(view.numel(), big)
        self.assertEqual(int(view[-1]), big - 1)

    def test_d2h_growth_extends_only_the_d2h_index_list(self):
        device_pool = _make_device_pool(size=8192)
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        h2d_before = host_pool._h2d_indices

        count = host_pool._d2h.capacity + 100
        src = torch.arange(count, device=DEVICE, dtype=torch.int64)
        dst = torch.arange(count, device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        self.assertIs(host_pool._h2d_indices, h2d_before)
        self.assertGreaterEqual(host_pool._d2h_index_view(count).numel(), count)

    def test_pointer_tables_follow_growth(self):
        device_pool = _make_device_pool(size=4096)
        host_pool = _make_host_pool(device_pool)
        count = host_pool._d2h.capacity + 1
        src = torch.arange(count, device=DEVICE, dtype=torch.int64)
        dst = torch.arange(count, device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()
        expected = [t.data_ptr() for t in host_pool._d2h.k_layer_views(count)]
        self.assertEqual([int(v) for v in host_pool._d2h_k_src_ptrs], expected)


class TestAllocFree(_Int8PoolTestCase):
    def test_alloc_free_reuse(self):
        """free() must make slots allocatable again when the list is exhausted.

        Freed slots are deliberately NOT handed straight back: alloc() takes from
        the FRONT of free_slots while free() appends to release_slots, merged
        onto the END only when the free list runs short
        (pool_host/base.py: _merge_release_slots). So immediately after freeing
        [0,1,2,3] the next alloc correctly returns [4,5,6,7] -- asserting that
        the same slots come back contradicts the contract.

        The property that actually matters is reclamation: once nothing else is
        free, the released slots must be reusable. Exhausting the pool first
        makes that observable.
        """
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)

        total = host_pool.available_size()
        self.assertGreater(total, 8, "pool too small for this test")

        # Take everything, so any later allocation must come from the release list.
        everything = host_pool.alloc(total)
        self.assertIsNotNone(everything)
        self.assertEqual(host_pool.available_size(), 0)
        self.assertIsNone(host_pool.alloc(PAGE_SIZE), "pool should be exhausted")

        # Release the first four slots and reclaim them.
        released = everything[:4].clone()
        self.assertEqual(host_pool.free(released), 4)
        self.assertEqual(host_pool.available_size(), 4)

        again = host_pool.alloc(4)
        self.assertIsNotNone(again, "released slots must be reallocatable")
        self.assertEqual(
            sorted(again.tolist()),
            sorted(released.tolist()),
            "with nothing else free, the released slots are the only candidates",
        )
        host_pool.destroy()

    def test_double_free_is_detected(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        slots = host_pool.alloc(PAGE_SIZE * 2)
        host_pool.free(slots)
        with self.assertRaises(AssertionError):
            host_pool.free(slots)
        host_pool.destroy()

    def test_alloc_beyond_capacity_returns_none(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        self.assertIsNone(host_pool.alloc(host_pool.logical_size + PAGE_SIZE))
        host_pool.destroy()


class TestStoragePages(_Int8PoolTestCase):
    def test_data_page_round_trip(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        _fill(device_pool)
        src = torch.tensor([5], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0], device=DEVICE, dtype=torch.int64)
        host_pool.backup_from_device_all_layer(device_pool, dst, src, "kernel")
        torch.cuda.synchronize()

        page = host_pool.get_data_page(0, flat=True)
        self.assertEqual(page.numel(), 2 * LAYER_NUM * codec.ROW_BYTES)
        self.assertEqual(page.dtype, torch.uint8)

        host_pool.set_from_flat_data_page(1, page)
        self.assertTrue(
            torch.equal(
                host_pool.get_data_page(1, flat=True),
                host_pool.get_data_page(0, flat=True),
            )
        )
        host_pool.destroy()

    def test_dummy_page_has_encoded_size(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        dummy = host_pool.get_dummy_flat_data_page()
        self.assertEqual(dummy.numel(), 2 * LAYER_NUM * codec.ROW_BYTES)
        self.assertTrue(bool((dummy == 0).all()))
        host_pool.destroy()

    def test_l3_meta_is_rejected(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        for call in (
            lambda: host_pool.get_page_buffer_meta(torch.tensor([0])),
            lambda: host_pool.get_split_heads_page_buffer_meta(torch.tensor([0]), 2),
        ):
            with self.assertRaises(NotImplementedError):
                call()
        host_pool.destroy()


class TestFailFast(_Int8PoolTestCase):
    """Every unsupported configuration must raise at construction, not corrupt
    generations later."""

    def test_rejects_non_layer_first_layout(self):
        device_pool = _make_device_pool()
        with self.assertRaises(NotImplementedError) as ctx:
            _make_host_pool(device_pool, layout="page_first")
        self.assertIn("layer_first", str(ctx.exception))

    def test_rejects_page_size_above_one(self):
        device_pool = _make_device_pool()
        with self.assertRaises(NotImplementedError) as ctx:
            MHATokenToKVPoolHostINT8(
                device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=16,
                layout="layer_first",
                pin_memory=True,
                device="cpu",
            )
        self.assertIn("page-size", str(ctx.exception))

    def test_rejects_quantized_device_pool(self):
        device_pool = _make_device_pool()
        # is_quantized_kv_cache IS a property on KVCache (memory_pool.py:2147),
        # so unlike v_head_dim and start_layer this one must be patched on the
        # class; there is no instance attribute to shadow.
        with mock.patch.object(
            type(device_pool), "is_quantized_kv_cache", property(lambda self: True)
        ):
            with self.assertRaises(NotImplementedError) as ctx:
                _make_host_pool(device_pool)
        self.assertIn("quantized", str(ctx.exception))

    def test_rejects_mtp_draft_pools(self):
        device_pool = _make_device_pool()
        with self.assertRaises(NotImplementedError) as ctx:
            _make_host_pool(device_pool, mtp_draft_device_pools=(device_pool,))
        self.assertIn("MTP", str(ctx.exception))

    def test_rejects_asymmetric_head_dims(self):
        device_pool = _make_device_pool()
        # v_head_dim is an INSTANCE attribute (memory_pool.py:2018), not a class
        # one, so mock.patch.object(type(pool), ...) raises AttributeError. Patch
        # the instance instead.
        with mock.patch.object(device_pool, "v_head_dim", 64):
            with self.assertRaises(NotImplementedError) as ctx:
                _make_host_pool(device_pool)
        self.assertIn("symmetric", str(ctx.exception))

    def test_rejects_wrong_head_geometry(self):
        """A local KV head count the fixed record cannot express is rejected.

        head_num=4 gives a 512-byte payload against the record's fixed 1024, so
        the pool must refuse. The TP guard fires first and raises
        NotImplementedError naming the restriction; codec.check_layout would
        raise ValueError for the same pool, which is why both are accepted here.
        """
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            head_num=4,  # 4 * 128 = 512 != 1024
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
        )
        with self.assertRaises((NotImplementedError, ValueError)) as ctx:
            _make_host_pool(pool)
        message = str(ctx.exception)
        self.assertTrue(
            "TP=1" in message or "payload" in message,
            f"unhelpful rejection message: {message}",
        )

    def test_rejects_non_two_byte_dtype(self):
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            head_num=HEAD_NUM,
            head_dim=HEAD_DIM,
            dtype=torch.float32,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
        )
        with self.assertRaises((NotImplementedError, ValueError)):
            _make_host_pool(pool)

    def test_h2d_mover_sees_one_dtype_on_both_sides(self):
        """run_one() binds a single SymbolicDType across src and dst.

        Regression: the H2D move once viewed only the *destination* as bf16 while
        leaving the uint8 arena as the source, which the kernel rejects. The move
        is now a straight uint8 byte copy at element_dim = ROW_BYTES, so both
        sides agree on dtype and the byte width is unchanged.
        """
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        num_tokens = 4

        # Exactly the tensors the H2D call passes to the mover.
        dst = host_pool._h2d.layer_k(0, num_tokens)
        src = host_pool.k_data_refs[0]

        self.assertEqual(dst.dtype, src.dtype, "H2D src/dst dtypes must match")
        self.assertEqual(dst.dtype, torch.uint8)

        dv = dst.view(-1, codec.ROW_BYTES)
        sv = src.view(-1, codec.ROW_BYTES)
        element_size = codec.ROW_BYTES * dv.element_size()
        self.assertEqual(element_size, codec.ROW_BYTES)
        self.assertEqual(element_size, 1152)
        # Element size must be a 128-byte multiple for the JIT kernel.
        self.assertEqual(element_size % codec.ALIGNMENT_BYTES, 0)
        # Per-token stride must be one full record on both sides.
        self.assertEqual(dv.stride()[0] * dv.element_size(), codec.ROW_BYTES)
        self.assertEqual(sv.stride()[0] * sv.element_size(), codec.ROW_BYTES)

    def test_accepts_a_normal_device_pool(self):
        """Regression: end_layer is INCLUSIVE, so a full pool reports layer_num-1.

        KVCache sets ``end_layer = end_layer or layer_num - 1``, and
        MHATokenToKVPool.__init__ does not accept start_layer/end_layer at all.
        An earlier version of the coverage check demanded end_layer == layer_num
        and rejected every ordinary pool.
        """
        device_pool = _make_device_pool()
        self.assertEqual(device_pool.start_layer, 0)
        self.assertEqual(device_pool.end_layer, LAYER_NUM - 1)
        host_pool = _make_host_pool(device_pool)
        self.assertEqual(host_pool.layer_num, LAYER_NUM)
        host_pool.destroy()

    def test_rejects_a_partial_device_pool(self):
        """A pool that does not start at layer 0 cannot be backed up wholesale."""
        device_pool = _make_device_pool()
        # start_layer is an INSTANCE attribute too; patch the instance.
        with mock.patch.object(device_pool, "start_layer", 2):
            with self.assertRaises(NotImplementedError) as ctx:
                _make_host_pool(device_pool)
        self.assertIn("covering every layer", str(ctx.exception))

    def test_rejects_tp_greater_than_one_by_name(self):
        """TP>1 must fail with a legible message, not a payload mismatch.

        The record is fixed at 8 local KV heads; at TP=2 there are 4, so the
        whole format changes. The error should say so.
        """
        pool = MHATokenToKVPool(
            size=POOL_SIZE,
            page_size=PAGE_SIZE,
            head_num=4,  # TP=2 on Qwen3-8B: 8 KV heads / 2
            head_dim=HEAD_DIM,
            dtype=torch.bfloat16,
            layer_num=LAYER_NUM,
            device=DEVICE,
            enable_memory_saver=False,
        )
        with self.assertRaises(NotImplementedError) as ctx:
            _make_host_pool(pool)
        message = str(ctx.exception)
        self.assertIn("TP=1", message)
        self.assertIn("local KV heads", message)

    def test_rejects_unknown_io_backend_at_transfer_time(self):
        device_pool = _make_device_pool()
        host_pool = _make_host_pool(device_pool)
        src = torch.tensor([1], device=DEVICE, dtype=torch.int64)
        dst = torch.tensor([0], device=DEVICE, dtype=torch.int64)
        with self.assertRaises(NotImplementedError):
            host_pool.backup_from_device_all_layer(device_pool, dst, src, "direct")
        with self.assertRaises(NotImplementedError):
            host_pool.load_to_device_per_layer(device_pool, dst, src, 0, "direct")
        host_pool.destroy()


class TestDispatch(_Int8PoolTestCase):
    def test_env_flag_selects_int8_pool(self):
        device_pool = _make_device_pool()
        with mock.patch.dict("os.environ", {"SGLANG_EXPERIMENTAL_HICACHE_INT8": "1"}):
            self.assertIs(get_mha_host_pool_cls(device_pool), MHATokenToKVPoolHostINT8)
        with mock.patch.dict("os.environ", {}, clear=False):
            os.environ.pop("SGLANG_EXPERIMENTAL_HICACHE_INT8", None)
            self.assertIs(get_mha_host_pool_cls(device_pool), MHATokenToKVPoolHost)

    def test_flag_defaults_to_off(self):
        device_pool = _make_device_pool()
        saved = os.environ.pop("SGLANG_EXPERIMENTAL_HICACHE_INT8", None)
        try:
            self.assertIs(get_mha_host_pool_cls(device_pool), MHATokenToKVPoolHost)
        finally:
            if saved is not None:
                os.environ["SGLANG_EXPERIMENTAL_HICACHE_INT8"] = saved


if __name__ == "__main__":
    unittest.main()
