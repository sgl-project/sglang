"""HiCache under decode context parallelism (DCP): host-pool index math.

Under DCP the radix/controller layer works in a widened logical index space
(page_size * dcp_size wide pages, dcp_size * physical capacity), while each
rank's device and host buffers only materialize the owned 1/dcp_size token
shard (owner rule: index % dcp_size == dcp_rank, physical row = index //
dcp_size — the same rule the device-side KV write and page-table kernels
use). These tests cover the translation helper, the logical/physical host
pool sizing, and that the transfer entry points hand *physical* rows to the
kernels.
"""

import tempfile
import unittest
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZE = 8
PHYSICAL_PAGE = 64
WIDENED_PAGE = PHYSICAL_PAGE * DCP_SIZE


def _fake_mla_device_pool(size: int = 1024) -> SimpleNamespace:
    return SimpleNamespace(
        size=size,
        # Match KVCache's default: this static pool's size already counts tokens.
        host_capacity_tokens=None,
        store_dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        start_layer=0,
        end_layer=1,
        device="cpu",
        layers_to_capture=None,
        layer_shard_enabled=False,
    )


def _make_host_pool(
    dcp_rank: int,
    device_size: int = 1024,
    dcp_size: int = DCP_SIZE,
    layout: str = "layer_first",
    dtype: torch.dtype = torch.float16,
) -> MLATokenToKVPoolHost:
    device = _fake_mla_device_pool(device_size)
    device.store_dtype = dtype
    return MLATokenToKVPoolHost(
        device,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=PHYSICAL_PAGE * dcp_size,
        layout=layout,
        pin_memory=False,
        device="cpu",
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
    )


class TestDcpKernelIndices(CustomTestCase):
    def test_identity_without_dcp(self):
        indices = torch.arange(37)
        self.assertIs(maybe_dcp_kernel_indices(indices, 1, 0), indices)

    def test_aligned_page_translates_to_full_physical_page(self):
        # One widened page starting at logical 512 covers physical rows
        # 64..127 on every rank.
        indices = torch.arange(WIDENED_PAGE, 2 * WIDENED_PAGE)
        for rank in range(DCP_SIZE):
            out = maybe_dcp_kernel_indices(indices, DCP_SIZE, rank)
            torch.testing.assert_close(
                out, torch.arange(PHYSICAL_PAGE, 2 * PHYSICAL_PAGE)
            )

    def test_matches_owner_rule_on_merged_unordered_pages(self):
        # Concatenation of non-adjacent widened pages in arbitrary order, as
        # produced by merged CacheOperations after allocator churn.
        pages = [3, 0, 5]
        indices = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in pages]
        )
        for rank in range(DCP_SIZE):
            out = maybe_dcp_kernel_indices(indices, DCP_SIZE, rank)
            expected = (
                indices[indices % DCP_SIZE == rank] // DCP_SIZE
            )  # owner rule, same as filter_dcp_local_kv_indices
            torch.testing.assert_close(out, expected)
            self.assertEqual(out.numel() * DCP_SIZE, indices.numel())

    def test_positional_residue_pairing_survives_host_sort(self):
        # move_indices (direct/layer_first) sorts host indices and permutes
        # device indices to match. Independent residue filtering of both
        # tensors must keep the same token positions on every rank.
        g = torch.Generator().manual_seed(0)
        host_pages = [7, 2]
        device_pages = [1, 4]
        host = torch.cat(
            [torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE) for p in host_pages]
        )
        device = torch.cat(
            [
                torch.arange(p * WIDENED_PAGE, (p + 1) * WIDENED_PAGE)
                for p in device_pages
            ]
        )
        # token identity: position i pairs host[i] <-> device[i]
        perm = torch.randperm(host.numel(), generator=g)
        # sort host as move_indices does, permuting device alongside
        host_sorted, order = host[perm].sort()
        device_matched = device[perm][order]
        for rank in range(DCP_SIZE):
            host_mask = host_sorted % DCP_SIZE == rank
            device_mask = device_matched % DCP_SIZE == rank
            # same positions selected on both sides -> pairing preserved
            torch.testing.assert_close(host_mask, device_mask)
            self.assertEqual(
                maybe_dcp_kernel_indices(host_sorted, DCP_SIZE, rank).numel(),
                host.numel() // DCP_SIZE,
            )


class TestHostPoolSizingUnderDcp(CustomTestCase):
    def test_logical_and_physical_sizing(self):
        pool = _make_host_pool(dcp_rank=3)
        # kernel-facing page is physical
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_page_size, WIDENED_PAGE)
        # physical rows = ratio * device physical size, page aligned
        self.assertEqual(pool.size, pool.page_num * PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size * DCP_SIZE)
        # buffers materialize physical rows only
        self.assertEqual(pool.kv_buffer.shape[1], pool.size)
        # allocator surface is logical
        self.assertEqual(pool.free_slots.numel(), pool.logical_size)
        self.assertEqual(pool.mem_state.numel(), pool.logical_size)

    def test_alloc_is_widened_page_granular(self):
        pool = _make_host_pool(dcp_rank=0)
        out = pool.alloc(WIDENED_PAGE)
        self.assertEqual(out.numel(), WIDENED_PAGE)
        with self.assertRaises(AssertionError):
            pool.alloc(PHYSICAL_PAGE)  # not a multiple of the widened page

    def test_non_dcp_pool_unchanged(self):
        pool = MLATokenToKVPoolHost(
            _fake_mla_device_pool(),
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PHYSICAL_PAGE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
        )
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size)
        self.assertEqual(pool.logical_page_size, PHYSICAL_PAGE)


class TestTransferEntryPointsTranslate(CustomTestCase):
    def _run_backup(self, pool, host_indices, device_indices):
        device_pool = SimpleNamespace(
            data_ptrs=torch.zeros(2, dtype=torch.uint64),
            kv_buffer=[torch.zeros(1)] * 2,
        )
        # create=True: mla.py imports the kernel only under `if _is_cuda or
        # _is_hip`, so the name is absent on the CPU runner this test targets.
        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mla.transfer_kv_all_layer_mla",
            create=True,
        ) as kernel:
            pool.can_use_jit = False
            pool.can_use_write_back_jit = False
            with mock.patch.object(
                MLATokenToKVPoolHost, "_is_device_layer_sharded", return_value=False
            ):
                pool.backup_from_device_all_layer(
                    device_pool, host_indices, device_indices, io_backend="kernel"
                )
            return kernel.call_args.kwargs

    def test_backup_receives_physical_rows(self):
        pool = _make_host_pool(dcp_rank=5)
        logical = torch.arange(2 * WIDENED_PAGE)
        kwargs = self._run_backup(pool, logical, logical.clone())
        expected = torch.arange(2 * PHYSICAL_PAGE)
        torch.testing.assert_close(kwargs["src_indices"], expected)
        torch.testing.assert_close(kwargs["dst_indices"], expected)


class TestDcpStoragePages(CustomTestCase):
    def test_file_round_trip_to_different_allocations(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.override(directory),
        ):
            for dcp_size, layout, kv_dtype in product(
                (2, 4),
                ("layer_first", "page_first", "page_first_direct"),
                (torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2),
            ):
                dtype = (
                    torch.uint8
                    if kv_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                    else kv_dtype
                )
                for rank in range(dcp_size):
                    with self.subTest(
                        dcp_size=dcp_size, rank=rank, layout=layout, dtype=kv_dtype
                    ):
                        source = _make_host_pool(
                            rank, dcp_size=dcp_size, layout=layout, dtype=dtype
                        )
                        target = _make_host_pool(
                            rank, dcp_size=dcp_size, layout=layout, dtype=dtype
                        )
                        values = (
                            torch.arange(source.kv_buffer.numel()) * 13 + rank * 17
                        ) % 251
                        source.kv_buffer.copy_(values.reshape(source.kv_buffer.shape))
                        target.kv_buffer.fill_(255)
                        expected = target.kv_buffer.clone()
                        backend = HiCacheFile(
                            HiCacheStorageConfig(
                                tp_rank=rank,
                                tp_size=dcp_size,
                                pp_rank=0,
                                pp_size=1,
                                attn_cp_rank=0,
                                attn_cp_size=1,
                                is_mla_model=True,
                                enable_storage_metrics=False,
                                is_page_first_layout=layout == "page_first",
                                model_name="page-test",
                                dcp_size=dcp_size,
                                dcp_rank=rank,
                                logical_page_size=64 * dcp_size,
                                kv_cache_dtype=kv_dtype,
                                host_layout=layout,
                                extra_config={
                                    "max_size": "0",
                                    "min_free_space": "0",
                                    "enable_metadata_cache": False,
                                },
                            )
                        )

                        def page_view(buffer, page):
                            if layout == "layer_first":
                                return buffer[:, page * 64 : (page + 1) * 64]
                            if layout == "page_first":
                                return buffer[page * 64 : (page + 1) * 64]
                            return buffer[page : page + 1]

                        for source_page, target_page in ((3, 5), (1, 0)):
                            key = f"page-{source_page}"
                            payload = source.get_data_page(source_page * 64 * dcp_size)
                            expected_bytes = (
                                64 * 2 * 12 * source.kv_buffer.element_size()
                            )
                            self.assertEqual(
                                payload.numel() * payload.element_size(), expected_bytes
                            )
                            torch.testing.assert_close(
                                payload,
                                page_view(source.kv_buffer, source_page).flatten(),
                            )
                            self.assertTrue(backend.set(key, payload))
                            path = (
                                Path(directory)
                                / f"{backend._get_suffixed_key(key)}.bin"
                            )
                            self.assertEqual(path.stat().st_size, expected_bytes)
                            restored = backend.get(
                                key, target.get_dummy_flat_data_page()
                            )
                            target.set_from_flat_data_page(
                                target_page * 64 * dcp_size, restored
                            )
                            page_view(expected, target_page).copy_(
                                page_view(source.kv_buffer, source_page)
                            )
                        torch.testing.assert_close(
                            target.kv_buffer, expected, rtol=0, atol=0
                        )

    def test_invalid_logical_starts_are_rejected(self):
        pool = _make_host_pool(1, dcp_size=2, layout="page_first")
        for index in (-128, 1, 64):
            with self.subTest(index=index), self.assertRaises(ValueError):
                pool.get_data_page(index)
            with self.subTest(index=index), self.assertRaises(ValueError):
                pool.set_from_flat_data_page(index, pool.get_dummy_flat_data_page())
        with self.assertRaises(IndexError):
            pool.get_data_page(pool.logical_size)
        pointers, sizes = pool.get_page_buffer_meta(torch.arange(128))
        self.assertEqual(pointers, [pool.kv_buffer.data_ptr()])
        self.assertEqual(sizes, [64 * 2 * 12 * pool.kv_buffer.element_size()])


if __name__ == "__main__":
    unittest.main()
