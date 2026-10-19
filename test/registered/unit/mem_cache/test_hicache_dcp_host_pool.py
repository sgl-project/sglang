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

import unittest
from unittest import mock

import torch

from sglang.srt.layers.dcp.layout import maybe_dcp_kernel_indices
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

DCP_SIZE = 8
PHYSICAL_PAGE = 64
WIDENED_PAGE = PHYSICAL_PAGE * DCP_SIZE


def _fake_mla_device_pool(size: int = 1024) -> MLATokenToKVPool:
    return MLATokenToKVPool(
        size=size,
        page_size=PHYSICAL_PAGE,
        dtype=torch.float16,
        kv_lora_rank=8,
        qk_rope_head_dim=4,
        layer_num=2,
        device="cpu",
        enable_memory_saver=False,
    )


def _make_host_pool(dcp_rank: int, device_size: int = 1024) -> MLATokenToKVPoolHost:
    device_pool = _fake_mla_device_pool(device_size)
    return MLATokenToKVPoolHost(
        device_pool.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=WIDENED_PAGE,
        layout="layer_first",
        pin_memory=False,
        device="cpu",
        dcp_size=DCP_SIZE,
        dcp_rank=dcp_rank,
        device_capacity_tokens=(
            device_pool.size
            if device_pool.host_capacity_tokens is None
            else device_pool.host_capacity_tokens
        ),
        start_layer=device_pool.start_layer,
        end_layer=device_pool.end_layer,
        target_layer_num=(
            (device_pool.layer_num + device_pool.layer_shard_size - 1)
            // device_pool.layer_shard_size
            if device_pool.layer_shard_enabled
            else device_pool.layer_num
        ),
        layer_shard_enabled=device_pool.layer_shard_enabled,
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
        device_pool = _fake_mla_device_pool()
        pool = MLATokenToKVPoolHost(
            device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=PHYSICAL_PAGE,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            device_capacity_tokens=(
                device_pool.size
                if device_pool.host_capacity_tokens is None
                else device_pool.host_capacity_tokens
            ),
            start_layer=device_pool.start_layer,
            end_layer=device_pool.end_layer,
            target_layer_num=(
                (device_pool.layer_num + device_pool.layer_shard_size - 1)
                // device_pool.layer_shard_size
                if device_pool.layer_shard_enabled
                else device_pool.layer_num
            ),
            layer_shard_enabled=device_pool.layer_shard_enabled,
        )
        self.assertEqual(pool.page_size, PHYSICAL_PAGE)
        self.assertEqual(pool.logical_size, pool.size)
        self.assertEqual(pool.logical_page_size, PHYSICAL_PAGE)


class TestTransferEntryPointsTranslate(CustomTestCase):
    def _run_backup(self, pool, host_indices, device_indices):
        device_pool = None
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

    def test_l3_data_page_is_guarded(self):
        pool = _make_host_pool(dcp_rank=0)
        with self.assertRaises(AssertionError):
            pool.get_data_page(0)


class TestLayerShardedHostTransfers(CustomTestCase):
    def test_layer_sharded_ascend_reports_unsupported_backend(self):
        """Unsupported Ascend backup must report the backend, not a buffer error."""
        pool = _fake_mla_device_pool(size=128)
        pool.layer_shard_enabled = True
        pool.layer_shard_size = 2
        pool._owned_local_layer_range = lambda: (0, 1)
        host = MLATokenToKVPoolHost(
            pool.get_device_pool_infos()[0].buffer_info,
            2,
            0,
            PHYSICAL_PAGE,
            "layer_first",
            pin_memory=False,
            device_capacity_tokens=(
                pool.size
                if pool.host_capacity_tokens is None
                else pool.host_capacity_tokens
            ),
            start_layer=pool.start_layer,
            end_layer=pool.end_layer,
            target_layer_num=(
                (pool.layer_num + pool.layer_shard_size - 1) // pool.layer_shard_size
                if pool.layer_shard_enabled
                else pool.layer_num
            ),
            layer_shard_enabled=pool.layer_shard_enabled,
        )
        self.addCleanup(host.destroy)
        # Ascend has no CUDA-style per-layer KV buffers.
        host.device_kv_buffers = ()
        indices = torch.arange(PHYSICAL_PAGE)
        with self.assertRaisesRegex(
            ValueError,
            "Layer-sharded HiCache backup does not support IO backend: kernel_ascend",
        ):
            host.backup_from_device_all_layer(
                pool, indices, indices, io_backend="kernel_ascend"
            )

    def test_owned_device_layers_keep_their_original_positions(self):
        # Rank 1 owns device layers 2/3, which occupy host layers 0/1.
        pool = _fake_mla_device_pool(size=128)
        pool.layer_num = 4
        pool.end_layer = 4
        pool.model_layer_ids = tuple(range(4))
        pool.layer_shard_enabled = True
        pool.layer_shard_size = 2
        pool._owned_local_layer_range = lambda: (2, 4)
        owned = [
            torch.full((192, 1, 12), value, dtype=torch.float16) for value in (2, 3)
        ]
        pool.kv_buffer = [
            torch.empty(0, 1, 12, dtype=torch.float16) for _ in range(2)
        ] + owned
        pool.data_ptrs = torch.tensor(
            [b.data_ptr() for b in pool.kv_buffer], dtype=torch.uint64
        )
        host = MLATokenToKVPoolHost(
            pool.get_device_pool_infos()[0].buffer_info,
            2,
            0,
            PHYSICAL_PAGE,
            "layer_first",
            pin_memory=False,
            device_capacity_tokens=(
                pool.size
                if pool.host_capacity_tokens is None
                else pool.host_capacity_tokens
            ),
            start_layer=pool.start_layer,
            end_layer=pool.end_layer,
            target_layer_num=(
                (pool.layer_num + pool.layer_shard_size - 1) // pool.layer_shard_size
                if pool.layer_shard_enabled
                else pool.layer_num
            ),
            layer_shard_enabled=pool.layer_shard_enabled,
        )
        self.addCleanup(host.destroy)
        self.assertEqual(host.layer_num, 2)
        source = torch.arange(64, 128)
        destination = torch.arange(64)
        host_indices = torch.arange(128, 192)

        def copy_pages(*, src_layers, dst_layers, src_indices, dst_indices, page_size):
            for src, dst in zip(src_layers, dst_layers, strict=True):
                dst[dst_indices] = src[src_indices]

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mla.transfer_kv_direct",
            side_effect=copy_pages,
            create=True,
        ) as transfer:
            host.backup_from_device_all_layer(pool, host_indices, source, "direct")
            for buffer in owned:
                buffer.zero_()
            for layer in range(4):
                host.load_to_device_per_layer(
                    pool, host_indices, destination, layer, "direct"
                )
        self.assertEqual(transfer.call_count, 4)
        for buffer, value in zip(owned, (2, 3), strict=True):
            torch.testing.assert_close(
                buffer[destination], torch.full((64, 1, 12), value, dtype=torch.float16)
            )
            self.assertEqual(torch.count_nonzero(buffer[64:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
