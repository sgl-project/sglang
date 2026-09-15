"""Unit tests for the MXFP8 MHA host pool: UE8M0 scales must round-trip
through L2 alongside the fp8 payload."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolMXFP8
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.mha_mxfp8 import MHATokenToKVPoolMXFP8Host
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MXFP8_MODULE = "sglang.srt.mem_cache.pool_host.mha_mxfp8"
MHA_MODULE = "sglang.srt.mem_cache.pool_host.mha"

PAGE_SIZE = 4
LAYER_NUM = 2
HEAD_NUM = 1
HEAD_DIM = 8
SF_DIM = HEAD_DIM // 32 or 1
SF_PAGE_BYTES = HEAD_NUM * PAGE_SIZE * SF_DIM


def _ptr_key(ptrs: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(ptr) for ptr in ptrs.cpu().tolist())


def _cpu_mla_staged_lf_pf_copy(
    src_registry, *, ptr_src, src_indices, dst_indices, dst, **_
):
    """CPU stand-in for the staged D2H kernel: per (row, layer) gather-scatter."""
    src_layers = src_registry[_ptr_key(ptr_src)]
    for layer_id, src in enumerate(src_layers):
        dst[dst_indices, layer_id] = src[src_indices]


def _cpu_one_layer_mla_copy(*, cache_dst, indices_dst, cache_src, indices_src, **_):
    cache_dst[indices_dst] = cache_src[indices_src]


def _make_host(k_rows, v_rows):
    """A page_first MXFP8 host pool over CPU tensors, built without the base
    constructor so no pinned memory or device pool is needed."""
    host = MHATokenToKVPoolMXFP8Host.__new__(MHATokenToKVPoolMXFP8Host)
    host.layout = "page_first"
    host.page_size = PAGE_SIZE
    host.layer_num = LAYER_NUM
    host.page_num = 4
    host.k_sf_page_bytes = SF_PAGE_BYTES
    host.v_sf_page_bytes = SF_PAGE_BYTES
    host.device_pool = SimpleNamespace(layer_shard_enabled=False, layer_num=LAYER_NUM)
    host.k_scale_host = torch.zeros(
        host.page_num, LAYER_NUM, SF_PAGE_BYTES, dtype=torch.uint8
    )
    host.v_scale_host = torch.zeros(
        host.page_num, LAYER_NUM, SF_PAGE_BYTES, dtype=torch.uint8
    )
    host.k_scale_host_layers = list(host.k_scale_host.transpose(0, 1))
    host.v_scale_host_layers = list(host.v_scale_host.transpose(0, 1))
    host.k_scale_device_rows = k_rows
    host.v_scale_device_rows = v_rows
    host.k_scale_device_ptrs = torch.tensor(
        [r.data_ptr() for r in k_rows], dtype=torch.uint64
    )
    host.v_scale_device_ptrs = torch.tensor(
        [r.data_ptr() for r in v_rows], dtype=torch.uint64
    )
    host.k_scale_staging = torch.empty(2, LAYER_NUM, SF_PAGE_BYTES, dtype=torch.uint8)
    host.v_scale_staging = torch.empty(2, LAYER_NUM, SF_PAGE_BYTES, dtype=torch.uint8)
    return host


class TestMXFP8MHATokenToKVPoolHost(CustomTestCase):
    def test_factory_selects_mxfp8_host_pool(self):
        mxfp8_pool = MHATokenToKVPoolMXFP8.__new__(MHATokenToKVPoolMXFP8)
        plain_pool = SimpleNamespace(head_dim=4, v_head_dim=4)

        self.assertIs(get_mha_host_pool_cls(mxfp8_pool), MHATokenToKVPoolMXFP8Host)
        self.assertIs(get_mha_host_pool_cls(plain_pool), MHATokenToKVPoolHost)

    def test_size_per_token_counts_scales(self):
        host = MHATokenToKVPoolMXFP8Host.__new__(MHATokenToKVPoolMXFP8Host)
        host.page_size = PAGE_SIZE
        host.k_sf_page_bytes = SF_PAGE_BYTES
        host.v_sf_page_bytes = SF_PAGE_BYTES
        payload = 2 * LAYER_NUM * HEAD_NUM * HEAD_DIM
        with mock.patch.object(
            MHATokenToKVPoolHost, "get_size_per_token", return_value=payload
        ):
            host.layer_num = LAYER_NUM
            self.assertEqual(
                host.get_size_per_token(),
                payload + 2 * HEAD_NUM * SF_DIM * LAYER_NUM,
            )

    def test_scales_round_trip_device_host_device(self):
        num_device_pages = 4
        k_rows = [
            (
                torch.arange(num_device_pages * SF_PAGE_BYTES, dtype=torch.uint8)
                + 10 * layer
            )
            .reshape(num_device_pages, SF_PAGE_BYTES)
            .clone()
            for layer in range(LAYER_NUM)
        ]
        v_rows = [
            (
                torch.arange(num_device_pages * SF_PAGE_BYTES, dtype=torch.uint8)
                + 100
                + 10 * layer
            )
            .reshape(num_device_pages, SF_PAGE_BYTES)
            .clone()
            for layer in range(LAYER_NUM)
        ]
        host = _make_host(k_rows, v_rows)
        # Two device pages (2, 3) back up into host pages (1, 0).
        device_indices = torch.arange(2 * PAGE_SIZE, 4 * PAGE_SIZE, dtype=torch.int64)
        host_indices = torch.cat(
            [torch.arange(PAGE_SIZE, 2 * PAGE_SIZE), torch.arange(0, PAGE_SIZE)]
        ).to(torch.int64)
        expected_k = [rows[2:4].clone() for rows in k_rows]
        expected_v = [rows[2:4].clone() for rows in v_rows]
        registry = {
            _ptr_key(host.k_scale_device_ptrs): k_rows,
            _ptr_key(host.v_scale_device_ptrs): v_rows,
        }

        with (
            mock.patch.object(
                MHATokenToKVPoolHost, "backup_from_device_all_layer"
            ) as payload_backup,
            mock.patch.object(
                MHATokenToKVPoolHost, "load_to_device_per_layer"
            ) as payload_load,
            mock.patch(
                f"{MXFP8_MODULE}.jit_transfer_hicache_all_layer_mla_staged_lf_pf",
                side_effect=lambda **kw: _cpu_mla_staged_lf_pf_copy(registry, **kw),
            ) as staged,
            mock.patch(
                f"{MXFP8_MODULE}.jit_transfer_hicache_one_layer_mla",
                side_effect=_cpu_one_layer_mla_copy,
            ) as one_layer,
        ):
            host.backup_from_device_all_layer(
                host.device_pool, host_indices, device_indices, io_backend="kernel"
            )
            self.assertEqual(staged.call_count, 2)
            # Device page 2 landed in host page 1, device page 3 in host page 0.
            self.assertTrue(
                torch.equal(host.k_scale_host[1], torch.stack([r[2] for r in k_rows]))
            )
            self.assertTrue(
                torch.equal(host.v_scale_host[0], torch.stack([r[3] for r in v_rows]))
            )

            for rows in k_rows + v_rows:
                rows.zero_()
            for layer_id in range(LAYER_NUM):
                host.load_to_device_per_layer(
                    host.device_pool,
                    host_indices,
                    device_indices,
                    layer_id,
                    io_backend="kernel",
                )

        payload_backup.assert_called_once()
        self.assertEqual(payload_load.call_count, LAYER_NUM)
        self.assertEqual(one_layer.call_count, 2 * LAYER_NUM)
        for layer in range(LAYER_NUM):
            self.assertTrue(torch.equal(k_rows[layer][2:4], expected_k[layer]))
            self.assertTrue(torch.equal(v_rows[layer][2:4], expected_v[layer]))
            self.assertTrue(torch.all(k_rows[layer][:2] == 0))

    def test_storage_pages_are_rejected(self):
        host = MHATokenToKVPoolMXFP8Host.__new__(MHATokenToKVPoolMXFP8Host)
        with self.assertRaises(NotImplementedError):
            host.get_dummy_flat_data_page()
        with self.assertRaises(NotImplementedError):
            host.get_data_page(0)


if __name__ == "__main__":
    unittest.main()
