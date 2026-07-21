"""Unit tests for asymmetric MHA host KV pool transfer dispatch."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _make_host(layout: str) -> AsymmetricMHATokenToKVPoolHost:
    host = AsymmetricMHATokenToKVPoolHost.__new__(AsymmetricMHATokenToKVPoolHost)
    host.layout = layout
    host.page_size = 2
    host.layer_num = 3
    host.head_num = 2
    host.head_dim = 4
    host.v_head_dim = 6
    host.dtype = torch.float16

    if layout == "page_first":
        k_dims = (8, host.layer_num, host.head_num, host.head_dim)
        v_dims = (8, host.layer_num, host.head_num, host.v_head_dim)
    elif layout == "page_first_direct":
        host.page_num = 4
        k_dims = (
            host.page_num,
            host.layer_num,
            host.page_size,
            host.head_num,
            host.head_dim,
        )
        v_dims = (
            host.page_num,
            host.layer_num,
            host.page_size,
            host.head_num,
            host.v_head_dim,
        )
    else:
        raise ValueError(f"Unsupported test layout: {layout}")

    host.kv_buffer = (torch.empty(k_dims), torch.empty(v_dims))
    return host


def _make_device_pool(host: AsymmetricMHATokenToKVPoolHost) -> SimpleNamespace:
    size = 8
    k_buffer = [
        torch.empty(size, host.head_num, host.head_dim) for _ in range(host.layer_num)
    ]
    v_buffer = [
        torch.empty(size, host.head_num, host.v_head_dim) for _ in range(host.layer_num)
    ]
    return SimpleNamespace(
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        k_data_ptrs=torch.tensor([x.data_ptr() for x in k_buffer], dtype=torch.uint64),
        v_data_ptrs=torch.tensor([x.data_ptr() for x in v_buffer], dtype=torch.uint64),
    )


class TestAsymmetricMHATokenToKVPoolHost(CustomTestCase):
    def test_factory_selects_asymmetric_pool_for_mismatched_kv_dims(self):
        symmetric_pool = SimpleNamespace(head_dim=4, v_head_dim=4)
        asymmetric_pool = SimpleNamespace(head_dim=4, v_head_dim=6)

        self.assertIs(get_mha_host_pool_cls(symmetric_pool), MHATokenToKVPoolHost)
        self.assertIs(
            get_mha_host_pool_cls(asymmetric_pool), AsymmetricMHATokenToKVPoolHost
        )

    def test_kernel_load_splits_k_and_v_with_separate_strides(self):
        # Dispatch-only test: the CUDA kernel is mocked; this verifies that K and
        # V are sent as separate single-buffer calls with their own byte strides.
        host = _make_host("page_first")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_mla_pf_lf",
            create=True,
        ) as transfer:
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id=1,
                io_backend="kernel",
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertIs(k_call.kwargs["src"], host.k_buffer)
        self.assertIs(k_call.kwargs["dst"], device_pool.k_buffer[1])
        self.assertEqual(k_call.kwargs["item_size"], 16)
        self.assertEqual(k_call.kwargs["src_layout_dim"], 48)
        self.assertIs(v_call.kwargs["src"], host.v_buffer)
        self.assertIs(v_call.kwargs["dst"], device_pool.v_buffer[1])
        self.assertEqual(v_call.kwargs["item_size"], 24)
        self.assertEqual(v_call.kwargs["src_layout_dim"], 72)

    def test_kernel_backup_splits_k_and_v_with_separate_strides(self):
        # Dispatch-only test: D2H backup must pass separate K/V layer pointer
        # tables so the single-buffer MLA kernel gets the correct stride per side.
        host = _make_host("page_first")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_mla_lf_pf",
            create=True,
        ) as transfer:
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="kernel"
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertIs(k_call.kwargs["src_layers"], device_pool.k_data_ptrs)
        self.assertIs(k_call.kwargs["dst"], host.k_buffer)
        self.assertEqual(k_call.kwargs["item_size"], 16)
        self.assertEqual(k_call.kwargs["dst_layout_dim"], 48)
        self.assertIs(v_call.kwargs["src_layers"], device_pool.v_data_ptrs)
        self.assertIs(v_call.kwargs["dst"], host.v_buffer)
        self.assertEqual(v_call.kwargs["item_size"], 24)
        self.assertEqual(v_call.kwargs["dst_layout_dim"], 72)

    def test_direct_load_splits_k_and_v_for_page_first_direct(self):
        # Direct kernels derive copy size from each call's first tensor, so K/V
        # must be dispatched separately when their head dims differ.
        host = _make_host("page_first_direct")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_per_layer_direct_pf_lf",
            create=True,
        ) as transfer:
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id=2,
                io_backend="direct",
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertEqual(len(k_call.kwargs["src_ptrs"]), 1)
        self.assertEqual(len(k_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(k_call.kwargs["src_ptrs"][0], host.k_buffer)
        self.assertIs(k_call.kwargs["dst_ptrs"][0], device_pool.k_buffer[2])
        self.assertEqual(len(v_call.kwargs["src_ptrs"]), 1)
        self.assertEqual(len(v_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(v_call.kwargs["src_ptrs"][0], host.v_buffer)
        self.assertIs(v_call.kwargs["dst_ptrs"][0], device_pool.v_buffer[2])

    def test_direct_backup_splits_k_and_v_for_page_first_direct(self):
        host = _make_host("page_first_direct")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with mock.patch(
            "sglang.srt.mem_cache.pool_host.mha.transfer_kv_all_layer_direct_lf_pf",
            create=True,
        ) as transfer:
            host.backup_from_device_all_layer(
                device_pool, host_indices, device_indices, io_backend="direct"
            )

        self.assertEqual(transfer.call_count, 2)
        k_call, v_call = transfer.call_args_list
        self.assertEqual(len(k_call.kwargs["src_ptrs"]), host.layer_num)
        self.assertEqual(len(k_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(k_call.kwargs["src_ptrs"][0], device_pool.k_buffer[0])
        self.assertIs(k_call.kwargs["dst_ptrs"][0], host.k_buffer)
        self.assertEqual(len(v_call.kwargs["src_ptrs"]), host.layer_num)
        self.assertEqual(len(v_call.kwargs["dst_ptrs"]), 1)
        self.assertIs(v_call.kwargs["src_ptrs"][0], device_pool.v_buffer[0])
        self.assertIs(v_call.kwargs["dst_ptrs"][0], host.v_buffer)

    def test_direct_requires_page_first_direct_layout(self):
        host = _make_host("page_first")
        device_pool = _make_device_pool(host)
        host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        device_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "expected 'page_first_direct'"):
            host.load_to_device_per_layer(
                device_pool,
                host_indices,
                device_indices,
                layer_id=2,
                io_backend="direct",
            )


if __name__ == "__main__":
    unittest.main()
