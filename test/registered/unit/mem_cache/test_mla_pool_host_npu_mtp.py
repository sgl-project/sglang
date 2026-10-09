import unittest
from enum import Enum
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.pool_host import mla
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Direction(Enum):
    H2D = 1
    D2H = 2


def make_device(layers, indexer_ids=(), *, start_layer=0, scale=False, fp8=False):
    # Deliberately no CUDA data_ptrs or kv_buffer attributes.
    pool = SimpleNamespace(
        layer_num=layers,
        start_layer=start_layer,
        end_layer=start_layer + layers,
        layer_shard_enabled=False,
        size=6,
        page_size=2,
        device="cpu",
        store_dtype=torch.float32,
        kv_lora_rank=4,
        qk_rope_head_dim=2,
        kv_cache_dim=8 if fp8 else 4,
        index_head_dim=3 if indexer_ids else None,
        num_indexer_layers=len(indexer_ids),
        indexer_layer_ids=tuple(start_layer + i for i in indexer_ids),
        dsa_kv_cache_store_fp8=fp8,
        k_buffer=torch.zeros(layers, 4, 2, 1, 8 if fp8 else 4),
        v_buffer=torch.zeros(layers, 4, 2, 1, 0 if fp8 else 2),
        index_k_buffer=(
            torch.zeros(len(indexer_ids), 4, 2, 1, 3) if indexer_ids else None
        ),
        index_k_scale_buffer=(
            torch.zeros(len(indexer_ids), 4, 2, 1) if scale else None
        ),
    )
    return pool


def cpu_exchange(**kwargs):
    """CPU oracle for the native page copy, including non-contiguous host views."""
    page_size = kwargs["page_size"]
    for name in ("k", "v", "index_k", "index_k_scale"):
        dev, host = kwargs["device_" + name], kwargs["host_" + name]
        if dev is None or host is None or not dev.numel():
            continue
        if dev.ndim == 4:
            dev = dev.unsqueeze(-1)
        assert dev.shape[0] == host.shape[1]
        for d, h in zip(
            kwargs["device_indices"][::page_size],
            kwargs["host_indices"][::page_size],
        ):
            d, h = int(d) // page_size, int(h) // page_size
            if kwargs["direction"] == Direction.D2H:
                host[h].copy_(dev[:, d])
            else:
                dev[:, d].copy_(host[h])


class TestNPUMTPHostPool(unittest.TestCase):
    def setUp(self):
        for name, value in (
            ("_is_npu", True),
            ("_is_cuda", False),
            ("_is_hip", False),
            ("ascendc_io_enabled", lambda: False),
            ("memfabric_host_memory_enabled", lambda: False),
            ("TransferDirection", Direction),
            ("transfer_kv_dim_exchange", cpu_exchange),
        ):
            patch = mock.patch.object(mla, name, value, create=True)
            patch.start()
            self.addCleanup(patch.stop)
        self.host_indices = torch.tensor([4, 5, 0, 1])
        self.device_indices = torch.tensor([0, 1, 4, 5])

    def make_host(self, target, drafts=()):
        def allocate(dims, *, dtype, **kwargs):
            return torch.full(dims, -1, dtype=dtype)

        with (
            mock.patch.dict(mla.ALLOC_MEMORY_FUNCS, {"cpu": allocate}),
            mock.patch.object(mla, "make_kernel_ptr_table", return_value=None),
            mock.patch(
                "sglang.srt.mem_cache.pool_host.base.host_memory_budget_bytes",
                return_value=1 << 40,
            ),
        ):
            return MLATokenToKVPoolHost(
                target,
                1,
                0,
                2,
                "page_first_kv_split",
                pin_memory=False,
                mtp_draft_device_pools=drafts,
            )

    def test_init_without_data_ptrs_and_packed_indexer_capacity(self):
        target = make_device(78, (0, 30, 77), scale=True)
        draft = make_device(1, (0,), scale=True)
        host = self.make_host(target, (draft,))
        self.assertEqual(host.k_buffer.shape[1], 79)
        self.assertEqual(host.index_k_buffer.shape[1], 4)
        self.assertEqual(host.index_k_scale_buffer.shape[1], 4)
        allocated = sum(t.nbytes for t in host.get_hybrid_pool_buffer())
        self.assertEqual(host.size_per_token * host.size, allocated)

    def test_d2h_backup_and_h2d_restore_target_and_two_drafts(self):
        target = make_device(4, (0, 3), scale=True)
        drafts = (make_device(1, (0,), scale=True), make_device(1, (0,), scale=True))
        pools = (target, *drafts)
        host = self.make_host(target, drafts)
        expected = []
        for i, pool in enumerate(pools):
            saved = {}
            for j, name in enumerate(("k", "v", "index_k", "index_k_scale")):
                tensor = getattr(pool, name + "_buffer")
                tensor.copy_(
                    torch.arange(tensor.numel()).reshape(tensor.shape)
                    + 1000 * (i + 1)
                    + 100 * j
                )
                saved[name] = tensor.clone()
            expected.append(saved)
        host.backup_from_device_all_layer(
            target, self.host_indices, self.device_indices, "kernel_ascend"
        )
        # Check the actual tail, not just a round trip (symmetric mistakes can cancel).
        for name in ("k", "v", "index_k", "index_k_scale"):
            buffer = getattr(host, name + "_buffer")
            for i, slot in enumerate((4, 5) if name in ("k", "v") else (2, 3)):
                want = expected[i + 1][name][0, 0]
                if want.ndim == 2:
                    want = want.unsqueeze(-1)
                torch.testing.assert_close(buffer[2, slot], want)
            self.assertTrue(torch.all(buffer[1] == -1))
        for pool in pools:
            for name in ("k", "v", "index_k", "index_k_scale"):
                getattr(pool, name + "_buffer").fill_(-2)
        for layer in range(4):
            host.load_to_device_per_layer(
                target, self.host_indices, self.device_indices, layer, "kernel_ascend"
            )
        for i, draft in enumerate(drafts):
            host.load_to_device_per_layer(
                draft,
                self.host_indices,
                self.device_indices,
                4 + i,
                "kernel_ascend",
                is_draft=True,
            )
        for pool, saved in zip(pools, expected):
            for name in saved:
                tensor = getattr(pool, name + "_buffer")
                torch.testing.assert_close(tensor[:, [0, 2]], saved[name][:, [0, 2]])
                self.assertTrue(torch.all(tensor[:, 1] == -2))

    def test_target_indexer_does_not_include_draft_tail(self):
        target, draft = make_device(4, (0, 3)), make_device(1, (0,))
        host = self.make_host(target, (draft,))
        target.index_k_buffer.fill_(7)
        host._transfer_npu_mla_range(
            target,
            self.host_indices,
            self.device_indices,
            device_layer_start=0,
            host_layer_start=0,
            layer_num=4,
            direction=Direction.D2H,
        )
        self.assertTrue(torch.all(host.index_k_buffer[2, :2] == 7))
        self.assertTrue(torch.all(host.index_k_buffer[:, 2] == -1))

    def test_draft_without_indexer_and_fp8_empty_v(self):
        target, draft = make_device(4, (0, 3), fp8=True), make_device(1, fp8=True)
        host = self.make_host(target, (draft,))
        draft.k_buffer.fill_(17)
        host.backup_from_device_all_layer(
            target, self.host_indices, self.device_indices, "kernel_ascend"
        )
        self.assertEqual(host.index_k_buffer.shape[1], 2)
        self.assertTrue(torch.all(host.k_buffer[2, 4] == 17))
        self.assertTrue(torch.all(host.v_buffer == -1))

    def test_ascendc_draft_metadata_uses_packed_tail_and_page_stride(self):
        target, draft = (
            make_device(4, (0, 3), scale=True),
            make_device(1, (0,), scale=True),
        )
        host = self.make_host(target, (draft,))
        with (
            mock.patch.object(mla, "ascendc_io_enabled", return_value=True),
            mock.patch.object(host, "_transfer_ascendc_sparse_copy") as transfer,
        ):
            host.load_to_device_per_layer(
                draft,
                self.host_indices,
                self.device_indices,
                4,
                "kernel_ascend",
                is_draft=True,
            )
        comps = transfer.call_args.kwargs["components"]
        self.assertEqual(len(comps), 4)
        for meta, name, slot in zip(
            comps, ("k", "v", "index_k", "index_k_scale"), (4, 4, 2, 2)
        ):
            dev, buf = getattr(draft, name + "_buffer"), getattr(host, name + "_buffer")
            self.assertEqual(meta[0], dev.data_ptr())
            self.assertEqual(meta[1], buf[0, slot].data_ptr())
            self.assertEqual(meta[4], buf.stride(0) * buf.element_size())
            self.assertEqual(meta[7:], (0, 1))

    def test_ascendc_sparse_indexer_and_pp_layer_ids(self):
        target, draft = make_device(4, (1, 3), start_layer=10), make_device(1, (0,))
        host = self.make_host(target, (draft,))
        with (
            mock.patch.object(mla, "ascendc_io_enabled", return_value=True),
            mock.patch.object(host, "_transfer_ascendc_sparse_copy") as transfer,
        ):
            for layer, count in ((0, 2), (1, 3), (2, 2), (3, 3)):
                host.load_to_device_per_layer(
                    target,
                    self.host_indices,
                    self.device_indices,
                    layer,
                    "kernel_ascend",
                )
                comps = transfer.call_args.kwargs["components"]
                self.assertEqual(len(comps), count)
                self.assertEqual(comps[0][1], host.k_buffer[0, layer].data_ptr())
                if count == 3:
                    self.assertEqual(
                        comps[2][1], host.index_k_buffer[0, layer // 2].data_ptr()
                    )

    def test_ascendc_d2h_keeps_target_and_draft_components_separate(self):
        target, draft = make_device(4, (0, 3)), make_device(1, (0,))
        host = self.make_host(target, (draft,))
        with (
            mock.patch.object(mla, "ascendc_io_enabled", return_value=True),
            mock.patch.object(host, "_transfer_ascendc_sparse_copy") as transfer,
        ):
            host.backup_from_device_all_layer(
                target, self.host_indices, self.device_indices, "kernel_ascend"
            )
        self.assertEqual(transfer.call_count, 2)
        target_call, draft_call = transfer.call_args_list
        self.assertEqual(target_call.args[3], Direction.D2H)
        self.assertEqual(draft_call.args[3], Direction.D2H)
        self.assertEqual(target_call.kwargs["components"][0][7:], (0, 4))
        self.assertEqual(target_call.kwargs["components"][2][7:], (0, 2))
        self.assertEqual(
            draft_call.kwargs["components"][2][1], host.index_k_buffer[0, 2].data_ptr()
        )

    def test_draft_indexer_scale_is_allocated_when_target_has_no_scale(self):
        target, draft = make_device(4, (0, 3)), make_device(1, (0,), scale=True)
        host = self.make_host(target, (draft,))
        draft.index_k_scale_buffer.fill_(23)
        host.backup_from_device_all_layer(
            target, self.host_indices, self.device_indices, "kernel_ascend"
        )
        self.assertTrue(torch.all(host.index_k_scale_buffer[2, 2] == 23))
        self.assertEqual(
            host.size * host.size_per_token,
            sum(t.nbytes for t in host.get_hybrid_pool_buffer()),
        )

    def test_cuda_style_packed_device_buffers_are_preserved(self):
        target, draft = make_device(4), make_device(1)
        target.data_ptrs, draft.data_ptrs = torch.arange(4), torch.tensor([9])
        target.kv_buffer, draft.kv_buffer = list(target.k_buffer), list(draft.k_buffer)
        with mock.patch.object(mla, "_is_npu", False):
            host = self.make_host(target, (draft,))
        torch.testing.assert_close(
            host.packed_device_data_ptrs, torch.tensor([0, 1, 2, 3, 9])
        )
        self.assertEqual(len(host.packed_device_kv_buffers), 5)
        self.assertIs(host.packed_device_kv_buffers[-1], draft.kv_buffer[0])

    def test_rejects_multi_layer_draft_pool(self):
        with self.assertRaisesRegex(ValueError, "one layer per draft"):
            self.make_host(make_device(4), (make_device(2),))

    def test_non_mtp_legacy_round_trip(self):
        target = make_device(4, (0, 3))
        host = self.make_host(target)
        target.k_buffer.fill_(19)
        host.backup_from_device_all_layer(
            target, self.host_indices, self.device_indices, "kernel_ascend"
        )
        target.k_buffer.zero_()
        host.load_to_device_per_layer(
            target, self.host_indices, self.device_indices, 0, "kernel_ascend"
        )
        self.assertTrue(torch.all(target.k_buffer[:, [0, 2]] == 19))


if __name__ == "__main__":
    unittest.main()
