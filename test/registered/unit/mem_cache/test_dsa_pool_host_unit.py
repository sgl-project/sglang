import inspect
import unittest

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=19, suite="stage-b-test-1-gpu-small-amd")


def _build_mla_host(device_pool, page_size: int, pin_memory: bool):
    """Mirror the production MLATokenToKVPoolHost construction for a DSA pool."""
    return MLATokenToKVPoolHost(
        buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
        host_to_device_ratio=2.0,
        host_size=0,
        page_size=page_size,
        layout="layer_first",
        pin_memory=pin_memory,
        device="cpu",
        allocator_type="default",
        override_kv_cache_dim=device_pool.kv_cache_dim,
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


class TestDSAOffloadSignatures(unittest.TestCase):
    def test_cpu_copy_methods_accept_mamba_indices(self):
        for method_name in ("get_cpu_copy", "load_cpu_copy"):
            with self.subTest(method_name=method_name):
                signature = inspect.signature(getattr(DSATokenToKVPool, method_name))
                self.assertIn("mamba_indices", signature.parameters)


class TestDSAHiCacheTransfer(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for DSA host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("DSA host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    @staticmethod
    def _token_indices_for_pages(pages: torch.Tensor, page_size: int, device: str):
        parts = [
            torch.arange(
                int(page_id) * page_size,
                (int(page_id) + 1) * page_size,
                device=device,
                dtype=torch.int64,
            )
            for page_id in pages.tolist()
        ]
        return torch.cat(parts, dim=0)

    def _run_device_to_host_indexer_copy(self, io_backend: str):
        page_size = 1 if is_hip() else 64
        layer_num = 2
        size = page_size * 4

        device_pool = DSATokenToKVPool(
            size=size,
            page_size=page_size,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=layer_num,
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
        )
        pin_memory = io_backend == "kernel"
        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        if pin_memory:
            ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            mla_host = _build_mla_host(device_pool, page_size, pin_memory)
            indexer_host = DSAIndexerPoolHost(
                decl=make_dsa_indexer_pool_decl(device_pool),
                anchor_host=mla_host,
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        for layer_id in range(layer_num):
            buf = device_pool.index_k_with_scale_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=torch.uint8
            ).view_as(buf)
            buf.copy_((data + layer_id) % 256)
            kv_buf = device_pool.kv_buffer[layer_id]
            kv_data = torch.arange(
                kv_buf.numel(), device=kv_buf.device, dtype=kv_buf.dtype
            ).view_as(kv_buf)
            kv_buf.copy_(kv_data + layer_id)

        device_pages = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int64)
        host_pages = torch.tensor(
            [0, 1, 2],
            device="cuda" if io_backend == "kernel" else "cpu",
            dtype=torch.int64,
        )
        device_indices = self._token_indices_for_pages(
            device_pages, page_size, device="cuda"
        )
        host_indices = self._token_indices_for_pages(
            host_pages,
            page_size,
            device="cuda" if io_backend == "kernel" else "cpu",
        )

        mla_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        indexer_host.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )

        for layer_id in range(layer_num):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                got = indexer_host.index_k_with_scale_buffer[layer_id][host_page].cpu()
                expected = device_pool.index_k_with_scale_buffer[layer_id][
                    device_page
                ].cpu()
                self.assertTrue(torch.equal(got, expected))
                host_start = host_page * page_size
                device_start = device_page * page_size
                got_kv = mla_host.kv_buffer[layer_id][
                    host_start : host_start + page_size
                ].cpu()
                expected_kv = device_pool.kv_buffer[layer_id][
                    device_start : device_start + page_size
                ].cpu()
                self.assertTrue(torch.equal(got_kv, expected_kv))

    @unittest.skipIf(
        is_hip(),
        '`io_backend="kernel"` path in MLATokenToKVPoolHost.backup_from_device_all_layer '
        "raises ValueError on AMD (only the `direct` IO backend is wired for ROCm). "
        "The other 62 tests in this file pass on AMD.",
    )
    def test_device_to_host_indexer_kernel(self):
        self._run_device_to_host_indexer_copy(io_backend="kernel")

    @unittest.skipIf(
        is_hip(),
        "DSATokenToKVPool with page_size=1 (used on HIP) trips the ROCm 7.2.0 "
        "HIP-preshuffle assert `page_size % 16 == 0` during pool construction "
        "(seen on pr-test-amd-rocm720). The rest of this file passes on AMD.",
    )
    def test_device_to_host_indexer_direct(self):
        self._run_device_to_host_indexer_copy(io_backend="direct")


class TestDSAIndexKElision(CustomTestCase):
    """Skip-topk layers get a 0-row index-K buffer, and the host (L2) indexer
    pool packs only the layers that actually own index keys."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for DSA pool allocation tests.")
        if is_npu() or is_xpu() or not (is_cuda() or is_hip()):
            self.skipTest("DSA pool allocation tests only support CUDA/ROCm.")
        cm = get_parallel().override(attn_dcp_size=1)
        cm.__enter__()
        self.addCleanup(cm.__exit__, None, None, None)

    @staticmethod
    def _build_pool(page_size: int, skip_topk_layers):
        return DSATokenToKVPool(
            size=page_size * 4,
            page_size=page_size,
            kv_lora_rank=128,
            dtype=torch.bfloat16,
            qk_rope_head_dim=32,
            layer_num=len(skip_topk_layers),
            device="cuda",
            enable_memory_saver=False,
            kv_cache_dim=576,
            index_head_dim=128,
            skip_topk_layers=skip_topk_layers,
        )

    def test_skip_topk_layers_get_zero_row_buffers(self):
        page_size = 1 if is_hip() else 64
        # Layer 1 reuses layer 0's top-k, so it never writes index-K.
        pool = self._build_pool(page_size, [False, True, False])

        buffers = pool.index_k_with_scale_buffer
        self.assertEqual(buffers[1].shape[0], 0)
        self.assertGreater(buffers[0].shape[0], 0)
        self.assertGreater(buffers[2].shape[0], 0)

        # The PD wire layer trims on these: a 0-byte region is never registered
        # (Mooncake rejects it) and `is_elided_entry` drops the transfer pair.
        _, data_lens, item_lens = pool.get_state_buf_infos()
        self.assertEqual(data_lens[1], 0)
        self.assertEqual(item_lens[1], 0)
        self.assertGreater(item_lens[0], 0)
        self.assertGreater(item_lens[2], 0)

    def test_all_layers_owned_when_nothing_is_skipped(self):
        page_size = 1 if is_hip() else 64
        pool = self._build_pool(page_size, [False, False])

        self.assertTrue(all(buf.shape[0] > 0 for buf in pool.index_k_with_scale_buffer))
        _, data_lens, item_lens = pool.get_state_buf_infos()
        self.assertTrue(all(length > 0 for length in data_lens))
        self.assertTrue(all(length > 0 for length in item_lens))

    def test_host_indexer_pool_packs_only_owning_layers(self):
        page_size = 1 if is_hip() else 64
        skip_topk_layers = [False, True, False, True]
        pool = self._build_pool(page_size, skip_topk_layers)

        decl = make_dsa_indexer_pool_decl(pool)
        self.assertEqual(decl.owned_device_layers, (0, 2))

        original_alloc = ALLOC_MEMORY_FUNCS["cuda"]
        ALLOC_MEMORY_FUNCS["cuda"] = alloc_with_pin_memory
        try:
            mla_host = _build_mla_host(pool, page_size, pin_memory=True)
            indexer_host = DSAIndexerPoolHost(
                decl=decl,
                anchor_host=mla_host,
                pin_memory=True,
                device="cpu",
                allocator_type="default",
            )
        finally:
            ALLOC_MEMORY_FUNCS["cuda"] = original_alloc

        # Host storage is compacted: 2 owning layers out of 4 device layers,
        # addressed by compact slot rather than by device layer id.
        self.assertEqual(indexer_host.target_layer_num, 2)
        self.assertEqual(indexer_host._host_layer_index(0), 0)
        self.assertEqual(indexer_host._host_layer_index(2), 1)
        self.assertEqual(sorted(indexer_host._device_to_host_layer), [0, 2])

    def test_pool_declares_no_indexer_host_when_every_layer_skips(self):
        page_size = 1 if is_hip() else 64
        pool = self._build_pool(page_size, [True, True])

        # No layer owns index keys, so there is nothing to mirror to the host.
        names = [decl.pool_name for decl in pool.host_pool_decls()]
        self.assertNotIn(PoolName.INDEXER, names)

    def test_hisparse_pool_elides_and_scales_the_saving_by_the_ratio(self):
        # HiSparse sizes index-K at logical (host) capacity on the device via
        # `index_buf_size = size * host_to_device_ratio`, so eliding a skip layer
        # saves ratio times as much as on a plain DSA pool. The elide decision
        # still arrives as the same `skip_topk_layers` kwarg.
        page_size = 1 if is_hip() else 64
        ratio = 2

        def build(skip_topk_layers):
            return HiSparseDSATokenToKVPool(
                size=page_size * 4,
                page_size=page_size,
                kv_lora_rank=128,
                dtype=torch.bfloat16,
                qk_rope_head_dim=32,
                layer_num=len(skip_topk_layers),
                device="cuda",
                index_head_dim=128,
                enable_memory_saver=False,
                kv_cache_dim=576,
                skip_topk_layers=skip_topk_layers,
                host_to_device_ratio=ratio,
            )

        elided = build([False, True, False, True])
        self.assertEqual(elided.index_k_with_scale_buffer[1].shape[0], 0)
        self.assertEqual(elided.index_k_with_scale_buffer[3].shape[0], 0)
        _, data_lens, item_lens = elided.get_state_buf_infos()
        self.assertEqual([data_lens[1], data_lens[3]], [0, 0])
        self.assertEqual([item_lens[1], item_lens[3]], [0, 0])

        full = build([False, False, False, False])
        elided_bytes = sum(b.nbytes for b in elided.index_k_with_scale_buffer)
        full_bytes = sum(b.nbytes for b in full.index_k_with_scale_buffer)
        # 2 of 4 layers own index-K.
        self.assertEqual(elided_bytes * 2, full_bytes)
        # Each owning layer still covers the full logical host capacity.
        rows_per_owning_layer = elided.index_k_with_scale_buffer[0].shape[0]
        self.assertEqual(
            rows_per_owning_layer, full.index_k_with_scale_buffer[0].shape[0]
        )
        self.assertGreater(rows_per_owning_layer, 0)


if __name__ == "__main__":
    unittest.main()
