import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
    build_anchor_sidecar_stack,
)
from sglang.srt.mem_cache.kv_cache_configurator import (
    get_dsa_hicache_indexer_layers,
)
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.common import (
    ALLOC_MEMORY_FUNCS,
    alloc_with_pin_memory,
)
from sglang.srt.mem_cache.pool_host.dsa import DSAIndexerPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")


class TestDSAOffloadSignatures(unittest.TestCase):
    def test_cpu_copy_methods_accept_mamba_indices(self):
        for method_name in ("get_cpu_copy", "load_cpu_copy"):
            with self.subTest(method_name=method_name):
                signature = inspect.signature(getattr(DSATokenToKVPool, method_name))
                self.assertIn("mamba_indices", signature.parameters)


class TestDSAProducerHostSidecar(unittest.TestCase):
    def test_glm_indexer_producer_layers(self):
        config = SimpleNamespace(
            architectures=["GlmMoeDsaForCausalLM"],
            index_topk=2048,
            index_topk_freq=4,
            index_skip_topk_offset=3,
        )

        producer_layers = get_dsa_hicache_indexer_layers(config, 0, 78)

        self.assertEqual(producer_layers, [0, 1, 2, *range(6, 75, 4)])
        self.assertEqual(len(producer_layers), 21)
        self.assertEqual(get_dsa_hicache_indexer_layers(config, 6, 14), [0, 4])

        config.index_topk_freq = 1
        self.assertEqual(get_dsa_hicache_indexer_layers(config, 0, 78), list(range(78)))

    def test_compacts_target_layers_and_preserves_packed_drafts(self):
        target_buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(4)]
        draft_buffers = [torch.empty(1, dtype=torch.uint8) for _ in range(2)]
        target_pool = SimpleNamespace(
            device="cpu",
            index_k_with_scale_buffer=target_buffers,
            layer_num=4,
        )
        host = DSAIndexerPoolHost.__new__(DSAIndexerPoolHost)
        host.device_pool = target_pool
        host.device_layer_ids = [0, 2]
        host.mtp_draft_device_pools = [
            SimpleNamespace(index_k_with_scale_buffer=[buffer])
            for buffer in draft_buffers
        ]
        host.layout = "layer_first"
        host.layer_num = 4
        host.indexer_page_num = 3
        host.indexer_page_stride_size = 8
        host.indexer_layout_dim = host.layer_num * host.indexer_page_stride_size
        host.indexer_dtype = torch.uint8
        host.device = "cpu"
        host.pin_memory = False
        host.allocator = mock.sentinel.allocator
        alloc = mock.Mock(
            side_effect=lambda dims, **kwargs: torch.empty(dims, dtype=kwargs["dtype"])
        )

        with mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": alloc}):
            host.init_kv_buffer()

        expected_buffers = [target_buffers[0], target_buffers[2], *draft_buffers]
        self.assertEqual(len(host.packed_device_index_buffers), len(expected_buffers))
        for actual, expected in zip(host.packed_device_index_buffers, expected_buffers):
            self.assertIs(actual, expected)
        self.assertEqual(len(host.index_k_data_refs), 4)
        host.target_layer_num = 2
        host.host_layer_by_device = {0: 0, 2: 1}
        self.assertEqual(host._host_layer_index(2), 1)
        self.assertIsNone(host._host_layer_index(1))
        self.assertEqual(host._draft_host_layer_index(4), 2)
        self.assertEqual(host._draft_host_layer_index(5), 3)

    def test_anchor_mapping_keeps_all_mla_and_draft_layers(self):
        target_pool = SimpleNamespace(layer_num=4, kv_cache_dim=576)
        draft_pool = SimpleNamespace(index_k_with_scale_buffer=[object()])
        target_pool.hicache_indexer_layers = [0, 2]
        params = SimpleNamespace(
            page_size=64,
            mtp_draft_device_pools=(draft_pool,),
            token_to_kv_pool_allocator=mock.sentinel.allocator,
            tp_cache_group=mock.sentinel.tp_group,
            attn_cp_cache_group=mock.sentinel.attn_cp_group,
            attn_tp_cache_group=mock.sentinel.attn_tp_group,
            pp_cache_group=mock.sentinel.pp_group,
        )
        anchor_host = SimpleNamespace(
            layout="layer_first",
            page_size=64,
            device="cpu",
            size=128,
            logical_size=128,
            can_use_write_back_jit=False,
        )
        sidecar_host = SimpleNamespace(can_use_write_back_jit=False)

        with (
            mock.patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "build_kv_host_pool",
                return_value=anchor_host,
            ),
            mock.patch(
                "sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler."
                "HybridCacheController"
            ),
        ):
            group, _ = build_anchor_sidecar_stack(
                params=params,
                kv_pool=target_pool,
                sidecar_pool_name=PoolName.INDEXER,
                full_layer_mapping={i: i for i in range(4)},
                sidecar_layer_mapping={0: 0, 2: 2},
                load_cache_event=mock.sentinel.load_cache_event,
                storage_backend=None,
                use_mla=True,
                sidecar_host_pool_factory=lambda _: sidecar_host,
            )

        anchor_mapper = group.get_entry(PoolName.KV).layer_mapper
        indexer_mapper = group.get_entry(PoolName.INDEXER).layer_mapper
        self.assertEqual([anchor_mapper(i) for i in range(5)], [0, 1, 2, 3, 4])
        self.assertEqual([indexer_mapper(i) for i in range(5)], [0, None, 2, None, 4])
        self.assertEqual(
            group.get_entry(PoolName.KV).packed_draft_device_pools, (draft_pool,)
        )
        self.assertEqual(
            group.get_entry(PoolName.INDEXER).packed_draft_device_pools,
            (draft_pool,),
        )


class TestDSAHiCacheTransfer(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for DSA host transfer tests.")
        if is_npu() or is_xpu():
            self.skipTest("DSA host transfer tests only support CUDA/ROCm.")
        if not (is_cuda() or is_hip()):
            self.skipTest("CUDA/ROCm not available.")

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
        layer_num = 4
        producer_layers = [0, 2]
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
            mla_host = MLATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=page_size,
                layout="layer_first",
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
                override_kv_cache_dim=device_pool.kv_cache_dim,
            )
            indexer_host = DSAIndexerPoolHost(
                device_pool=device_pool,
                anchor_host=mla_host,
                layout="layer_first",
                pin_memory=pin_memory,
                device="cpu",
                allocator_type="default",
                device_layer_ids=producer_layers,
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

        for host_layer_id, layer_id in enumerate(producer_layers):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                got = indexer_host.index_k_with_scale_buffer[host_layer_id][
                    host_page
                ].cpu()
                expected = device_pool.index_k_with_scale_buffer[layer_id][
                    device_page
                ].cpu()
                self.assertTrue(torch.equal(got, expected))

        # Compacting the indexer sidecar must not compact ordinary MLA KV.
        for layer_id in range(layer_num):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
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


if __name__ == "__main__":
    unittest.main()
