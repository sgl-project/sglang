import inspect
import unittest

import torch

from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=9, suite="stage-b-test-1-gpu-small-amd")


def _make_indexer_host(pool, anchor, *, packed_draft_device_pools=(), **kwargs):
    from sglang.srt.mem_cache.hybrid_cache.host_pool_config import (
        prepare_host_pool_config,
    )

    config = prepare_host_pool_config(
        root=pool,
        full_layer_mapping={i: i for i in range(pool.layer_num)},
        transfer_layer_id_max=pool.layer_num,
        transfer_page_size=anchor.page_size,
        packed_draft_device_pools=packed_draft_device_pools,
    ).sidecars[0]
    return DSAIndexerPoolHost(
        config.buffer_info,
        layer_mapping=config.device_to_host_layer,
        target_device_layer_num=pool.layer_num,
        num_host_pages=anchor.page_num,
        transfer_page_size=anchor.page_size,
        layout=anchor.layout,
        buffer_device=pool.device,
        dtype=pool.store_dtype,
        layer_sharded=pool.layer_shard_enabled,
        **kwargs,
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

    def _run_device_to_host_indexer_copy(
        self, io_backend: str, use_fp4_indexer: bool = False
    ):
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
            use_fp4_indexer=use_fp4_indexer,
        )
        pin_memory = io_backend == "kernel"
        mla_host = MLATokenToKVPoolHost(
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
        indexer_host = _make_indexer_host(
            device_pool,
            mla_host,
            pin_memory=pin_memory,
            device="cpu",
            allocator_type="default",
        )

        self.addCleanup(mla_host.destroy)
        self.addCleanup(indexer_host.destroy)
        for layer_id in range(layer_num):
            buf = device_pool.index_k_with_scale_buffer[layer_id]
            data = torch.arange(
                buf.numel(), device=buf.device, dtype=torch.int32
            ).view_as(buf)
            buf.copy_((data + layer_id).remainder(251).to(torch.uint8))
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

        expected_index_pages = [
            buffer[device_pages].clone()
            for buffer in device_pool.index_k_with_scale_buffer
        ]
        for buffer in device_pool.index_k_with_scale_buffer:
            buffer.zero_()
        for layer_id in range(layer_num):
            indexer_host.load_to_device_per_layer(
                device_pool, host_indices, device_indices, layer_id, io_backend
            )
        torch.cuda.synchronize()
        for buffer, expected in zip(
            device_pool.index_k_with_scale_buffer, expected_index_pages, strict=True
        ):
            self.assertTrue(torch.equal(buffer[device_pages], expected))
            self.assertEqual(torch.count_nonzero(buffer[0]).item(), 0)

    @unittest.skipIf(
        is_hip(),
        '`io_backend="kernel"` path in MLATokenToKVPoolHost.backup_from_device_all_layer '
        "raises ValueError on AMD (only the `direct` IO backend is wired for ROCm). "
        "The other 62 tests in this file pass on AMD.",
    )
    def test_device_to_host_indexer_kernel(self):
        for fp4 in (False, True):
            with self.subTest(fp4=fp4):
                self._run_device_to_host_indexer_copy("kernel", fp4)

    @unittest.skipIf(
        is_hip(),
        "DSATokenToKVPool with page_size=1 (used on HIP) trips the ROCm 7.2.0 "
        "HIP-preshuffle assert `page_size % 16 == 0` during pool construction "
        "(seen on pr-test-amd-rocm720). The rest of this file passes on AMD.",
    )
    def test_device_to_host_indexer_direct(self):
        for fp4 in (False, True):
            with self.subTest(fp4=fp4):
                self._run_device_to_host_indexer_copy("direct", fp4)

    @unittest.skipUnless(is_cuda(), "CUDA compact HiSparse regression")
    def test_shared_indexer_l2_roundtrip_with_packed_draft(self):
        """A PP stage starting in a shared layer retains KV and the draft tail."""
        for layout in ("layer_first", "page_first_direct"):
            with self.subTest(layout=layout):
                # Match the controller: direct transfers use a dedicated CUDA stream.
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    self._run_shared_indexer_roundtrip(layout)

    def _run_shared_indexer_roundtrip(self, layout):
        def pool(layers, start, mask=None):
            return DSATokenToKVPool(
                size=256,
                page_size=64,
                kv_lora_rank=512,
                dtype=torch.bfloat16,
                qk_rope_head_dim=64,
                layer_num=layers,
                device="cuda",
                index_head_dim=128,
                enable_memory_saver=False,
                kv_cache_dim=576,
                start_layer=start,
                end_layer=start + layers,
                skip_topk_layers=mask,
            )

        target = pool(6, 21, [True, False, True, True, True, False])
        draft = pool(1, 78)
        anchor = MLATokenToKVPoolHost(
            buffer_info=target.get_device_pool_infos()[0].buffer_info,
            device_capacity_tokens=target.size,
            target_layer_num=target.layer_num,
            start_layer=target.start_layer,
            end_layer=target.end_layer,
            host_to_device_ratio=2,
            host_size=0,
            page_size=64,
            layout=layout,
            pin_memory=True,
            device="cpu",
            allocator_type="default",
            override_kv_cache_dim=target.kv_cache_dim,
        )
        host = _make_indexer_host(
            target,
            anchor,
            packed_draft_device_pools=(draft,),
            pin_memory=True,
            device="cpu",
            allocator_type="default",
        )
        self.assertEqual(host._live_target_layers, [1, 5])
        self.assertEqual(host.layer_num, 3)  # Two producers plus the draft.
        self.assertTrue(all(buf.numel() for buf in target.kv_buffer))
        expected = {}
        for i in (1, 5):
            target.index_k_with_scale_buffer[i].random_(0, 256)
            expected[i] = target.index_k_with_scale_buffer[i][1:3].clone()
        draft.index_k_with_scale_buffer[0].random_(0, 256)
        expected_draft = draft.index_k_with_scale_buffer[0][1:3].clone()
        for i, buf in enumerate(target.kv_buffer):
            buf.fill_(i + 1)
        device_indices = torch.arange(64, 192, dtype=torch.int64, device="cpu")
        host_indices = torch.arange(0, 128, dtype=torch.int64, device="cpu")
        anchor.backup_from_device_all_layer(
            target, host_indices, device_indices, "direct"
        )
        host.backup_from_device_all_layer(
            target, host_indices, device_indices, "direct"
        )
        torch.cuda.synchronize()
        for buf in target.index_k_with_scale_buffer + draft.index_k_with_scale_buffer:
            buf.zero_()
        for buf in target.kv_buffer:
            buf.zero_()
        for i in range(6):
            anchor.load_to_device_per_layer(
                target, host_indices, device_indices, i, "direct"
            )
            host.load_to_device_per_layer(
                target, host_indices, device_indices, i, "direct"
            )
        host.load_to_device_per_layer(
            draft, host_indices, device_indices, 6, "direct", is_draft=True
        )
        torch.cuda.synchronize()
        for i in (1, 5):
            torch.testing.assert_close(
                target.index_k_with_scale_buffer[i][1:3], expected[i]
            )
        torch.testing.assert_close(
            draft.index_k_with_scale_buffer[0][1:3], expected_draft
        )
        for i, buf in enumerate(target.kv_buffer):
            self.assertTrue(torch.all(buf[64:192] == i + 1).item())
        self.assertEqual(
            [buf.numel() == 0 for buf in target.index_k_with_scale_buffer],
            [True, False, True, True, True, False],
        )
        host.destroy()
        anchor.destroy()

    @unittest.skipUnless(is_cuda(), "CUDA compact HiSparse regression")
    def test_hisparse_pool_elides_only_shared_indexer_storage(self):
        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool

        mask = [not (i < 3 or (i - 2) % 4 == 0) for i in range(78)]
        pool = HiSparseDSATokenToKVPool(
            size=128,
            page_size=64,
            kv_lora_rank=512,
            dtype=torch.float8_e4m3fn,
            qk_rope_head_dim=64,
            layer_num=78,
            device="cuda",
            index_head_dim=128,
            enable_memory_saver=False,
            kv_cache_dim=656,
            skip_topk_layers=mask,
            host_to_device_ratio=20,
        )
        self.assertEqual(pool.index_buf_size, 2560)
        self.assertEqual(len(pool.kv_buffer), 78)
        self.assertTrue(all(buf.numel() for buf in pool.kv_buffer))
        self.assertEqual(
            sum(buf.nbytes for buf in pool.index_k_with_scale_buffer),
            21 * (2560 + 64) * 132,
        )
        _, lengths, items = pool.get_state_buf_infos()
        self.assertEqual(len(lengths), 78)
        self.assertEqual(sum(length > 0 for length in lengths), 21)
        for i, shared in enumerate(mask):
            if shared:
                self.assertEqual((lengths[i], items[i]), (0, 0))
            else:
                data = pool.get_index_k_with_scale_buffer(i)
                self.assertEqual(data.shape, (41, 64 * 132))


if __name__ == "__main__":
    unittest.main()
