import inspect
import unittest

import torch

from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cuda, is_hip, is_npu, is_xpu
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")
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
        self, io_backend: str, *, layout="layer_first", index_kpool=1, relocate=False
    ):
        page_size = 1 if is_hip() else 64 * index_kpool
        layer_num = 2
        size = page_size * 8

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
            index_kpool=index_kpool,
            index_kpool_compress=index_kpool > 1,
            max_running_requests=1,
        )
        pin_memory = io_backend == "kernel"
        mla_host = MLATokenToKVPoolHost(
            buffer_info=device_pool.get_device_pool_infos()[0].buffer_info,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=page_size,
            layout=layout,
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

        device_pages = torch.tensor([1, 3, 5], device="cuda", dtype=torch.int64)
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

        # A hybrid model can place these two full-attention layers among
        # recurrent layers. Both buffers must be ready at each layer callback.
        full_layer_mapping = {3: 0, 7: 1} if relocate else {0: 0, 1: 1}

        def transfers(indices, *, backup=False):
            return [
                L2Transfer(
                    host_pool=host_pool,
                    device_pool=device_pool,
                    # Match the controller's _move_write_operation: the
                    # staged page-first write-back reads destination IDs on CPU.
                    host_indices=(
                        host_indices.cpu()
                        if backup
                        and io_backend == "kernel"
                        and layout == "page_first"
                        and host_pool.can_use_write_back_jit
                        else host_indices
                    ),
                    device_indices=indices,
                    layer_mapper=full_layer_mapping.get,
                )
                for host_pool in (mla_host, indexer_host)
            ]

        engine = L2TransferEngine(io_backend)
        engine.submit_device_to_host(
            transfers(device_indices, backup=True)
        ).finish_event.synchronize()

        for layer_id in range(layer_num):
            for host_page, device_page in zip(
                host_pages.tolist(), device_pages.tolist()
            ):
                index_buffer = indexer_host.index_k_with_scale_buffer
                got = (
                    (
                        index_buffer[layer_id, host_page]
                        if layout == "layer_first"
                        else index_buffer[host_page, layer_id]
                    )
                    .reshape(-1)
                    .cpu()
                )
                expected = device_pool.index_k_with_scale_buffer[layer_id][
                    device_page
                ].cpu()
                self.assertTrue(torch.equal(got, expected))
                host_start = host_page * page_size
                device_start = device_page * page_size
                host_slice = slice(host_start, host_start + page_size)
                got_kv = (
                    mla_host.kv_buffer[layer_id, host_slice]
                    if layout == "layer_first"
                    else mla_host.kv_buffer[host_slice, layer_id]
                ).cpu()
                expected_kv = device_pool.kv_buffer[layer_id][
                    device_start : device_start + page_size
                ].cpu()
                self.assertTrue(torch.equal(got_kv, expected_kv))

        expected_index_pages = [
            buffer[device_pages].clone()
            for buffer in device_pool.index_k_with_scale_buffer
        ]
        expected_kv = [
            buffer[device_indices].clone() for buffer in device_pool.kv_buffer
        ]
        # Overwrite the device buffers to model reuse by another request. A
        # same-address restore without this step can hide a missing transfer.
        for buffer in device_pool.kv_buffer:
            buffer.fill_(-11)
        for buffer in device_pool.index_k_with_scale_buffer:
            buffer.fill_(205)
        restore_pages = device_pages + 1 if relocate else device_pages
        restore_indices = self._token_indices_for_pages(
            restore_pages, page_size, device="cuda"
        )
        snapshots = {}

        def on_layer_done(layer_id):
            if layer_id in full_layer_mapping:
                local_layer = full_layer_mapping[layer_id]
                # These clones run on the transfer stream at the callback,
                # so a callback before the indexer copy cannot pass later.
                snapshots[local_layer] = (
                    device_pool.kv_buffer[local_layer][restore_indices].clone(),
                    device_pool.index_k_with_scale_buffer[local_layer][
                        restore_pages
                    ].clone(),
                )

        engine.submit_host_to_device(
            transfers(restore_indices),
            transfer_layer_id_max=max(full_layer_mapping) + 1,
            on_layer_done=on_layer_done,
        ).finish_event.synchronize()
        self.assertEqual(set(snapshots), set(range(layer_num)))
        for layer_id, (kv, indexer) in snapshots.items():
            torch.testing.assert_close(kv, expected_kv[layer_id], rtol=0, atol=0)
            torch.testing.assert_close(
                indexer, expected_index_pages[layer_id], rtol=0, atol=0
            )
            untouched_pages = (
                torch.cat((device_pages, device_pages.new_zeros(1)))
                if relocate
                else device_pages.new_zeros(1)
            )
            untouched_indices = self._token_indices_for_pages(
                untouched_pages, page_size, device="cuda"
            )
            self.assertTrue(
                torch.all(device_pool.kv_buffer[layer_id][untouched_indices] == -11)
            )
            self.assertTrue(
                torch.all(
                    device_pool.index_k_with_scale_buffer[layer_id][untouched_pages]
                    == 205
                )
            )

    @unittest.skipUnless(is_cuda(), "CUDA kernels required for relocated DSA pages")
    def test_kv_and_indexer_restore_to_reused_pages_before_layer_ready(self):
        for layout in ("layer_first", "page_first"):
            for index_kpool in (1, 4):
                with self.subTest(layout=layout, index_kpool=index_kpool):
                    self._run_device_to_host_indexer_copy(
                        "kernel",
                        layout=layout,
                        index_kpool=index_kpool,
                        relocate=True,
                    )

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
