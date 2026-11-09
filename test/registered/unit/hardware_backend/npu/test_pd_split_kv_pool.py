import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


def _make_pool():
    pool = NPUMLATokenToKVPool.__new__(NPUMLATokenToKVPool)
    pool.start_layer = 0
    pool.layer_num = 2
    pool.device = "cpu"
    pool.page_size = 128
    pool.cpu_offloading_chunk_size = 2
    pool.dcp_size = 1
    pool.is_draft_worker = False
    pool.dsa_kv_cache_store_fp8 = False
    pool.index_head_dim = 2
    pool.indexer_layer_id_to_slot = {0: 0, 1: 1}
    pool.k_buffer = torch.zeros(2, 2, 128, 1, 512, dtype=torch.bfloat16)
    pool.v_buffer = torch.zeros(2, 2, 128, 1, 64, dtype=torch.bfloat16)
    pool.index_k_buffer = torch.zeros(2, 2, 128, 1, 2, dtype=torch.bfloat16)
    pool.index_k_scale_buffer = None
    return pool


class TestPDSplitKVPool(unittest.TestCase):
    def test_export_matches_prefill_page_order(self):
        prefill, decode = _make_pool(), _make_pool()
        k, v = list(decode.k_buffer), list(decode.v_buffer)
        decode.sparse_kv_manager = SimpleNamespace(
            host_pool=SimpleNamespace(buffers=(k, v))
        )
        decode.k_buffer = decode.v_buffer = None

        ptrs, lens, page_bytes = decode.get_contiguous_buf_infos()
        self.assertEqual(
            ptrs, [t.data_ptr() for t in k + v + list(decode.index_k_buffer)]
        )
        self.assertEqual((lens, page_bytes), prefill.get_contiguous_buf_infos()[1:])
        self.assertEqual(page_bytes, [131072] * 2 + [16384] * 2 + [512] * 2)

    def test_retraction_restores_shared_payload_and_local_indexer(self):
        pool = _make_pool()
        k, v = list(pool.k_buffer), list(pool.v_buffer)
        published, resets = [], []

        def copy_rows(sources, destinations, source_rows, destination_rows, valid):
            for src, dst in zip(sources, destinations, strict=True):
                dst[destination_rows[valid]] = src[source_rows[valid]]

        host_pool = SimpleNamespace(
            buffers=(k, v),
            rank=0,
            copy_rows=copy_rows,
            publish=lambda: published.append(True),
        )
        pool.sparse_kv_manager = SimpleNamespace(
            host_pool=host_pool, reset_requests=resets.append
        )
        pool.k_buffer = pool.v_buffer = None
        src = torch.tensor([129, 17, 250])
        dst = torch.tensor([5, 134, 128])
        buffers = [t.flatten(0, 1) for t in k + v + list(pool.index_k_buffer)]
        for i, buf in enumerate(buffers):
            buf[src] = torch.tensor([i + 1, i + 2, i + 3], dtype=buf.dtype).view(
                -1, 1, 1
            )
        expected = [buf[src].clone() for buf in buffers]

        with patch.object(
            torch, "npu", SimpleNamespace(synchronize=lambda: None), create=True
        ):
            snapshot = pool.get_cpu_copy(src)
            for buf in buffers:
                buf.zero_()
            pool.load_cpu_copy(snapshot, dst, req_pool_index=3)

            for buf, want in zip(buffers, expected, strict=True):
                torch.testing.assert_close(buf[dst], want, rtol=0, atol=0)

            host_pool.rank = 1
            for buf in buffers:
                buf.zero_()
            pool.load_cpu_copy(snapshot, dst, req_pool_index=4)

        self.assertEqual(published, [True, True])
        self.assertEqual(resets, [[3], [4]])
        for buf in buffers[:4]:
            self.assertEqual(torch.count_nonzero(buf).item(), 0)
        for buf, want in zip(buffers[4:], expected[4:], strict=True):
            torch.testing.assert_close(buf[dst], want, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
