"""Regression coverage for cache hits larger than a DCP pack buffer."""

import concurrent.futures
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.common.dcp_pack import iter_dcp_transfer_chunks
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestChunkPlanning(CustomTestCase):
    def test_fragmented_destination_and_partial_tail(self):
        src = np.arange(1, 22, 2, dtype=np.int64)
        dst = np.array([64, 65, 66, 128, 129, 130, 3, 4, 5, 6, 7])
        packed_rows = []
        buf = SimpleNamespace(get_size=lambda: 3 * (16 + 32))

        def pack(**kwargs):
            rows = kwargs["src_token_indices"].copy()
            packed_rows.append(rows)
            return [1000, 2000], np.arange(rows.size)

        with patch(
            "sglang.srt.disaggregation.common.dcp_pack.try_pack_dcp_src",
            side_effect=pack,
        ):
            chunks = list(
                iter_dcp_transfer_chunks(
                    pack_buffer=buf,
                    kv_data_ptrs=[10, 20],
                    src_token_indices=src,
                    dst_token_indices=dst,
                    token_item_lens=[16, 32],
                )
            )
        self.assertEqual([len(c[1]) for c in chunks], [3, 3, 3, 2])
        np.testing.assert_array_equal(np.concatenate(packed_rows), src)
        np.testing.assert_array_equal(np.concatenate([c[2] for c in chunks]), dst)

    def test_empty_and_too_small_buffer(self):
        args = dict(kv_data_ptrs=[1], token_item_lens=[16])
        empty = np.array([], dtype=np.int64)
        self.assertEqual(
            list(
                iter_dcp_transfer_chunks(
                    pack_buffer=None,
                    src_token_indices=empty,
                    dst_token_indices=empty,
                    **args
                )
            ),
            [],
        )
        with self.assertRaisesRegex(ValueError, "one token"):
            list(
                iter_dcp_transfer_chunks(
                    pack_buffer=SimpleNamespace(get_size=lambda: 15),
                    src_token_indices=np.array([1]),
                    dst_token_indices=np.array([2]),
                    **args
                )
            )

    def test_failed_layer_drains_other_sends_before_reusing_pack(self):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        second_started = threading.Event()
        second_finished = threading.Event()
        release_second = threading.Event()
        advance = []

        def chunks(**kwargs):
            advance.append(1)
            yield [100, 200], np.array([0]), np.array([0])
            advance.append(2)
            yield [100, 200], np.array([0]), np.array([1])

        def transfer(session, blocks):
            if blocks[0][0] == 100:
                self.assertTrue(second_started.wait(5))
                return -1
            second_started.set()
            self.assertTrue(release_second.wait(5))
            second_finished.set()
            return 0

        mgr = SimpleNamespace(
            kv_args=SimpleNamespace(
                page_size=64, kv_layer_ids=None, kv_data_ptrs=[10, 20]
            ),
            enable_custom_mem_pool=True,
            enable_deferred_decode_kv_release=False,
            get_mla_kv_ptrs_with_pp=lambda source, target: (source, target, 2),
            _transfer_data=transfer,
        )
        mgr._await_transfer_futures = (
            lambda futures: MooncakeKVManager._await_transfer_futures(mgr, futures)
        )
        timer = threading.Timer(0.25, release_second.set)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor, patch(
            "sglang.srt.disaggregation.common.dcp_pack.iter_dcp_transfer_chunks",
            side_effect=chunks,
        ):
            timer.start()
            try:
                ret = MooncakeKVManager.send_kvcache_dcp(
                    mgr,
                    "unit-test",
                    np.array([0], dtype=np.int32),
                    [30, 40],
                    np.array([0], dtype=np.int32),
                    dcp_token_item_lens=[16, 32],
                    dst_dcp_size=2,
                    dst_dcp_rank=0,
                    src_page_offset=0,
                    decode_prefix_len=0,
                    num_kv_tokens=4,
                    executor=executor,
                    dst_layer_ids=None,
                    pack_buffer=object(),
                )
                self.assertEqual(ret, -1)
                self.assertTrue(second_finished.is_set())
                self.assertEqual(advance, [1])
            finally:
                release_second.set()
                timer.join()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for byte-exact gather")
class TestPackedMoocakeSend(CustomTestCase):
    def run_send(self, rank, fail_at=None, custom_pool=False):
        from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager

        # Different row widths exercise byte layout, including GLM FP8 KV rows.
        widths = [656, 128]
        torch.manual_seed(43)
        src = [
            torch.randint(0, 256, (768, w), dtype=torch.uint8, device="cuda")
            for w in widths
        ]
        dst = [torch.full((1024, w), 251, dtype=torch.uint8) for w in widths]
        expected = [x.clone() for x in dst]
        src_pages = np.array([7, 2, 5, 1, 9], dtype=np.int32)
        dst_pages = np.array([9, 3, 12], dtype=np.int32)
        num_tokens, page_offset, prefix = 257, 1, 128
        buf = StagingBuffer(37 * sum(widths), "cuda:0", 0)
        calls = []

        def transfer(session, blocks):
            calls.append(blocks)
            if fail_at is not None and len(calls) == fail_at:
                return -1
            for source, target, nbytes in blocks:
                packed_offset = source - buf.get_ptr()
                self.assertGreaterEqual(packed_offset, 0)
                self.assertLessEqual(packed_offset + nbytes, buf.get_size())
                for tensor in dst:
                    dst_offset = target - tensor.data_ptr()
                    if 0 <= dst_offset < tensor.numel():
                        tensor.view(-1)[dst_offset : dst_offset + nbytes].copy_(
                            buf.buffer[packed_offset : packed_offset + nbytes].cpu()
                        )
                        break
                else:
                    self.fail("Destination address is outside the selected PP layers")
            return 0

        mgr = SimpleNamespace(
            kv_args=SimpleNamespace(
                page_size=64,
                kv_layer_ids=None,
                kv_data_ptrs=[x.data_ptr() for x in src],
            ),
            enable_custom_mem_pool=custom_pool,
            get_mla_kv_ptrs_with_pp=lambda source, target: (
                source,
                target,
                len(source),
            ),
            _transfer_data=transfer,
            _await_transfer_futures=lambda futures: next(
                (r for r in [f.result() for f in futures] if r != 0), 0
            ),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            ret = MooncakeKVManager.send_kvcache_dcp(
                mgr,
                "unit-test",
                src_pages,
                [x.data_ptr() for x in dst],
                dst_pages,
                dcp_token_item_lens=widths,
                dst_dcp_size=2,
                dst_dcp_rank=rank,
                src_page_offset=page_offset,
                decode_prefix_len=prefix,
                num_kv_tokens=num_tokens,
                executor=executor,
                dst_layer_ids=None,
                pack_buffer=buf,
            )
        if fail_at is not None:
            self.assertEqual(ret, -1)
            self.assertEqual(len(calls), fail_at)
            return
        self.assertEqual(ret, 0)
        for offset in range(num_tokens):
            logical = prefix + page_offset * 64 + offset
            if logical % 2 != rank:
                continue
            source = int(src_pages[offset // 64]) * 64 + offset % 64
            local = (page_offset * 64 + offset) // 2
            target = int(dst_pages[local // 64]) * 64 + local % 64
            for i in range(len(src)):
                expected[i][target].copy_(src[i][source].cpu())
        for actual, reference in zip(dst, expected):
            self.assertTrue(torch.equal(actual, reference))
        self.assertEqual(len(calls), 8 if custom_pool else 4)

    def test_both_dcp_owners_with_nonzero_prefix_and_partial_page(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                self.run_send(rank)

    def test_failed_chunk_stops_before_buffer_reuse(self):
        self.run_send(0, fail_at=2)

    def test_parallel_layer_sends_finish_before_buffer_reuse(self):
        self.run_send(1, custom_pool=True)


if __name__ == "__main__":
    unittest.main()
