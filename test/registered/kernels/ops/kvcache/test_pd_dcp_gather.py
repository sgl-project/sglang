import concurrent.futures
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import (
    copy_mla_rows_into_pack,
    copy_pack_into_mla_rows,
    mla_pack_required_bytes,
)
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestPdDcpGather(CustomTestCase):
    def test_gathers_strided_rows_layer_major(self):
        dim = 8
        kv0 = torch.arange(32 * dim, dtype=torch.float32, device="cuda").view(
            32, 1, dim
        )
        kv1 = torch.arange(32 * 5, dtype=torch.float16, device="cuda").view(32, 1, 5)
        row_indices = torch.tensor([0, 4, 9, 12], dtype=torch.int64, device="cuda")
        item_lens = [int(kv0[0].nbytes), int(kv1[0].nbytes)]
        pack = torch.zeros(
            row_indices.numel() * sum(item_lens), dtype=torch.uint8, device="cuda"
        )

        copy_mla_rows_into_pack(
            [kv0.data_ptr(), kv1.data_ptr()],
            row_indices,
            pack,
            item_lens,
        )
        torch.cuda.synchronize()

        split = row_indices.numel() * item_lens[0]
        packed0 = pack[:split].view(torch.float32).view(4, 1, dim)
        packed1 = pack[split:].view(torch.float16).view(4, 1, 5)
        torch.testing.assert_close(packed0, kv0[row_indices], rtol=0, atol=0)
        torch.testing.assert_close(packed1, kv1[row_indices], rtol=0, atol=0)

    def test_scatter_round_trip_with_pack_offset(self):
        src = torch.arange(16 * 12, dtype=torch.uint8, device="cuda").view(16, 12)
        dst = torch.full_like(src, 0xA5)
        src_rows = torch.tensor([1, 5, 9], dtype=torch.int64, device="cuda")
        dst_rows = torch.tensor([2, 7, 11], dtype=torch.int64, device="cuda")
        prefix = 37
        pack = torch.zeros(
            prefix + src_rows.numel() * 12, dtype=torch.uint8, device="cuda"
        )

        copy_mla_rows_into_pack(
            [src.data_ptr()],
            src_rows,
            pack,
            [12],
            pack_offset_bytes=prefix,
        )
        copy_pack_into_mla_rows(
            pack,
            [dst.data_ptr()],
            dst_rows,
            [12],
            pack_offset_bytes=prefix,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(dst[dst_rows], src[src_rows], rtol=0, atol=0)
        untouched = torch.ones(16, dtype=torch.bool, device="cuda")
        untouched[dst_rows] = False
        self.assertTrue(torch.all(dst[untouched] == 0xA5))

    def test_gather_rejects_oversize_before_launch_and_preserves_guard(self):
        source = torch.arange(64, dtype=torch.uint8, device="cuda").view(8, 8)
        rows = torch.tensor([1, 3, 5], dtype=torch.int64, device="cuda")
        required = mla_pack_required_bytes(rows.numel(), [8], 4)
        storage = torch.full((required + 16,), 165, dtype=torch.uint8, device="cuda")

        with self.assertRaisesRegex(ValueError, "overflow before gather"):
            copy_mla_rows_into_pack(
                [source.data_ptr()],
                rows,
                storage[: required - 1],
                [8],
                pack_offset_bytes=4,
            )
        self.assertTrue(torch.all(storage == 165))
        copy_mla_rows_into_pack(
            [source.data_ptr()], rows, storage[:required], [8], pack_offset_bytes=4
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(storage[:4], torch.full_like(storage[:4], 165))
        torch.testing.assert_close(
            storage[required:], torch.full_like(storage[required:], 165)
        )

    def test_bounded_gather_scatter_preserves_guards(self):
        source = torch.arange(16 * 12, dtype=torch.uint8, device="cuda").view(16, 12)
        destination = torch.full_like(source, 165)
        state = torch.arange(16, dtype=torch.uint8, device="cuda").view(4, 4)
        state_dst = torch.full_like(state, 165)
        src_rows = torch.tensor([15, 2, 7, 1, 14, 4, 11, 0, 5, 8], device="cuda")
        dst_rows = src_rows.flip(0)
        cap, slot, guard = 3, 44, 16
        tx = torch.full((slot + 2 * guard,), 165, dtype=torch.uint8, device="cuda")
        rx = torch.full_like(tx, 165)
        tx_slot, rx_slot = tx[guard:-guard], rx[guard:-guard]
        for start in range(0, len(src_rows), cap):
            rows = src_rows[start : start + cap]
            targets = dst_rows[start : start + cap]
            last = start + cap >= len(src_rows)
            nbytes = len(rows) * 12
            self.assertLessEqual(nbytes + (8 if last else 0), slot)
            copy_mla_rows_into_pack([source.data_ptr()], rows, tx_slot, [12])
            if last:
                copy_mla_rows_into_pack(
                    [state.data_ptr()],
                    torch.tensor([1, 3], device="cuda"),
                    tx_slot,
                    [4],
                    pack_offset_bytes=nbytes,
                )
            rx_slot.copy_(tx_slot)
            copy_pack_into_mla_rows(rx_slot, [destination.data_ptr()], targets, [12])
            if last:
                copy_pack_into_mla_rows(
                    rx_slot,
                    [state_dst.data_ptr()],
                    torch.tensor([0, 2], device="cuda"),
                    [4],
                    pack_offset_bytes=nbytes,
                )
        torch.cuda.synchronize()
        expected = torch.full_like(source, 165)
        expected[dst_rows] = source[src_rows]
        torch.testing.assert_close(destination, expected, rtol=0, atol=0)
        expected_state = torch.full_like(state, 165)
        expected_state[[0, 2]] = state[[1, 3]]
        torch.testing.assert_close(state_dst, expected_state, rtol=0, atol=0)
        for storage in (tx, rx):
            self.assertTrue(torch.all(storage[:guard] == 165))
            self.assertTrue(torch.all(storage[-guard:] == 165))

    def test_packed_tp2_pp2_to_dcp4_preserves_kv(self):
        """Packing must preserve both target rows and draft head shards across PP stages."""
        for custom_pool in (False, True):
            for capacity in (256 * (2 * 64 + 2 * 512), 256 * 2 * 64 // 4):
                for rank in range(4):
                    with self.subTest(
                        custom_pool=custom_pool, capacity=capacity, rank=rank
                    ):
                        self._check_packed_transfer(rank, custom_pool, capacity)

    def _check_packed_transfer(self, rank, custom_pool, capacity):
        page, tokens, chunk = 64, 521, 256
        src_pages = np.array([7, 1, 9, 3, 4, 11, 2, 5, 8], dtype=np.int32)
        dst_pages = np.array([4, 1, 6], dtype=np.int32)
        layers, widths = [3, 11, 19, 27, 28, 28], [64] * 4 + [256] * 2
        logical = torch.arange(tokens, device="cuda")
        src_rows = (
            torch.as_tensor(src_pages, device="cuda")[logical // page] * page
            + logical % page
        )
        values = [
            (
                (logical[:, None] + 256) * 13
                + torch.arange(width, device="cuda") * 7
                + entry * 31
            )
            .remainder(251)
            .to(torch.uint8)
            for entry, width in enumerate([64] * 4 + [1024] * 2)
        ]
        destinations = [
            torch.full((2048, w), 165, dtype=torch.uint8, device="cuda") for w in widths
        ]
        expected = [x.clone() for x in destinations]
        owned = logical[rank::4]
        target_rows = (
            torch.as_tensor(dst_pages, device="cuda")[owned // 256] * page
            + owned % 256 // 4
        )
        draft_rows = (
            torch.as_tensor(dst_pages, device="cuda")[logical // 256] * 256
            + logical % 256
        )
        for entry in range(4):
            expected[entry][target_rows] = values[entry][owned]
        for entry in (4, 5):
            expected[entry][draft_rows] = values[entry][
                :, rank * 256 : (rank + 1) * 256
            ]

        pack = StagingBuffer(capacity, "cuda:0", 0)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            for stage, entries in enumerate(([0, 1], [2, 3, 4, 5])):
                sources = []
                for entry in entries:
                    data = values[entry]
                    if entry >= 4:
                        start = (rank // 2) * 512
                        data = data[:, start : start + 512]
                    source = torch.full(
                        (1024, data.shape[1]), 165, dtype=torch.uint8, device="cuda"
                    )
                    source[src_rows] = data
                    sources.append(source)
                buffers = sources + destinations + [pack.buffer]

                def transfer(session, blocks):
                    def view(ptr, size):
                        for tensor in buffers:
                            offset = ptr - tensor.data_ptr()
                            if 0 <= offset and offset + size <= tensor.numel():
                                return tensor.flatten()[offset : offset + size]
                        raise AssertionError(
                            f"Transfer outside registered buffers: {ptr}, {size}"
                        )

                    for src, dst, size in blocks:
                        view(dst, size).copy_(view(src, size))
                    torch.cuda.synchronize()
                    return 0

                manager = SimpleNamespace(
                    is_mla_backend=False,
                    kv_args=SimpleNamespace(
                        page_size=page,
                        kv_layer_ids=[layers[e] for e in entries],
                        kv_data_ptrs=[x.data_ptr() for x in sources],
                        num_draft_entries=2 if stage else 0,
                        engine_rank=stage * 2 + rank // 2,
                    ),
                    attn_tp_size=2,
                    max_transfer_batch_indices=37,
                    enable_custom_mem_pool=custom_pool,
                    enable_deferred_decode_kv_release=False,
                    _transfer_data=transfer,
                )
                manager._await_transfer_futures = lambda futures: (
                    MooncakeKVManager._await_transfer_futures(manager, futures)
                )
                for start in range(0, tokens, chunk):
                    count = min(chunk, tokens - start)
                    result = MooncakeKVManager.send_kvcache_dcp(
                        manager,
                        "session",
                        src_pages[start // page : (start + count + page - 1) // page],
                        [x.data_ptr() for x in destinations],
                        dst_pages,
                        dcp_token_item_lens=[x.shape[1] for x in sources],
                        dst_dcp_size=4,
                        dst_dcp_rank=rank,
                        src_page_offset=start // page,
                        decode_prefix_len=256,
                        num_kv_tokens=count,
                        executor=executor,
                        dst_layer_ids=layers,
                        pack_buffer=pack,
                        dst_kv_item_lens=[
                            page * w * (4 if e >= 4 else 1)
                            for e, w in enumerate(widths)
                        ],
                        dst_tp_rank=rank,
                        dst_attn_tp_size=4,
                    )
                    self.assertEqual(result, 0)
                for entry in entries:
                    torch.testing.assert_close(
                        destinations[entry], expected[entry], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
