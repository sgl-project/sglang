import concurrent.futures
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.kernels.ops.kvcache.mla_buffer import unpack_dcp_kv
from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.layers.dcp.comm import (
    all_gather_kv_cache_for_dcp,
    all_gather_kv_cache_for_mha_chunk_extend,
    all_gather_kv_cache_for_mha_extend,
    all_gather_kv_cache_for_mla_extend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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


class TestDcpGather(CustomTestCase):
    def test_final_layouts(self):
        # Protect rank/token mapping, unaligned chunk starts, empty requests,
        # strided destinations, FP8 transport, and prefix/suffix boundaries.
        for world in (1, 2, 4, 8):
            for dtype in (
                torch.bfloat16,
                torch.float16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ):
                for split in (False, True):
                    with self.subTest(world=world, dtype=dtype, split=split):
                        self._check_layout(world, dtype, split)

    def test_long_prefix_grid(self):
        # Token tiles must use grid.x: grid.y is limited to 65535 CTAs.
        self._check_layout(8, torch.bfloat16, True, long_prefix=True)

    def _check_layout(self, world, dtype, split, long_prefix=False):
        lengths, starts, suffixes = [19, 0, 4099, 1], [3, 0, 17, 2], [2, 3, 1, 0]
        if long_prefix:
            lengths, starts, suffixes = [262145], [3], [2]
        k_dim, pe_dim = 512, 64
        dim = k_dim + pe_dim
        rank_parts = [[] for _ in range(world)]
        metadata, expected = [], []
        padded_start = output_start = 0
        generator = torch.Generator().manual_seed(17)
        for n, start, suffix in zip(lengths, starts, suffixes):
            padded = ((start % world + n + world - 1) // world) * world
            rows = torch.randn(padded, 1, dim, generator=generator).to(dtype)
            # Striding a logical token sequence is the independent shard oracle.
            for rank in range(world):
                rank_parts[rank].append(rows.float()[rank::world])
            metadata.append([padded_start, start % world, n, output_start])
            expected.extend(
                (
                    rows.float()[start % world : start % world + n],
                    torch.full((suffix, 1, dim), -7.0),
                )
            )
            padded_start += padded // world
            output_start += n + suffix
        gathered = torch.cat([torch.cat(parts) for parts in rank_parts]).to(
            device="cuda", dtype=dtype
        )
        metadata = torch.tensor(
            metadata, dtype=torch.int64, device="cpu" if long_prefix else "cuda"
        )
        expected = torch.cat(expected).cuda()
        output_dtype = torch.bfloat16 if split else dtype
        if split:
            outputs = (
                torch.full(
                    (output_start, 1, k_dim), -7, device="cuda", dtype=output_dtype
                ),
                torch.full(
                    (output_start, 1, pe_dim), -7, device="cuda", dtype=output_dtype
                ),
            )
        else:
            combined = torch.full(
                (output_start, 1, dim), -7, device="cuda", dtype=output_dtype
            )
            outputs = combined.split([k_dim, pe_dim], dim=-1)
        unpack_dcp_kv(gathered, metadata, *outputs, world, max(lengths))
        actual = torch.cat([part.float() for part in outputs], dim=-1)
        torch.testing.assert_close(
            actual, expected.to(output_dtype).float(), rtol=0, atol=0
        )

    def test_ragged_consumers(self):
        for dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ):
            for lengths, starts in (
                ([4096], [0]),
                ([2731, 0, 19], [3, 0, 17]),
                ([1, 0, 1], [0, 0, 0]),
                ([0, 0, 0], [0, 0, 0]),
            ):
                for rank in range(4):
                    with self.subTest(
                        dtype=dtype, lengths=lengths, starts=starts, rank=rank
                    ):
                        self._check_consumers(dtype, lengths, starts, rank)

    def test_portable_direct_layout(self):
        # Exercise the portable copy path on CUDA, using the same shard oracle.
        with patch("sglang.srt.layers.dcp.comm._is_hip", True):
            for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                for lengths, starts in (
                    ([4096], [0]),
                    ([19, 0, 1], [3, 0, 17]),
                    ([19, 0, 1], [0, 0, 0]),
                ):
                    for rank in range(4):
                        with self.subTest(
                            dtype=dtype, lengths=lengths, starts=starts, rank=rank
                        ):
                            self._check_consumers(dtype, lengths, starts, rank)

    def _check_consumers(self, dtype, lengths, starts, rank):
        parallel = SimpleNamespace(dcp_enabled=True, dcp_size=4, dcp_rank=rank)
        generator = torch.Generator().manual_seed(731)
        rows = [
            torch.randn(n, 1, 576, generator=generator).to(dtype).float()
            for n in lengths
        ]
        # Emulate only the collective boundary. Build rank-major data from
        # logical positions, and verify this rank's real send-buffer contents.
        parts, masks = [[] for _ in range(4)], [[] for _ in range(4)]
        for values, start in zip(rows, starts):
            positions = torch.arange(start // 4 * 4, (start + len(values) + 3) // 4 * 4)
            valid = (positions >= start) & (positions < start + len(values))
            padded = torch.full((len(positions), 1, 576), float("nan"))
            padded[valid] = values
            for r in range(4):
                parts[r].append(padded[r::4])
                masks[r].append(valid[r::4])
        gathered = torch.cat([torch.cat(part) for part in parts]).to(
            device="cuda", dtype=dtype
        )
        local_valid = torch.cat(masks[rank]).cuda()

        def all_gather_into_tensor(output, send):
            reference = (
                gathered.view(torch.uint8)
                if send.dtype == torch.uint8
                else gathered.to(send.dtype)
            )
            expected_send = reference.view(4, *send.shape)[rank]
            torch.testing.assert_close(
                send[local_valid], expected_send[local_valid], rtol=0, atol=0
            )
            output.copy_(reference)

        parallel.dcp_group = SimpleNamespace(
            all_gather_into_tensor=all_gather_into_tensor
        )
        with patch("sglang.srt.layers.dcp.comm.get_parallel", return_value=parallel):
            local = [
                x[(parallel.dcp_rank - start) % parallel.dcp_size :: parallel.dcp_size]
                for x, start in zip(rows, starts)
            ]
            local = torch.cat(local).to(device="cuda", dtype=dtype)
            expected = torch.cat(rows).cuda()
            k, pe = local.split([512, 64], dim=-1)
            lens, offsets = torch.tensor(lengths), torch.tensor(starts)
            combined_prefix = local.new_empty(sum(lengths), 1, 576)
            all_gather_kv_cache_for_dcp(
                k, pe, lens, offsets, output=combined_prefix.split([512, 64], dim=-1)
            )
            actual = all_gather_kv_cache_for_mha_chunk_extend(
                k.squeeze(1), pe, lens, offsets
            )
            torch.testing.assert_close(
                combined_prefix.float(), expected, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[0].float(), expected[:, 0, :512], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual[1].float(), expected[..., 512:], rtol=0, atol=0
            )
            self.assertTrue(all(x.is_contiguous() for x in actual))

            if any(starts):
                return
            # The cache reader is a boundary here; exercise the real merge/layout
            # consumers with exact local KV, including ranks owning no prefix rows.
            pool = SimpleNamespace(
                get_mla_kv_buffer=lambda *args, dst_dtype=None: tuple(
                    x.to(dst_dtype or dtype) for x in (k, pe)
                )
            )
            extend_lens = [2, 1, 3][: len(lengths)]
            suffix = (
                torch.randn(sum(extend_lens), 1, 576, generator=generator)
                .cuda()
                .to(torch.bfloat16)
            )
            combined = torch.empty(
                sum(lengths) + sum(extend_lens), 1, 576, device="cuda", dtype=dtype
            )
            all_gather_kv_cache_for_mla_extend(
                pool,
                None,
                lengths,
                None,
                sum(lengths),
                combined,
                512,
                suffix[..., :512],
                suffix[..., 512:],
            )
            torch.testing.assert_close(
                combined.float(),
                torch.cat([expected, suffix.to(dtype).float()]),
                rtol=0,
                atol=0,
            )
            outputs = all_gather_kv_cache_for_mha_extend(
                pool,
                None,
                None,
                lengths,
                extend_lens,
                suffix[:, 0, :512],
                suffix[..., 512:],
            )
            expected_mha = torch.cat(
                [
                    part
                    for pair in zip(
                        expected.to(torch.bfloat16).split(lengths),
                        suffix.split(extend_lens),
                    )
                    for part in pair
                ]
            )
            torch.testing.assert_close(
                outputs[0], expected_mha[:, 0, :512], rtol=0, atol=0
            )
            torch.testing.assert_close(
                outputs[1], expected_mha[..., 512:], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
