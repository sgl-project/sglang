import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.kernels.ops.kvcache.strided_staging import (
    gather_strided_staging,
    scatter_strided_staging,
)
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    _needs_flat_torch_staging,
    _needs_strided_staging,
    gather_all_layers_to_staging,
    scatter_staging_to_kv,
    validate_staging_buffers,
)
from sglang.srt.disaggregation.common.staging_handler import required_staging_bytes
from sglang.srt.disaggregation.utils import (
    build_staging_slot_metadata,
    build_transfer_entry_pairs,
)
from sglang.srt.distributed.utils import get_pp_indices
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.memory.views import MHAKVView
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

PAGE, ROWS = 4, 32
FULL_IDS = list(range(1, 62, 4))
SOURCE_PAGES = np.array([1, 4], dtype=np.int32)
SOURCE_ROWS = [4, 5, 6, 7, 16, 17, 18, 19]
DEST_ROWS = [8, 9, 10, 11, 20, 21, 22, 23]


def _global_bytes(layer, group, dtype):
    draft = layer >= 62
    heads, dim = (8, 128) if draft else (16, 64)
    itemsize = torch.empty((), dtype=dtype).element_size()
    values = torch.arange(
        ROWS * heads * dim * itemsize, device="cuda", dtype=torch.int64
    )
    values = (values + 17 * layer + 73 * group) % 251
    return values.to(torch.uint8).reshape(ROWS, heads, dim * itemsize)


class _Pool(MHAKVView):
    def __init__(
        self, ids, tp, rank, dtype, *, draft=False, strided=False, source=False
    ):
        self.model_layer_ids = tuple(ids)
        self.layer_router = PlainPoolRouter(self, IdSpaceKind.FULL)
        self.start_layer, self.layer_num = 0, len(ids)
        heads, dim = (8 // tp, 128) if draft else (16 // tp, 64)
        size = torch.empty((), dtype=dtype).element_size()
        width = heads * dim * size
        stride = width * (2 * len(ids) + 1) if strided else width
        self.storage = torch.full(
            (ROWS * stride,), 255, dtype=torch.uint8, device="cuda"
        )
        self.k_buffer, self.v_buffer = [], []
        for group, buffers in enumerate((self.k_buffer, self.v_buffer)):
            for index, lid in enumerate(ids):
                if strided:
                    offset = (group * len(ids) + index) * width
                    raw = torch.as_strided(
                        self.storage,
                        (ROWS, heads, dim * size),
                        (stride, dim * size, 1),
                        offset,
                    )
                else:
                    raw = torch.full(
                        (ROWS, heads, dim * size), 255, dtype=torch.uint8, device="cuda"
                    )
                if source:
                    reference = _global_bytes(lid + 62 if draft else lid, group, dtype)
                    raw.copy_(reference[:, rank * heads : (rank + 1) * heads])
                store_dtype = torch.uint8 if dtype == torch.float8_e4m3fn else dtype
                buffers.append(raw.view(store_dtype))


def _slots(target, draft):
    return build_staging_slot_metadata(
        kv_layer_ids=[],
        num_draft_entries=10 if draft is not None else 0,
        kv_pool=target,
        draft_kv_pool=draft,
        num_hidden_layers=62,
    )


class TestPDStridedStaging(CustomTestCase):
    def test_target_and_draft_roundtrip(self):
        cases = (
            (torch.float8_e4m3fn, 2, 4, 8, True, True),
            (torch.bfloat16, 8, 1, 8, True, True),
            (torch.float8_e4m3fn, 1, 1, 1, False, False),
            (torch.bfloat16, 1, 8, 4, False, True),
            (torch.bfloat16, 1, 8, 4, True, True),
            (torch.float8_e4m3fn, 1, 1, 1, True, False),
        )
        for use_triton in (False, True):
            with patch(
                "sglang.srt.disaggregation.common.staging_buffer._USE_TRITON_STAGING",
                use_triton,
            ):
                for case in cases:
                    with self.subTest(use_triton=use_triton, case=case):
                        self._roundtrip(*case)

    def test_invalid_layout_is_rejected(self):
        for shape in ((2, 3, 4, 8), (2, 3, 4, 8, 1)):
            buffers = [torch.zeros(shape)]
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "NHD"):
                validate_staging_buffers(buffers, buffers)
        with self.assertRaisesRegex(RuntimeError, "matching flat KV row widths"):
            validate_staging_buffers([torch.zeros(2, 4, 64)], [torch.zeros(2, 4, 32)])
        legacy = torch.zeros(2, 4, 64)
        self.assertFalse(_needs_strided_staging([legacy], [legacy]))
        self.assertFalse(_needs_flat_torch_staging([legacy], [legacy]))

    def test_strided_byte_offset_above_signed_int32(self):
        offset = 1 << 31
        source = torch.empty(offset + 65, dtype=torch.int8, device="cuda")
        destination = torch.empty_like(source)
        values = torch.arange(64, dtype=torch.int8, device="cuda")
        source[offset : offset + 64].copy_(values)
        destination[offset - 1 : offset + 65].fill_(-1)
        destination[:64].fill_(-1)
        strides = torch.tensor([1 << 30], dtype=torch.int64, device="cuda")
        pages = torch.tensor([2], dtype=torch.int64, device="cuda")
        staging = torch.empty(64, dtype=torch.int8, device="cuda")
        pointers = torch.tensor([source.data_ptr()], dtype=torch.int64, device="cuda")
        gather_strided_staging(
            pointers, strides, pages, staging, 64, 0, elems_per_token=64, page_size=1
        )
        self.assertTrue(torch.equal(staging, values))
        pointers = torch.tensor(
            [destination.data_ptr()], dtype=torch.int64, device="cuda"
        )
        offsets = torch.zeros(1, dtype=torch.int64, device="cuda")
        scatter_strided_staging(
            pointers,
            strides,
            pages,
            staging,
            offsets,
            64,
            elems_per_token=64,
            page_size=1,
            num_slots=1,
            num_writers=1,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(destination[offset : offset + 64], values))
        self.assertTrue(torch.all(destination[:64] == -1).item())
        self.assertEqual(destination[offset - 1].item(), -1)
        self.assertEqual(destination[offset + 64].item(), -1)

    def _roundtrip(self, dtype, pp, src_tp, dst_tp, strided, with_draft):
        size = torch.empty((), dtype=dtype).element_size()
        destinations = range(dst_tp)
        num_writers = max(1, src_tp // dst_tp)
        wire_heads = 16 // max(src_tp, dst_tp)
        per_slot = len(SOURCE_ROWS) * wire_heads * 64 * size
        for rank in destinations:
            target = _Pool(FULL_IDS, dst_tp, rank, dtype, strided=strided)
            draft = (
                _Pool(list(range(5)), dst_tp, rank, dtype, draft=True, strided=strided)
                if with_draft
                else None
            )
            dst_ks, dst_vs, dst_ids = _slots(target, draft)
            assembled = torch.full(
                (num_writers * len(dst_ids) * per_slot,),
                254,
                dtype=torch.uint8,
                device="cuda",
            )
            args = SimpleNamespace(
                total_kv_head_num=16,
                kv_head_num=16 // dst_tp,
                page_size=PAGE,
                engine_rank=rank,
            )
            required = required_staging_bytes(
                args,
                {"k_buffers": dst_ks, "v_buffers": dst_vs},
                src_tp,
                dst_tp,
                len(SOURCE_PAGES),
            )
            self.assertEqual(required, assembled.numel())
            for writer in range(num_writers):
                source_rank = (
                    rank * num_writers + writer
                    if src_tp > dst_tp
                    else rank // (dst_tp // src_tp)
                )
                head_start = (
                    0 if src_tp >= dst_tp else rank % (dst_tp // src_tp) * wire_heads
                )
                for stage in range(pp):
                    start, end = get_pp_indices(62, stage, pp)
                    ids = [lid for lid in FULL_IDS if start <= lid < end]
                    source = _Pool(
                        ids, src_tp, source_rank, dtype, strided=strided, source=True
                    )
                    source_draft = (
                        _Pool(
                            list(range(5)),
                            src_tp,
                            source_rank,
                            dtype,
                            draft=True,
                            strided=strided,
                            source=True,
                        )
                        if with_draft and stage == pp - 1
                        else None
                    )
                    ks, vs, slot_ids = _slots(source, source_draft)
                    snapshots = [buf.view(torch.uint8).clone() for buf in ks + vs]
                    staging = StagingBuffer(len(slot_ids) * per_slot, "cuda", 0)
                    written = gather_all_layers_to_staging(
                        ks, vs, SOURCE_PAGES, staging, head_start, wire_heads, PAGE
                    )
                    self.assertEqual(written, len(slot_ids) * per_slot)
                    for index, buf in enumerate(ks + vs):
                        expected = buf.view(torch.uint8)[SOURCE_ROWS].reshape(
                            len(SOURCE_ROWS), -1
                        )
                        begin = head_start * 64 * size
                        expected = (
                            expected[:, begin : begin + wire_heads * 64 * size]
                            .contiguous()
                            .flatten()
                        )
                        self.assertTrue(
                            torch.equal(
                                staging.buffer[
                                    index * per_slot : (index + 1) * per_slot
                                ],
                                expected,
                            )
                        )
                        self.assertTrue(
                            torch.equal(buf.view(torch.uint8), snapshots[index])
                        )
                    for i, j in build_transfer_entry_pairs(
                        slot_ids, dst_ids, len(slot_ids), len(dst_ids)
                    ):
                        begin = (writer * len(dst_ids) + j) * per_slot
                        assembled[begin : begin + per_slot].copy_(
                            staging.buffer[i * per_slot : (i + 1) * per_slot]
                        )
            pages = torch.tensor([2, 5], device="cuda", dtype=torch.int64)
            scatter_staging_to_kv(
                assembled, dst_ks, dst_vs, pages, PAGE, src_tp, dst_tp, rank, 16
            )
            torch.cuda.synchronize()
            if strided:
                for pool in (target, draft):
                    if pool is not None:
                        width = pool.k_buffer[0].numel() // ROWS * size
                        self.assertTrue(
                            torch.all(
                                pool.storage.view(ROWS, -1)[:, -width:] == 255
                            ).item()
                        )
            for group, buffers in enumerate((dst_ks, dst_vs)):
                for lid, buf in zip(dst_ids[: len(dst_ks)], buffers):
                    heads = buf.shape[1]
                    expected = _global_bytes(lid, group, dtype)[
                        SOURCE_ROWS, rank * heads : (rank + 1) * heads
                    ]
                    self.assertTrue(
                        torch.equal(buf.view(torch.uint8)[DEST_ROWS], expected)
                    )
                    untouched = torch.ones(ROWS, dtype=torch.bool, device="cuda")
                    untouched[DEST_ROWS] = False
                    self.assertTrue(
                        torch.all(buf.view(torch.uint8)[untouched] == 255).item()
                    )


if __name__ == "__main__":
    unittest.main()
