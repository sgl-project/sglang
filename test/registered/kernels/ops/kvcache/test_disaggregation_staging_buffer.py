"""Tests for the aligned staging gather/scatter kernels."""

import unittest
from unittest import mock

import numpy as np
import torch

import sglang.srt.disaggregation.common.staging_buffer as staging_buffer_module
from sglang.srt.disaggregation.common.staging_buffer import (
    StagingBuffer,
    _can_use_aligned_staging_copy,
    _gather_all_layers_triton,
    _scatter_staging_to_kv_triton,
    kv_buffers_preserve_16_byte_alignment,
)
from sglang.srt.disaggregation.common.staging_handler import (
    STAGING_COPY_BUFFERS_ALIGNED_16_KEY,
    build_staging_kv_buffer_info,
)
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class _AlignedDispatchCapture:
    def __init__(self, kernel):
        self.kernel = kernel
        self.values = []

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def capture(*args, **kwargs):
            self.values.append(kwargs.get("ALIGNED_16", args[-1]))
            return launch(*args, **kwargs)

        return capture


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDisaggregationStagingBuffer(CustomTestCase):
    NUM_LAYERS = 2
    PAGE_SIZE = 4
    POOL_TOKENS = 32
    TOTAL_HEADS = 8
    HEAD_DIM = 128

    def _buffers(
        self,
        *,
        storage_offset: int = 0,
        head_dim: int = HEAD_DIM,
        dtype: torch.dtype = torch.bfloat16,
    ):
        stride_pool_token = self.TOTAL_HEADS * head_dim
        raw = [
            torch.empty(
                storage_offset + self.POOL_TOKENS * stride_pool_token,
                dtype=dtype,
                device="cuda",
            )
            for _ in range(2 * self.NUM_LAYERS)
        ]
        for tensor in raw:
            tensor.random_(0, 100)
        buffers = [
            torch.as_strided(
                tensor,
                (self.POOL_TOKENS, self.TOTAL_HEADS, head_dim),
                (stride_pool_token, head_dim, 1),
                storage_offset=storage_offset,
            )
            for tensor in raw
        ]
        return raw, buffers

    def _alignment_gate(self, buffers, staging):
        head_dim = buffers[0].shape[-1]
        buffers_aligned_16 = kv_buffers_preserve_16_byte_alignment(
            buffers,
            stride_pool_token=self.TOTAL_HEADS * head_dim,
            head_dim=head_dim,
        )
        return _can_use_aligned_staging_copy(
            staging,
            buffers_aligned_16=buffers_aligned_16,
        )

    def test_manager_registration_caches_alignment(self):
        _, buffers = self._buffers()
        k_buffers = buffers[: self.NUM_LAYERS]
        v_buffers = buffers[self.NUM_LAYERS :]
        slot_layer_ids = list(range(2 * self.NUM_LAYERS))

        for manager_cls in (MooncakeKVManager, NixlKVManager):
            with self.subTest(manager=manager_cls.__name__):
                manager = object.__new__(manager_cls)
                manager.set_kv_buffer_tensors(
                    k_buffers=k_buffers,
                    v_buffers=v_buffers,
                    page_size=self.PAGE_SIZE,
                    slot_layer_ids=slot_layer_ids,
                )
                info = manager.kv_buffer_tensors
                self.assertTrue(info[STAGING_COPY_BUFFERS_ALIGNED_16_KEY])
                self.assertEqual(info["page_size"], self.PAGE_SIZE)
                self.assertEqual(info["slot_layer_ids"], slot_layer_ids)

        empty_info = build_staging_kv_buffer_info([], [], self.PAGE_SIZE)
        self.assertFalse(empty_info[STAGING_COPY_BUFFERS_ALIGNED_16_KEY])

    def test_gather_aligned_and_generic_paths_are_bit_exact(self):
        page_indices = np.array([1, 3], dtype=np.int64)
        token_indices = torch.tensor(
            [4, 5, 6, 7, 12, 13, 14, 15], dtype=torch.int64, device="cuda"
        )
        num_heads = 2
        src_head_start = 1
        num_tokens = len(page_indices) * self.PAGE_SIZE
        per_layer_elems = num_tokens * num_heads * self.HEAD_DIM
        total_bytes = per_layer_elems * self.NUM_LAYERS * 2 * 2

        for storage_offset in (0, 1):
            with self.subTest(storage_offset=storage_offset):
                _, buffers = self._buffers(storage_offset=storage_offset)
                k_buffers = buffers[: self.NUM_LAYERS]
                v_buffers = buffers[self.NUM_LAYERS :]
                kv_buffer_info = build_staging_kv_buffer_info(
                    k_buffers=k_buffers,
                    v_buffers=v_buffers,
                    page_size=self.PAGE_SIZE,
                )
                staging = StagingBuffer(
                    size_bytes=total_bytes,
                    device="cuda:0",
                    gpu_id=0,
                )
                # Offset 0 must take the aligned path and offset 1 the generic one;
                # without this the loop could silently measure one path twice.
                self.assertEqual(
                    self._alignment_gate(
                        buffers,
                        staging.buffer[:total_bytes].view(torch.int16),
                    ),
                    storage_offset == 0,
                )

                dispatch = _AlignedDispatchCapture(
                    staging_buffer_module._fused_gather_to_staging_kernel
                )
                with mock.patch.object(
                    staging_buffer_module,
                    "_fused_gather_to_staging_kernel",
                    dispatch,
                ):
                    written = _gather_all_layers_triton(
                        k_buffers,
                        v_buffers,
                        page_indices,
                        staging,
                        src_head_start,
                        num_heads,
                        self.PAGE_SIZE,
                        buffers_aligned_16=kv_buffer_info[
                            STAGING_COPY_BUFFERS_ALIGNED_16_KEY
                        ],
                    )
                self.assertEqual(dispatch.values, [storage_offset == 0])

                expected = torch.cat(
                    [
                        buf[
                            token_indices,
                            src_head_start : src_head_start + num_heads,
                            :,
                        ]
                        .contiguous()
                        .view(torch.int16)
                        .reshape(-1)
                        for buf in buffers
                    ]
                )
                actual = staging.buffer[:written].view(torch.int16)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_scatter_aligned_and_generic_paths_are_bit_exact(self):
        page_indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        token_indices = torch.tensor(
            [4, 5, 6, 7, 12, 13, 14, 15], dtype=torch.int64, device="cuda"
        )
        prefill_attn_tp_size = 4
        decode_attn_tp_size = 1
        num_writers = 4
        num_heads = 2
        num_tokens = page_indices.numel() * self.PAGE_SIZE
        per_layer_elems = num_tokens * num_heads * self.HEAD_DIM
        total_elems = per_layer_elems * self.NUM_LAYERS * 2 * num_writers

        for storage_offset in (0, 1):
            with self.subTest(storage_offset=storage_offset):
                raw, buffers = self._buffers(storage_offset=storage_offset)
                k_buffers = buffers[: self.NUM_LAYERS]
                v_buffers = buffers[self.NUM_LAYERS :]
                kv_buffer_info = build_staging_kv_buffer_info(
                    k_buffers=k_buffers,
                    v_buffers=v_buffers,
                    page_size=self.PAGE_SIZE,
                )
                staging = torch.randint(
                    -32768,
                    32767,
                    (total_elems,),
                    dtype=torch.int16,
                    device="cuda",
                ).view(torch.uint8)
                expected = [tensor.clone() for tensor in raw]
                expected_views = [
                    torch.as_strided(
                        tensor,
                        buffer.shape,
                        buffer.stride(),
                        storage_offset=buffer.storage_offset(),
                    )
                    for tensor, buffer in zip(expected, buffers)
                ]
                staging_typed = staging.view(torch.int16)
                # Offset 0 must take the aligned path and offset 1 the generic one.
                self.assertEqual(
                    self._alignment_gate(buffers, staging_typed),
                    storage_offset == 0,
                )

                for writer_id in range(num_writers):
                    head_start = writer_id * num_heads
                    rank_base = writer_id * per_layer_elems * 2 * self.NUM_LAYERS
                    for layer_kv_id, expected_view in enumerate(expected_views):
                        start = rank_base + layer_kv_id * per_layer_elems
                        values = staging_typed[start : start + per_layer_elems]
                        expected_view[
                            token_indices, head_start : head_start + num_heads, :
                        ] = values.view(torch.bfloat16).view(
                            num_tokens, num_heads, self.HEAD_DIM
                        )

                dispatch = _AlignedDispatchCapture(
                    staging_buffer_module._fused_scatter_from_staging_kernel
                )
                with mock.patch.object(
                    staging_buffer_module,
                    "_fused_scatter_from_staging_kernel",
                    dispatch,
                ):
                    _scatter_staging_to_kv_triton(
                        staging,
                        k_buffers,
                        v_buffers,
                        page_indices,
                        self.PAGE_SIZE,
                        prefill_attn_tp_size,
                        decode_attn_tp_size,
                        0,
                        self.TOTAL_HEADS,
                        buffers_aligned_16=kv_buffer_info[
                            STAGING_COPY_BUFFERS_ALIGNED_16_KEY
                        ],
                    )
                torch.cuda.synchronize()
                self.assertEqual(dispatch.values, [storage_offset == 0])

                for actual, reference in zip(raw, expected):
                    torch.testing.assert_close(
                        actual.view(torch.int16),
                        reference.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )

    def _assert_aligned_gather_dtype(self, dtype, int_dtype):
        page_indices = np.array([1, 3], dtype=np.int64)
        token_indices = torch.tensor(
            [4, 5, 6, 7, 12, 13, 14, 15], dtype=torch.int64, device="cuda"
        )
        num_heads = 2
        src_head_start = 1
        num_tokens = len(page_indices) * self.PAGE_SIZE
        per_layer_elems = num_tokens * num_heads * self.HEAD_DIM
        dtype_size = torch.empty((), dtype=dtype).element_size()
        total_bytes = per_layer_elems * self.NUM_LAYERS * 2 * dtype_size

        _, buffers = self._buffers(dtype=dtype)
        k_buffers = buffers[: self.NUM_LAYERS]
        v_buffers = buffers[self.NUM_LAYERS :]
        info = build_staging_kv_buffer_info(k_buffers, v_buffers, self.PAGE_SIZE)
        staging = StagingBuffer(total_bytes, "cuda:0", 0)
        dispatch = _AlignedDispatchCapture(
            staging_buffer_module._fused_gather_to_staging_kernel
        )
        with mock.patch.object(
            staging_buffer_module,
            "_fused_gather_to_staging_kernel",
            dispatch,
        ):
            written = _gather_all_layers_triton(
                k_buffers,
                v_buffers,
                page_indices,
                staging,
                src_head_start,
                num_heads,
                self.PAGE_SIZE,
                buffers_aligned_16=info[STAGING_COPY_BUFFERS_ALIGNED_16_KEY],
            )

        expected = torch.cat(
            [
                buffer[
                    token_indices,
                    src_head_start : src_head_start + num_heads,
                    :,
                ]
                .contiguous()
                .view(int_dtype)
                .reshape(-1)
                for buffer in buffers
            ]
        )
        torch.testing.assert_close(
            staging.buffer[:written].view(int_dtype), expected, rtol=0, atol=0
        )
        self.assertEqual(dispatch.values, [True])

    def _assert_scatter_case(
        self,
        dtype,
        int_dtype,
        *,
        head_dim=HEAD_DIM,
        staging_byte_offset=0,
        expected_dispatch,
    ):
        page_indices = torch.tensor([1, 3], dtype=torch.int64, device="cuda")
        token_indices = torch.tensor(
            [4, 5, 6, 7, 12, 13, 14, 15], dtype=torch.int64, device="cuda"
        )
        num_writers = 4
        num_heads = 2
        num_tokens = page_indices.numel() * self.PAGE_SIZE
        per_layer_elems = num_tokens * num_heads * head_dim
        total_elems = per_layer_elems * self.NUM_LAYERS * 2 * num_writers
        dtype_size = torch.empty((), dtype=dtype).element_size()

        raw, buffers = self._buffers(dtype=dtype, head_dim=head_dim)
        k_buffers = buffers[: self.NUM_LAYERS]
        v_buffers = buffers[self.NUM_LAYERS :]
        info = build_staging_kv_buffer_info(k_buffers, v_buffers, self.PAGE_SIZE)
        staging_storage = torch.empty(
            staging_byte_offset + total_elems * dtype_size,
            dtype=torch.uint8,
            device="cuda",
        )
        staging = staging_storage[staging_byte_offset:]
        staging_typed = staging.view(int_dtype)[:total_elems]
        source_values = torch.arange(total_elems, dtype=torch.int64, device="cuda")
        staging_typed.copy_(source_values.to(int_dtype))

        expected = [tensor.clone() for tensor in raw]
        expected_views = [
            torch.as_strided(
                tensor,
                buffer.shape,
                buffer.stride(),
                storage_offset=buffer.storage_offset(),
            )
            for tensor, buffer in zip(expected, buffers)
        ]
        for writer_id in range(num_writers):
            head_start = writer_id * num_heads
            rank_base = writer_id * per_layer_elems * 2 * self.NUM_LAYERS
            for layer_kv_id, expected_view in enumerate(expected_views):
                start = rank_base + layer_kv_id * per_layer_elems
                values = staging_typed[start : start + per_layer_elems]
                expected_view[token_indices, head_start : head_start + num_heads, :] = (
                    values.view(dtype).view(num_tokens, num_heads, head_dim)
                )

        dispatch = _AlignedDispatchCapture(
            staging_buffer_module._fused_scatter_from_staging_kernel
        )
        with mock.patch.object(
            staging_buffer_module,
            "_fused_scatter_from_staging_kernel",
            dispatch,
        ):
            _scatter_staging_to_kv_triton(
                staging,
                k_buffers,
                v_buffers,
                page_indices,
                self.PAGE_SIZE,
                4,
                1,
                0,
                self.TOTAL_HEADS,
                buffers_aligned_16=info[STAGING_COPY_BUFFERS_ALIGNED_16_KEY],
            )
        torch.cuda.synchronize()

        self.assertEqual(dispatch.values, [expected_dispatch])
        for actual, reference in zip(raw, expected):
            torch.testing.assert_close(
                actual.view(int_dtype),
                reference.view(int_dtype),
                rtol=0,
                atol=0,
            )

    def test_one_and_four_byte_aligned_specializations_are_bit_exact(self):
        for dtype, int_dtype in (
            (torch.uint8, torch.int8),
            (torch.float32, torch.int32),
        ):
            with self.subTest(dtype=dtype, operation="gather"):
                self._assert_aligned_gather_dtype(dtype, int_dtype)
            with self.subTest(dtype=dtype, operation="scatter"):
                self._assert_scatter_case(
                    dtype,
                    int_dtype,
                    expected_dispatch=True,
                )

    def test_scatter_staging_offset_reject_is_bit_exact(self):
        self._assert_scatter_case(
            torch.bfloat16,
            torch.int16,
            staging_byte_offset=torch.bfloat16.itemsize,
            expected_dispatch=False,
        )

    def test_scatter_geometry_reject_is_bit_exact(self):
        self._assert_scatter_case(
            torch.bfloat16,
            torch.int16,
            head_dim=65,
            expected_dispatch=False,
        )


if __name__ == "__main__":
    unittest.main()
