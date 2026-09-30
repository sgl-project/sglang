"""Correctness tests for disaggregation staging-buffer alignment dispatch."""

import unittest
from types import SimpleNamespace
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
    DecodeStagingHandler,
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


class _ForwardingCaptured(Exception):
    pass


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

    def _alignment_gate(
        self,
        buffers,
        staging,
        *,
        head_dim=HEAD_DIM,
        head_offsets=None,
        elems_per_token=None,
        per_layer_elems=None,
        buffers_aligned_16=None,
    ):
        head_offsets = [0] if head_offsets is None else head_offsets
        elems_per_token = 2 * head_dim if elems_per_token is None else elems_per_token
        per_layer_elems = (
            8 * elems_per_token if per_layer_elems is None else per_layer_elems
        )
        if buffers_aligned_16 is None:
            buffers_aligned_16 = kv_buffers_preserve_16_byte_alignment(
                buffers,
                stride_pool_token=self.TOTAL_HEADS * head_dim,
                head_dim=head_dim,
            )
        return _can_use_aligned_staging_copy(
            staging,
            stride_pool_token=self.TOTAL_HEADS * head_dim,
            head_dim=head_dim,
            head_offsets=head_offsets,
            elems_per_token=elems_per_token,
            per_layer_elems=per_layer_elems,
            buffers_aligned_16=buffers_aligned_16,
        )

    def test_alignment_gate_covers_every_row_base_term(self):
        _, buffers = self._buffers()
        staging = torch.empty(4096, dtype=torch.int16, device="cuda")
        self.assertTrue(self._alignment_gate(buffers, staging))

        _, misaligned_buffers = self._buffers(storage_offset=1)
        self.assertFalse(self._alignment_gate(misaligned_buffers, staging))
        self.assertFalse(self._alignment_gate(buffers, staging[1:]))
        self.assertFalse(
            _can_use_aligned_staging_copy(
                staging,
                stride_pool_token=self.TOTAL_HEADS * self.HEAD_DIM + 1,
                head_dim=self.HEAD_DIM,
                head_offsets=[0],
                elems_per_token=2 * self.HEAD_DIM,
                per_layer_elems=8 * 2 * self.HEAD_DIM,
                buffers_aligned_16=True,
            )
        )
        self.assertFalse(self._alignment_gate(buffers, staging, head_offsets=[1]))
        self.assertFalse(
            self._alignment_gate(
                buffers, staging, elems_per_token=2 * self.HEAD_DIM + 1
            )
        )
        self.assertFalse(self._alignment_gate(buffers, staging, per_layer_elems=1025))

        _, odd_head_buffers = self._buffers(head_dim=65)
        self.assertFalse(self._alignment_gate(odd_head_buffers, staging, head_dim=65))

    def test_cached_alignment_verdict_is_required(self):
        _, buffers = self._buffers()
        staging = torch.empty(4096, dtype=torch.int16, device="cuda")
        with self.assertRaisesRegex(TypeError, "cached registration-time verdict"):
            _can_use_aligned_staging_copy(
                staging,
                stride_pool_token=self.TOTAL_HEADS * self.HEAD_DIM,
                head_dim=self.HEAD_DIM,
                head_offsets=[0],
                elems_per_token=2 * self.HEAD_DIM,
                per_layer_elems=8 * 2 * self.HEAD_DIM,
                buffers_aligned_16=None,
            )

    def test_buffer_metadata_proof_fails_closed(self):
        _, buffers = self._buffers()
        stride_pool_token = self.TOTAL_HEADS * self.HEAD_DIM

        bad_dtype = list(buffers)
        bad_dtype[0] = torch.empty_like(buffers[0], dtype=torch.float32)

        bad_ndim = list(buffers)
        bad_ndim[0] = torch.empty(
            (self.POOL_TOKENS, self.TOTAL_HEADS, self.HEAD_DIM, 1),
            dtype=torch.bfloat16,
            device="cuda",
        )

        _, bad_shape = self._buffers(head_dim=self.HEAD_DIM - 1)

        def strided_buffer(stride):
            storage = torch.empty(
                self.POOL_TOKENS * stride_pool_token * 2,
                dtype=torch.bfloat16,
                device="cuda",
            )
            return torch.as_strided(
                storage,
                (self.POOL_TOKENS, self.TOTAL_HEADS, self.HEAD_DIM),
                stride,
            )

        cases = {
            "dtype": bad_dtype,
            "ndim": bad_ndim,
            "shape": bad_shape,
            "stride0": [
                strided_buffer((stride_pool_token + 1, self.HEAD_DIM, 1)),
                *buffers[1:],
            ],
            "stride1": [
                strided_buffer((stride_pool_token, self.HEAD_DIM + 1, 1)),
                *buffers[1:],
            ],
            "stride2": [
                strided_buffer((stride_pool_token, self.HEAD_DIM, 2)),
                *buffers[1:],
            ],
        }
        for label, candidate in cases.items():
            with self.subTest(label=label):
                self.assertFalse(
                    kv_buffers_preserve_16_byte_alignment(
                        candidate,
                        stride_pool_token=stride_pool_token,
                        head_dim=self.HEAD_DIM,
                    )
                )

    def test_prevalidated_buffer_alignment_avoids_hot_path_rescan(self):
        _, buffers = self._buffers()
        staging = torch.empty(4096, dtype=torch.int16, device="cuda")
        buffers_aligned_16 = kv_buffers_preserve_16_byte_alignment(
            buffers,
            stride_pool_token=self.TOTAL_HEADS * self.HEAD_DIM,
            head_dim=self.HEAD_DIM,
        )
        self.assertTrue(buffers_aligned_16)

        # Buffer metadata is stable after pool registration. The hot path only
        # rechecks request-dependent staging and geometry terms.
        self.assertTrue(
            _can_use_aligned_staging_copy(
                staging,
                stride_pool_token=self.TOTAL_HEADS * self.HEAD_DIM,
                head_dim=self.HEAD_DIM,
                head_offsets=[0],
                elems_per_token=2 * self.HEAD_DIM,
                per_layer_elems=8 * 2 * self.HEAD_DIM,
                buffers_aligned_16=buffers_aligned_16,
            )
        )
        self.assertFalse(
            _can_use_aligned_staging_copy(
                staging,
                stride_pool_token=self.TOTAL_HEADS * self.HEAD_DIM,
                head_dim=self.HEAD_DIM,
                head_offsets=[0],
                elems_per_token=2 * self.HEAD_DIM,
                per_layer_elems=8 * 2 * self.HEAD_DIM,
                buffers_aligned_16=False,
            )
        )

    def test_manager_registration_snapshots_cached_alignment(self):
        _, buffers = self._buffers()
        k_buffers = buffers[: self.NUM_LAYERS]
        v_buffers = buffers[self.NUM_LAYERS :]
        slot_layer_ids = list(range(2 * self.NUM_LAYERS))

        infos = []
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
                infos.append(info)
                self.assertTrue(info[STAGING_COPY_BUFFERS_ALIGNED_16_KEY])
                self.assertEqual(info["page_size"], self.PAGE_SIZE)
                self.assertEqual(info["slot_layer_ids"], slot_layer_ids)
                self.assertIsNot(info["k_buffers"], k_buffers)
                self.assertIsNot(info["v_buffers"], v_buffers)

        k_buffers.clear()
        v_buffers.clear()
        slot_layer_ids.clear()
        for info in infos:
            self.assertEqual(len(info["k_buffers"]), self.NUM_LAYERS)
            self.assertEqual(len(info["v_buffers"]), self.NUM_LAYERS)
            self.assertEqual(len(info["slot_layer_ids"]), 2 * self.NUM_LAYERS)

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
                        head_offsets=[src_head_start * self.HEAD_DIM],
                        elems_per_token=num_heads * self.HEAD_DIM,
                        per_layer_elems=per_layer_elems,
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
                        0,
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
                    self._alignment_gate(
                        buffers,
                        staging_typed,
                        head_offsets=[
                            w * num_heads * self.HEAD_DIM for w in range(num_writers)
                        ],
                        elems_per_token=num_heads * self.HEAD_DIM,
                        per_layer_elems=per_layer_elems,
                    ),
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
                0,
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

    def test_production_sites_forward_cached_alignment(self):
        _, buffers = self._buffers()
        k_buffers = buffers[: self.NUM_LAYERS]
        v_buffers = buffers[self.NUM_LAYERS :]
        info = build_staging_kv_buffer_info(k_buffers, v_buffers, self.PAGE_SIZE)
        self.assertTrue(info[STAGING_COPY_BUFFERS_ALIGNED_16_KEY])

        def capture_gather(*args, **kwargs):
            self.assertTrue(kwargs["buffers_aligned_16"])
            raise _ForwardingCaptured

        staging = SimpleNamespace(fits=lambda _size: True)
        for manager_cls in (MooncakeKVManager, NixlKVManager):
            with self.subTest(manager=manager_cls.__name__):
                manager = object.__new__(manager_cls)
                manager.kv_buffer_tensors = info
                manager.kv_args = SimpleNamespace(engine_rank=0, gpu_id=0)
                manager.attn_tp_size = 1
                if manager_cls is MooncakeKVManager:
                    manager.pp_size = 1
                with (
                    mock.patch.object(
                        staging_buffer_module,
                        "resolve_total_kv_heads",
                        return_value=self.TOTAL_HEADS,
                    ),
                    mock.patch.object(
                        staging_buffer_module,
                        "compute_head_slice_params",
                        return_value=(0, 2, 0, 2),
                    ),
                    mock.patch.object(
                        staging_buffer_module,
                        "compute_staging_layout",
                        return_value=(1, [4096], 4096),
                    ),
                    mock.patch.object(
                        staging_buffer_module,
                        "gather_all_layers_to_staging",
                        side_effect=capture_gather,
                    ),
                    self.assertRaises(_ForwardingCaptured),
                ):
                    if manager_cls is MooncakeKVManager:
                        manager.send_kvcache_staged(
                            "session",
                            np.array([1], dtype=np.int32),
                            0,
                            1 << 20,
                            0,
                            1,
                            self.HEAD_DIM * 2,
                            list(range(2 * self.NUM_LAYERS)),
                            staging,
                        )
                    else:
                        manager.send_kvcache_staged(
                            "peer",
                            np.array([1], dtype=np.int32),
                            0,
                            1 << 20,
                            0,
                            0,
                            1,
                            self.HEAD_DIM * 2,
                            "notif",
                            staging,
                        )

        def capture_scatter(*args, **kwargs):
            self.assertTrue(kwargs["buffers_aligned_16"])
            raise _ForwardingCaptured

        handler = object.__new__(DecodeStagingHandler)
        handler.kv_buffer_info = info
        handler.kv_manager = SimpleNamespace(kv_args=SimpleNamespace(engine_rank=0))
        handler.decode_tp = 1
        handler.total_kv_heads = self.TOTAL_HEADS
        handler.staging_allocator = SimpleNamespace(
            _scatter_stream=torch.cuda.Stream(device="cuda"),
            buffer=SimpleNamespace(
                buffer=torch.empty(8192, dtype=torch.uint8, device="cuda")
            ),
        )
        handler.scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(
                    self.POOL_TOKENS, dtype=torch.int64, device="cuda"
                ).reshape(1, -1)
            )
        )
        decode_req = SimpleNamespace(
            req=SimpleNamespace(
                kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=0)
            )
        )
        receiver = SimpleNamespace(prefill_info=SimpleNamespace(attn_tp_size=1))
        with (
            mock.patch.object(
                staging_buffer_module,
                "scatter_staging_to_kv",
                side_effect=capture_scatter,
            ),
            self.assertRaises(_ForwardingCaptured),
        ):
            handler._scatter_region(0, 0, 1, decode_req, receiver)


if __name__ == "__main__":
    unittest.main()
