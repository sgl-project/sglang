"""Correctness tests for disaggregation staging-buffer alignment dispatch."""

import unittest

import numpy as np
import torch

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


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestDisaggregationStagingBuffer(CustomTestCase):
    NUM_LAYERS = 2
    PAGE_SIZE = 4
    POOL_TOKENS = 32
    TOTAL_HEADS = 8
    HEAD_DIM = 128

    def _buffers(self, *, storage_offset: int = 0, head_dim: int = HEAD_DIM):
        stride_pool_token = self.TOTAL_HEADS * head_dim
        raw = [
            torch.empty(
                storage_offset + self.POOL_TOKENS * stride_pool_token,
                dtype=torch.bfloat16,
                device="cuda",
            )
            for _ in range(2 * self.NUM_LAYERS)
        ]
        for tensor in raw:
            tensor.random_(-100, 100)
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
    ):
        head_offsets = [0] if head_offsets is None else head_offsets
        elems_per_token = 2 * head_dim if elems_per_token is None else elems_per_token
        per_layer_elems = (
            8 * elems_per_token if per_layer_elems is None else per_layer_elems
        )
        return _can_use_aligned_staging_copy(
            buffers,
            staging,
            stride_pool_token=self.TOTAL_HEADS * head_dim,
            head_dim=head_dim,
            head_offsets=head_offsets,
            elems_per_token=elems_per_token,
            per_layer_elems=per_layer_elems,
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
                buffers,
                staging,
                stride_pool_token=self.TOTAL_HEADS * self.HEAD_DIM + 1,
                head_dim=self.HEAD_DIM,
                head_offsets=[0],
                elems_per_token=2 * self.HEAD_DIM,
                per_layer_elems=8 * 2 * self.HEAD_DIM,
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
                [],
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
                [],
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

                for actual, reference in zip(raw, expected):
                    torch.testing.assert_close(
                        actual.view(torch.int16),
                        reference.view(torch.int16),
                        rtol=0,
                        atol=0,
                    )


if __name__ == "__main__":
    unittest.main()
