# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The DeepSpec Authors
# See LICENSE in this directory for the original license.

"""Stable replay-side buffers for UBTree target-verification metadata."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.utils import create_flashinfer_kv_indices_triton


@triton.jit
def _build_full_tree_mask_kernel(
    visibility,
    prefix_lens,
    prefix_lens_cumsum,
    custom_mask,
    visibility_stride_batch: tl.constexpr,
    visibility_stride_query: tl.constexpr,
    visibility_stride_key: tl.constexpr,
    raw_bs: tl.constexpr,
    batch_size: tl.constexpr,
    num_nodes: tl.constexpr,
    num_values,
    block_size: tl.constexpr,
):
    """Materialize prefix and tree visibility in one pass.

    Each request owns ``num_nodes * (prefix_len + num_nodes)`` bits. Prefix
    columns and CUDA-graph padding requests are all-visible; real tree suffixes
    are read from ``visibility``. This replaces the eager full-buffer fill plus
    sparse tree-suffix store with one launch.
    """

    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    valid = offsets < num_values

    request_indices = tl.zeros((block_size,), dtype=tl.int32)
    for request_index in range(batch_size - 1):
        request_end = num_nodes * (
            tl.load(prefix_lens_cumsum + request_index)
            + (request_index + 1) * num_nodes
        )
        request_indices += (offsets >= request_end).to(tl.int32)

    prefix_sum_before = tl.load(
        prefix_lens_cumsum + request_indices - 1,
        mask=valid & (request_indices > 0),
        other=0,
    )
    request_starts = num_nodes * (prefix_sum_before + request_indices * num_nodes)
    request_offsets = offsets - request_starts
    prefix_len = tl.load(prefix_lens + request_indices, mask=valid, other=1)
    row_width = prefix_len + num_nodes
    query = request_offsets // row_width
    column = request_offsets - query * row_width
    tree_key = column - prefix_len

    real_tree_value = valid & (request_indices < raw_bs) & (tree_key >= 0)
    visibility_offset = (
        request_indices * visibility_stride_batch
        + query * visibility_stride_query
        + tree_key * visibility_stride_key
    )
    value = tl.load(visibility + visibility_offset, mask=real_tree_value, other=1)
    tl.store(custom_mask + offsets, value, mask=valid)


@triton.jit
def _store_packed_mask_indptr_kernel(
    prefix_lens,
    packed_mask_indptr,
    num_nodes: tl.constexpr,
    batch_size: tl.constexpr,
):
    index = tl.program_id(0)
    packed_offset = 0
    for request_index in range(batch_size):
        prefix_len = tl.load(prefix_lens + request_index)
        num_bits = num_nodes * (prefix_len + num_nodes)
        num_bytes = ((num_bits + 7) // 8).to(tl.int32)
        packed_offset += tl.where(request_index < index, num_bytes, 0)
    tl.store(packed_mask_indptr + index, packed_offset)


@triton.jit
def _pack_custom_mask_kernel(
    custom_mask,
    prefix_lens,
    prefix_lens_cumsum,
    packed_custom_mask,
    packed_mask_indptr,
    num_nodes: tl.constexpr,
    batch_size: tl.constexpr,
    num_output_bytes: tl.constexpr,
    block_size: tl.constexpr,
):
    output_offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    total_output_bytes = tl.load(packed_mask_indptr + batch_size)
    valid_output = (output_offsets < num_output_bytes) & (
        output_offsets < total_output_bytes
    )

    request_indices = tl.zeros((block_size,), dtype=tl.int32)
    for request_index in range(batch_size):
        next_request_offset = tl.load(packed_mask_indptr + request_index + 1)
        request_indices += (output_offsets >= next_request_offset).to(tl.int32)

    request_byte_offsets = output_offsets - tl.load(
        packed_mask_indptr + request_indices, mask=valid_output, other=0
    )
    prefix_sum_before = tl.load(
        prefix_lens_cumsum + request_indices - 1,
        mask=valid_output & (request_indices > 0),
        other=0,
    )
    request_bit_starts = num_nodes * (prefix_sum_before + request_indices * num_nodes)
    request_num_bits = num_nodes * (
        tl.load(prefix_lens + request_indices, mask=valid_output, other=0) + num_nodes
    )
    byte_bit_starts = request_byte_offsets * 8

    packed_values = tl.zeros((block_size,), dtype=tl.int32)
    for bit_index in range(8):
        request_bit_offsets = byte_bit_starts + bit_index
        bit_valid = valid_output & (request_bit_offsets < request_num_bits)
        bit_values = tl.load(
            custom_mask + request_bit_starts + request_bit_offsets,
            mask=bit_valid,
            other=0,
        ).to(tl.int32)
        packed_values |= bit_values << bit_index
    tl.store(
        packed_custom_mask + output_offsets,
        packed_values.to(tl.uint8),
        mask=valid_output,
    )


class UBTreeVerifyMetadataBuffers:
    """Own fixed-address mask and FlashInfer verify metadata buffers.

    The mask is packed for the CUDA-graph batch bucket, including synthetic
    padding requests of prefix length one.  This makes the prefix consumed by
    any replay bucket deterministic and prevents stale tree-mask values from a
    previous, larger batch.
    """

    def __init__(
        self,
        *,
        batch_sizes: Iterable[int],
        num_nodes: int,
        max_context_len: int,
        device: torch.device | str,
    ) -> None:
        requested = sorted({int(bs) for bs in batch_sizes if int(bs) > 0})
        if not requested:
            raise ValueError("batch_sizes must contain at least one positive value")
        self.batch_sizes = tuple(requested)
        self.max_bs = requested[-1]
        self.num_nodes = int(num_nodes)
        self.max_context_len = int(max_context_len)
        self.device = torch.device(device)
        if self.num_nodes <= 0 or self.max_context_len <= 0:
            raise ValueError("num_nodes and max_context_len must be positive")

        max_kv_tokens = self.max_bs * (self.max_context_len + self.num_nodes)
        max_mask_numel = self.num_nodes * max_kv_tokens
        self.custom_mask = torch.ones(
            max_mask_numel, dtype=torch.bool, device=self.device
        )
        self.prefix_lens = torch.ones(
            self.max_bs, dtype=torch.int64, device=self.device
        )
        self.prefix_lens_cumsum = torch.empty_like(self.prefix_lens)
        self.paged_kernel_lens = torch.empty(
            self.max_bs, dtype=torch.int64, device=self.device
        )
        self.kv_indices = torch.empty(
            max_kv_tokens, dtype=torch.int32, device=self.device
        )
        self.kv_indptr = torch.zeros(
            self.max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.qo_indptr = torch.arange(
            0,
            (self.max_bs + 1) * self.num_nodes,
            self.num_nodes,
            dtype=torch.int32,
            device=self.device,
        )
        self._prepared_bs = 0
        self._prepared_mask_numel = 0
        self._prepared_packed_mask_numel = 0

    def graph_batch_size(self, raw_bs: int) -> int:
        raw_bs = int(raw_bs)
        for batch_size in self.batch_sizes:
            if raw_bs <= batch_size:
                return batch_size
        return raw_bs

    def can_prepare(self, raw_bs: int, prefix_lens_sum: int) -> bool:
        padded_bs = self.graph_batch_size(raw_bs)
        return (
            0 <= int(raw_bs) <= self.max_bs
            and padded_bs <= self.max_bs
            and int(prefix_lens_sum) + int(raw_bs) * self.num_nodes
            <= int(raw_bs) * (self.max_context_len + self.num_nodes)
        )

    def stage_prefix_lens(self, prefix_lens: torch.Tensor, raw_bs: int) -> int:
        """Stage real and graph-padding prefix lengths at stable addresses."""

        raw_bs = int(raw_bs)
        if prefix_lens.shape != (raw_bs,):
            raise ValueError("prefix_lens must contain one length per tree")
        padded_bs = self.graph_batch_size(raw_bs)
        if padded_bs > self.max_bs:
            raise ValueError("verify batch exceeds the fixed metadata capacity")
        self.prefix_lens.fill_(1)
        self.prefix_lens[:raw_bs].copy_(prefix_lens)
        torch.cumsum(
            self.prefix_lens[:padded_bs],
            dim=0,
            out=self.prefix_lens_cumsum[:padded_bs],
        )
        return padded_bs

    def prepare_custom_mask(
        self,
        visibility: torch.Tensor,
        prefix_lens: torch.Tensor,
        *,
        prefix_lens_sum: int,
        prefix_prepared: bool = False,
    ) -> torch.Tensor:
        raw_bs, num_queries, num_keys = visibility.shape
        if num_queries != self.num_nodes or num_keys != self.num_nodes:
            raise ValueError("visibility shape must be [batch, num_nodes, num_nodes]")
        if prefix_lens.shape != (raw_bs,):
            raise ValueError("prefix_lens must contain one length per tree")
        padded_bs = self.graph_batch_size(raw_bs)
        if not self.can_prepare(raw_bs, prefix_lens_sum):
            raise ValueError("verify metadata exceeds the fixed buffer capacity")

        # Graph padding uses seq_len_fill_value=1 for FlashInfer. When proposal
        # construction is captured, these stable buffers were already written
        # in the graph tail; eager/fallback paths stage them here.
        if not prefix_prepared:
            self.stage_prefix_lens(prefix_lens, raw_bs)
        mask_numel = self.num_nodes * (
            int(prefix_lens_sum)
            + raw_bs * self.num_nodes
            + (padded_bs - raw_bs) * (self.num_nodes + 1)
        )
        output = self.custom_mask[:mask_numel]

        if visibility.device.type == "cuda":
            block_size = 256
            _build_full_tree_mask_kernel[(triton.cdiv(mask_numel, block_size),)](
                visibility,
                self.prefix_lens,
                self.prefix_lens_cumsum,
                self.custom_mask,
                visibility_stride_batch=visibility.stride(0),
                visibility_stride_query=visibility.stride(1),
                visibility_stride_key=visibility.stride(2),
                raw_bs=raw_bs,
                batch_size=padded_bs,
                num_nodes=self.num_nodes,
                num_values=mask_numel,
                block_size=block_size,
            )
        else:
            # CPU is only a correctness fallback; preserve the reference layout.
            output.fill_(True)
            for batch_index in range(raw_bs):
                prefix_len = int(self.prefix_lens[batch_index])
                request_start = self.num_nodes * (
                    (
                        0
                        if batch_index == 0
                        else int(self.prefix_lens_cumsum[batch_index - 1])
                    )
                    + batch_index * self.num_nodes
                )
                rows = output[
                    request_start : request_start
                    + self.num_nodes * (prefix_len + self.num_nodes)
                ].view(self.num_nodes, prefix_len + self.num_nodes)
                rows[:, prefix_len:].copy_(visibility[batch_index])

        self._prepared_bs = padded_bs
        self._prepared_mask_numel = mask_numel
        return output

    def prepare_packed_custom_mask(
        self,
        packed_custom_mask: torch.Tensor,
        packed_mask_indptr: torch.Tensor,
        *,
        batch_size: int,
    ) -> bool:
        """Pack the prepared bool mask directly into FlashInfer graph buffers.

        FlashInfer consumes a little-endian packed segment per request and its
        mask indptr is measured in bytes.  Writing the wrapper-owned buffers
        here avoids ``segment_packbits`` allocations, its D2H size sync, and
        the subsequent copies performed by ``plan()``.
        """
        batch_size = int(batch_size)
        if batch_size != self._prepared_bs or batch_size <= 0:
            return False
        if packed_mask_indptr.numel() < batch_size + 1:
            raise ValueError("FlashInfer packed mask indptr buffer is too small")

        # Each request is padded to a byte boundary.  This is a tight host-known
        # upper bound; the exact final length is written device-side to indptr.
        packed_mask_numel = (self._prepared_mask_numel + 7 * batch_size + 7) // 8
        if packed_custom_mask.numel() < packed_mask_numel:
            raise ValueError("FlashInfer packed custom mask buffer is too small")

        if self.device.type == "cuda":
            _store_packed_mask_indptr_kernel[(batch_size + 1,)](
                self.prefix_lens,
                packed_mask_indptr,
                num_nodes=self.num_nodes,
                batch_size=batch_size,
            )
            block_size = 256
            _pack_custom_mask_kernel[(triton.cdiv(packed_mask_numel, block_size),)](
                self.custom_mask,
                self.prefix_lens,
                self.prefix_lens_cumsum,
                packed_custom_mask,
                packed_mask_indptr,
                num_nodes=self.num_nodes,
                batch_size=batch_size,
                num_output_bytes=packed_mask_numel,
                block_size=block_size,
            )
        else:
            packed_offset = 0
            packed_mask_indptr[0] = 0
            bit_offset = 0
            for request_index in range(batch_size):
                prefix_len = int(self.prefix_lens[request_index])
                num_bits = self.num_nodes * (prefix_len + self.num_nodes)
                segment = self.custom_mask[bit_offset : bit_offset + num_bits]
                num_bytes = (num_bits + 7) // 8
                destination = packed_custom_mask[
                    packed_offset : packed_offset + num_bytes
                ]
                destination.zero_()
                for bit_index in range(num_bits):
                    if bool(segment[bit_index]):
                        destination[bit_index // 8] |= 1 << (bit_index % 8)
                bit_offset += num_bits
                packed_offset += num_bytes
                packed_mask_indptr[request_index + 1] = packed_offset

        self._prepared_packed_mask_numel = packed_mask_numel
        return True

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)
        if batch_size > self.max_bs:
            raise ValueError("verify batch exceeds the fixed metadata capacity")

        lengths = self.paged_kernel_lens[:batch_size]
        torch.add(paged_kernel_lens, self.num_nodes, out=lengths)
        self.kv_indptr[0].zero_()
        torch.cumsum(lengths, dim=0, out=self.kv_indptr[1 : batch_size + 1])
        num_kv_indices = int(paged_kernel_lens_sum) + self.num_nodes * batch_size
        kv_indices = self.kv_indices[:num_kv_indices]
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            lengths,
            self.kv_indptr,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        mask_numel = self.num_nodes * num_kv_indices
        if mask_numel > self._prepared_mask_numel:
            raise ValueError(
                "custom mask was not prepared for the CUDA-graph padded batch"
            )
        return (
            kv_indices,
            self.kv_indptr[: batch_size + 1],
            self.qo_indptr[: batch_size + 1],
            self.custom_mask[:mask_numel],
        )


__all__ = ["UBTreeVerifyMetadataBuffers"]
