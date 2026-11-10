"""Mooncake token-interleaved DSA DCP transfer on Ascend."""

from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
import torch

from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.utils import resolve_dcp_dst_entry_indices
from sglang.srt.utils import device_stream_context


class _PackBuffer:
    def __init__(self, size: int, device_id: int):
        device = torch.device("npu", device_id)
        torch.get_device_module(device).set_device(device_id)
        self.buffer = torch.empty(size, dtype=torch.uint8, device=device)
        self.stream = None
        self.failed = False


def _pack_rows(pack_buffer, buffers, src_indices, token_widths):
    """Gather row-contiguous NPU MLA bytes into a registered per-worker buffer."""
    count = src_indices.size
    size = count * sum(token_widths)
    if pack_buffer.failed or size > pack_buffer.buffer.nbytes:
        return None

    device = pack_buffer.buffer.device
    if pack_buffer.stream is None:
        pack_buffer.stream = torch.get_device_module(device).Stream(device=device)
    stream = pack_buffer.stream
    base = pack_buffer.buffer.data_ptr()
    packed_ptrs = []
    offset = 0
    # The transfer worker already waited for the Prefill forward to finish.
    with device_stream_context(stream):
        indices = torch.as_tensor(src_indices, device=device, dtype=torch.int64)
        for buffer, width in zip(buffers, token_widths):
            src = buffer.view(torch.uint8).view(-1, width)
            dst = pack_buffer.buffer.narrow(0, offset, count * width).view(count, width)
            torch.index_select(src, 0, indices, out=dst)
            packed_ptrs.append(base + offset)
            offset += count * width
    # Mooncake reads raw addresses and has no dependency on the NPU stream.
    stream.synchronize()
    return packed_ptrs, np.arange(count, dtype=np.int64)


def _prepare_dsa_token_dcp(
    kv_args,
    peer,
    *,
    target_buffers,
    supports_relayout,
    pack_buffers,
    num_workers,
    register,
):
    if not supports_relayout:
        raise ValueError(
            "NPU DSA token PD relayout requires a static row-contiguous KV pool."
        )
    if (
        not target_buffers
        or not peer.requires_dcp_relayout
        or peer.dst_draft_rows_dcp_collapsed
    ):
        raise ValueError(
            "NPU DSA token DCP requires matching PD layouts with replicated draft rows"
        )
    if len(peer.dst_kv_ptrs) != len(peer.dst_kv_item_lens):
        raise ValueError("NPU DSA token DCP requires per-entry destination widths")
    entries = resolve_dcp_dst_entry_indices(
        kv_args.kv_layer_ids,
        peer.dst_kv_layer_ids,
        len(kv_args.kv_data_ptrs),
        len(peer.dst_kv_ptrs),
    )
    widths = []
    for ptr, src_bytes, dst_entry in zip(
        kv_args.kv_data_ptrs, kv_args.kv_item_lens, entries
    ):
        scale = 1 if ptr in target_buffers else peer.dst_dcp_size
        width, remainder = divmod(src_bytes, kv_args.page_size)
        if remainder or width <= 0 or dst_entry >= len(peer.dst_kv_item_lens):
            raise ValueError("Invalid NPU DSA token DCP page geometry")
        if peer.dst_kv_item_lens[dst_entry] != src_bytes * scale:
            raise ValueError(
                f"NPU DSA token DCP entry {dst_entry} has incompatible page bytes: "
                f"prefill={src_bytes}, decode={peer.dst_kv_item_lens[dst_entry]}, "
                f"scale={scale}"
            )
        widths.append(width)
    if pack_buffers is None:
        # A worker reuses one bounded buffer after each synchronous transfer.
        # Long chunks are split into multiple packs in the send path.
        size = 32 * 1024 * 1024
        device_id = kv_args.gpu_id
        pack_buffers = [_PackBuffer(size, device_id) for _ in range(num_workers)]
        register(
            [pack.buffer.data_ptr() for pack in pack_buffers],
            [size] * len(pack_buffers),
        )
    return widths, pack_buffers


def _send_dsa_token_dcp(
    *, transfer_data, plan, src_ptrs, dst_ptrs, widths, pack_buffer, target_buffers
) -> int:
    entries = list(zip(src_ptrs, dst_ptrs, widths, strict=True))
    target = [entry for entry in entries if entry[0] in target_buffers]
    replicas = [entry for entry in entries if entry[0] not in target_buffers]

    def transfer(entries, src_rows, dst_rows):
        src_groups, dst_groups = group_concurrent_contiguous(src_rows, dst_rows)
        return transfer_data(
            [
                (src + int(a[0]) * width, dst + int(b[0]) * width, len(a) * width)
                for src, dst, width in entries
                for a, b in zip(src_groups, dst_groups)
            ],
        )

    if target and plan.target_src_token_indices.size:
        target_widths = [width for _, _, width in target]
        tensors = [target_buffers[src] for src, _, _ in target]
        max_rows = 128
        if pack_buffer is not None and not pack_buffer.failed:
            max_rows = max(1, pack_buffer.buffer.nbytes // sum(target_widths))
        for start in range(0, plan.target_src_token_indices.size, max_rows):
            src_rows = plan.target_src_token_indices[start : start + max_rows]
            dst_rows = plan.target_dst_token_indices[start : start + max_rows]
            packed = (
                _pack_rows(pack_buffer, tensors, src_rows, target_widths)
                if pack_buffer is not None
                else None
            )
            if packed is None:
                ret = transfer(target, src_rows, dst_rows)
            else:
                packed_ptrs, packed_rows = packed
                ret = transfer(
                    [
                        (ptr, dst, width)
                        for ptr, (_, dst, width) in zip(packed_ptrs, target)
                    ],
                    packed_rows,
                    dst_rows,
                )
                if ret != 0:
                    # A synchronous TE timeout can leave reads in flight.
                    pack_buffer.failed = True
            if ret != 0:
                return ret
    if replicas and plan.draft_src_token_indices.size:
        return transfer(
            replicas, plan.draft_src_token_indices, plan.draft_dst_token_indices
        )
    return 0


@dataclass(frozen=True)
class _DSATokenDCPTransfer:
    prepare: Callable[..., tuple[list[int], list[_PackBuffer]]]
    send: Callable[..., int]


def build_dsa_token_dcp_transfer(
    target_buffers: dict[int, torch.Tensor],
    *,
    supports_relayout: bool,
) -> _DSATokenDCPTransfer:
    return _DSATokenDCPTransfer(
        prepare=partial(
            _prepare_dsa_token_dcp,
            target_buffers=target_buffers,
            supports_relayout=supports_relayout,
        ),
        send=partial(_send_dsa_token_dcp, target_buffers=target_buffers),
    )
