from __future__ import annotations

from collections.abc import Sequence

import torch


def copy_canary_buffers_indexed(
    *,
    src_buffers: Sequence[torch.Tensor],
    dst_buffers: Sequence[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
    io_backend: str,
) -> None:
    """Copy K/V head+tail canary rows with the KV transfer's index mapping."""
    if len(src_buffers) != 4 or len(dst_buffers) != 4:
        raise ValueError("kv-canary: HiCache transfer requires four K/V canary buffers")
    if src_indices.numel() != dst_indices.numel():
        raise ValueError(
            "kv-canary: HiCache source/destination index counts differ: "
            f"{src_indices.numel()} != {dst_indices.numel()}"
        )
    if src_indices.numel() == 0:
        return

    slot_bytes = int(src_buffers[0].shape[1])
    for name, buffers in (("src", src_buffers), ("dst", dst_buffers)):
        for tensor in buffers:
            if tensor.dtype is not torch.uint8 or tensor.ndim != 2:
                raise ValueError(
                    f"kv-canary: HiCache {name} buffers must be 2-D uint8 tensors"
                )
            if int(tensor.shape[1]) != slot_bytes or not tensor.is_contiguous():
                raise ValueError(
                    f"kv-canary: HiCache {name} buffers must be contiguous with "
                    f"slot width {slot_bytes}"
                )

    if all(t.device.type == "cpu" for t in (*src_buffers, *dst_buffers)):
        _copy_cpu(
            src_buffers=src_buffers,
            dst_buffers=dst_buffers,
            src_indices=src_indices,
            dst_indices=dst_indices,
        )
        return

    if io_backend == "direct":
        from sgl_kernel.kvcacheio import transfer_kv_direct

        transfer_kv_direct(
            src_layers=list(src_buffers),
            dst_layers=list(dst_buffers),
            src_indices=src_indices,
            dst_indices=dst_indices,
            page_size=1,
        )
        return

    if io_backend != "kernel":
        raise NotImplementedError(
            f"kv-canary: HiCache canary transfer does not support io_backend={io_backend!r}"
        )

    from sgl_kernel.kvcacheio import transfer_kv_all_layer

    accelerator = next(
        (t.device for t in (*src_buffers, *dst_buffers) if t.device.type != "cpu"),
        None,
    )
    assert accelerator is not None

    src_ptrs = torch.tensor(
        [t.data_ptr() for t in src_buffers], dtype=torch.uint64, device=accelerator
    )
    dst_ptrs = torch.tensor(
        [t.data_ptr() for t in dst_buffers], dtype=torch.uint64, device=accelerator
    )
    src_indices_device = src_indices.to(
        device=accelerator, dtype=torch.int64, non_blocking=True
    )
    dst_indices_device = dst_indices.to(
        device=accelerator, dtype=torch.int64, non_blocking=True
    )
    transfer_kv_all_layer(
        src_k_layers=src_ptrs[:2],
        dst_k_layers=dst_ptrs[:2],
        src_v_layers=src_ptrs[2:],
        dst_v_layers=dst_ptrs[2:],
        src_indices=src_indices_device,
        dst_indices=dst_indices_device,
        item_size=slot_bytes,
        num_layers=2,
    )
    src_ptrs.record_stream(torch.cuda.current_stream(accelerator))
    dst_ptrs.record_stream(torch.cuda.current_stream(accelerator))
    src_indices_device.record_stream(torch.cuda.current_stream(accelerator))
    dst_indices_device.record_stream(torch.cuda.current_stream(accelerator))


def _copy_cpu(
    *,
    src_buffers: Sequence[torch.Tensor],
    dst_buffers: Sequence[torch.Tensor],
    src_indices: torch.Tensor,
    dst_indices: torch.Tensor,
) -> None:
    src_indices_cpu = src_indices.to(device="cpu", dtype=torch.int64)
    dst_indices_cpu = dst_indices.to(device="cpu", dtype=torch.int64)
    for src, dst in zip(src_buffers, dst_buffers):
        dst.index_copy_(0, dst_indices_cpu, src.index_select(0, src_indices_cpu))
