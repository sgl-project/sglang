"""Bounded, reusable registered Host slots with explicit transfer ownership."""

from __future__ import annotations

import math
import threading
from typing import Protocol

import msgspec
import torch
from sglang.srt.training_capture.protocol import (
    DTYPES,
    ELEMENT_BYTES,
    CaptureError,
    ContractError,
    KVSpec,
    aux_specs,
)
from sglang.srt.training_capture.topology import CapturePartition


class BufferRegistrar(Protocol):
    def register(self, tensor: torch.Tensor) -> None: ...

    def unregister(self, tensor: torch.Tensor) -> None: ...


class HostSlot(msgspec.Struct, eq=False):
    storage: torch.Tensor
    tensors: dict[str, torch.Tensor]
    manifest_buffer: torch.Tensor
    state: str = "free"
    device_storage: torch.Tensor | None = None
    device_tensors: dict[str, torch.Tensor] | None = None
    teacher_device_tensors: dict[str, torch.Tensor] | None = None


class HostBufferPool:
    """Allocate/register once, then reuse only after both CUDA and Store completion.

    Quarantined slots keep strong references and their registrations until an
    operator closes the transport. A failed transfer never makes them reusable.
    """

    def __init__(
        self,
        *,
        kv: KVSpec,
        max_tokens: int,
        slots: int,
        max_bytes: int,
        registrar: BufferRegistrar,
        manifest_bytes: int = 1 << 20,
        pin_memory: bool = True,
        device: torch.device | None = None,
        kv_d2h_batch_tokens: int = 1,
        teacher_d2h_batch_tokens: int = 1,
        max_device_bytes: int = 0,
        partition: CapturePartition | None = None,
    ):
        if min(max_tokens, slots, max_bytes, manifest_bytes) <= 0:
            raise ValueError("Host pool sizes must be positive")
        if (
            min(kv_d2h_batch_tokens, teacher_d2h_batch_tokens) < 1
            or max_device_bytes < 0
        ):
            raise ValueError("invalid device staging limits")
        layers = kv.layers if partition is None else partition.local_layers(kv)
        include_aux = partition is None or partition.include_aux
        if partition is not None and not partition.active:
            raise ContractError("rank does not own a capture payload")
        specs = aux_specs(max_tokens, max_tokens) if include_aux else {}
        kv_names = []
        for layer in layers:
            for component, dim in (
                ("k", layer.key_head_dim),
                ("v", layer.value_head_dim),
            ):
                name = f"target_{component}.{layer.layer_id}"
                kv_names.append(name)
                specs[name] = (
                    kv.dtype,
                    [max_tokens, layer.num_kv_heads, dim],
                )
        layout = {}
        offset = 0
        for name, (dtype, shape) in specs.items():
            offset = (offset + 63) // 64 * 64
            length = math.prod(shape) * ELEMENT_BYTES[dtype]
            layout[name] = (offset, length, dtype, shape)
            offset += length
        manifest_offset = (offset + 63) // 64 * 64
        slot_bytes = manifest_offset + (manifest_bytes if include_aux else 0)
        if slot_bytes * slots > max_bytes:
            raise ValueError(
                f"Host pool requires {slot_bytes * slots} bytes, budget is {max_bytes}"
            )
        device_layout = {}
        device_bytes = 0
        staging_groups = (
            (kv_names, kv_d2h_batch_tokens, "KV"),
            (
                (
                    ["teacher_topk_ids", "teacher_topk_logits", "teacher_logsumexp"]
                    if include_aux
                    else []
                ),
                teacher_d2h_batch_tokens,
                "teacher",
            ),
        )
        for names, batch_tokens, label in staging_groups:
            if batch_tokens <= 1 or not names:
                continue
            if device is None or not max_device_bytes:
                raise ValueError(f"{label} staging requires a device and a byte budget")
            capacity = min(batch_tokens, max_tokens)
            for name in names:
                dtype, shape = specs[name]
                shape = [capacity, *shape[1:]]
                device_bytes = (device_bytes + 63) // 64 * 64
                length = math.prod(shape) * ELEMENT_BYTES[dtype]
                device_layout[name] = (device_bytes, length, dtype, shape)
                device_bytes += length
            if device_bytes * slots > max_device_bytes:
                raise ValueError(
                    f"{label} staging requires {device_bytes * slots} bytes, "
                    f"budget is {max_device_bytes}"
                )
        self.registrar = registrar
        self.lock = threading.Lock()
        self.slots: list[HostSlot] = []
        self.allocated_bytes = 0
        self.device_allocated_bytes = 0
        self.device_limit_bytes = max_device_bytes
        self.closed = False
        try:
            for _ in range(slots):
                storage = torch.empty(
                    slot_bytes, dtype=torch.uint8, pin_memory=pin_memory
                )
                views = {
                    name: storage[start : start + length]
                    .view(DTYPES[dtype])
                    .reshape(shape)
                    for name, (start, length, dtype, shape) in layout.items()
                }
                slot = HostSlot(storage, views, storage[manifest_offset:])
                if device_layout:
                    slot.device_storage = torch.empty(
                        device_bytes, dtype=torch.uint8, device=device
                    )
                    device_views = {
                        name: slot.device_storage[start : start + length]
                        .view(DTYPES[dtype])
                        .reshape(shape)
                        for name, (start, length, dtype, shape) in device_layout.items()
                    }
                    slot.device_tensors = {
                        name: value
                        for name, value in device_views.items()
                        if name in kv_names
                    } or None
                    slot.teacher_device_tensors = {
                        name: value
                        for name, value in device_views.items()
                        if name not in kv_names
                    } or None
                self.registrar.register(storage)
                self.slots.append(slot)
                self.allocated_bytes += slot_bytes
                self.device_allocated_bytes += device_bytes
        except Exception:
            self.close()
            raise

    def acquire(self) -> HostSlot | None:
        with self.lock:
            if self.closed:
                return None
            for slot in self.slots:
                if slot.state == "free":
                    slot.state = "filling"
                    return slot
        return None

    def release(self, slot: HostSlot, *, transfer_complete: bool) -> None:
        with self.lock:
            if not any(item is slot for item in self.slots) or slot.state != "filling":
                raise CaptureError("invalid or duplicate Host slot release")
            slot.state = "free" if transfer_complete else "quarantined"

    def stats(self) -> dict[str, int]:
        with self.lock:
            return {
                "allocated_bytes": self.allocated_bytes,
                "device_allocated_bytes": self.device_allocated_bytes,
                "device_limit_bytes": self.device_limit_bytes,
                **{
                    s: sum(slot.state == s for slot in self.slots)
                    for s in ("free", "filling", "quarantined")
                },
            }

    def close(self) -> None:
        with self.lock:
            if any(slot.state != "free" for slot in self.slots):
                raise CaptureError(
                    "cannot unregister buffers with outstanding or uncertain transfers"
                )
            # Keep failed registrations referenced if unregister itself fails.
            while self.slots:
                slot = self.slots[-1]
                self.registrar.unregister(slot.storage)
                self.slots.pop()
            self.allocated_bytes = 0
            self.device_allocated_bytes = 0
            self.closed = True
