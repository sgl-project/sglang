"""Frozen Scheduler buffer descriptors for the Rust PD transport."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

PD_REGION_COUNT = 58
PD_KV_REGION_COUNT = 56
PD_PAGE_SIZE = 64
PD_KV_HEADS = 8
PD_HEAD_DIM = 128
PD_KV_ROW_BYTES = PD_KV_HEADS * PD_HEAD_DIM * 2
PD_KV_PAGE_BYTES = PD_PAGE_SIZE * PD_KV_ROW_BYTES
PD_AUX_SLOTS = 32
PD_AUX_BYTES = 64
PD_COMPLETION_SLOTS = 32
PD_COMPLETION_BYTES = 192


@dataclass(frozen=True)
class PdRegionDescriptor:
    region_id: int
    address: int
    length_bytes: int
    device: str
    dtype: str
    shape: tuple[int, ...]
    stride_bytes: tuple[int, ...]
    generation: int

    def transport_value(self) -> dict[str, Any]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        value["stride_bytes"] = list(self.stride_bytes)
        return value


class StablePdRegionTable:
    """Immutable 56-KV + pinned aux/completion Scheduler buffer snapshot."""

    def __init__(
        self,
        descriptors: Sequence[PdRegionDescriptor],
        tensor_owners: Sequence[Any],
        pool_generation: int,
    ) -> None:
        if len(descriptors) != PD_REGION_COUNT or len(tensor_owners) != PD_REGION_COUNT:
            raise ValueError("PD_PROTOCOL_MISMATCH")
        if [descriptor.region_id for descriptor in descriptors] != list(
            range(PD_REGION_COUNT)
        ):
            raise ValueError("PD_PROTOCOL_MISMATCH")
        self.descriptors = tuple(descriptors)
        self.tensor_owners = tuple(tensor_owners)
        self.pool_generation = pool_generation

    @classmethod
    def capture(
        cls,
        pool: Any,
        aux_tensor: Any,
        completion_tensor: Any,
        *,
        generation: int,
    ) -> StablePdRegionTable:
        if generation <= 0 or getattr(pool, "pd_generation", generation) != generation:
            raise ValueError("PD_STALE_EPOCH")
        if getattr(pool, "page_size", None) != PD_PAGE_SIZE:
            raise ValueError("PD_UNSUPPORTED")
        tensors = _registerable_tensors(pool)
        if len(tensors) != PD_KV_REGION_COUNT:
            raise ValueError("PD_UNSUPPORTED")

        descriptors = [
            _kv_descriptor(index, tensor, generation)
            for index, tensor in enumerate(tensors)
        ]
        descriptors.append(
            _host_descriptor(
                56,
                aux_tensor,
                generation,
                expected_shape=(PD_AUX_SLOTS, PD_AUX_BYTES),
            )
        )
        descriptors.append(
            _host_descriptor(
                57,
                completion_tensor,
                generation,
                expected_shape=(PD_COMPLETION_SLOTS, PD_COMPLETION_BYTES),
            )
        )
        devices = {descriptor.device for descriptor in descriptors[:56]}
        if len(devices) != 1:
            raise ValueError("PD_UNSUPPORTED")
        return cls(
            descriptors,
            [*tensors, aux_tensor, completion_tensor],
            generation,
        )

    def verify(self, pool: Any) -> None:
        if getattr(pool, "pd_generation", self.pool_generation) != self.pool_generation:
            raise RuntimeError("PD_STALE_EPOCH")
        if getattr(pool, "page_size", None) != PD_PAGE_SIZE:
            raise RuntimeError("PD_STALE_EPOCH")
        tensors = _registerable_tensors(pool)
        if len(tensors) != PD_KV_REGION_COUNT:
            raise RuntimeError("PD_STALE_EPOCH")
        current = [
            _kv_descriptor(index, tensor, self.pool_generation)
            for index, tensor in enumerate(tensors)
        ]
        current.extend(
            [
                _host_descriptor(
                    56,
                    self.tensor_owners[56],
                    self.pool_generation,
                    expected_shape=(PD_AUX_SLOTS, PD_AUX_BYTES),
                ),
                _host_descriptor(
                    57,
                    self.tensor_owners[57],
                    self.pool_generation,
                    expected_shape=(PD_COMPLETION_SLOTS, PD_COMPLETION_BYTES),
                ),
            ]
        )
        if tuple(current) != self.descriptors:
            raise RuntimeError("PD_STALE_EPOCH")

    def transport_values(self) -> list[dict[str, Any]]:
        return [descriptor.transport_value() for descriptor in self.descriptors]


def _registerable_tensors(pool: Any) -> list[Any]:
    get_tensors = getattr(pool, "_pd_registerable_tensors", None)
    if get_tensors is None:
        return [*getattr(pool, "k_buffer", []), *getattr(pool, "v_buffer", [])]
    return list(get_tensors())


def _kv_descriptor(
    region_id: int,
    tensor: Any,
    generation: int,
) -> PdRegionDescriptor:
    address = int(tensor.data_ptr())
    raw_shape = tuple(int(value) for value in tensor.shape)
    raw_stride = tuple(int(value) for value in tensor.stride())
    device = _device_name(tensor.device)
    rows = raw_shape[0] if len(raw_shape) == 3 else 0
    if (
        address <= 0
        or address % 64 != 0
        or rows <= 0
        or rows % PD_PAGE_SIZE != 0
        or raw_shape[1:] != (PD_KV_HEADS, PD_HEAD_DIM)
        or raw_stride != (PD_KV_HEADS * PD_HEAD_DIM, PD_HEAD_DIM, 1)
        or int(tensor.element_size()) != 2
        or str(tensor.dtype) != "torch.bfloat16"
        or not tensor.is_contiguous()
        or not device.startswith("cuda:")
    ):
        raise ValueError("PD_UNSUPPORTED")
    page_capacity = rows // PD_PAGE_SIZE
    return PdRegionDescriptor(
        region_id=region_id,
        address=address,
        length_bytes=page_capacity * PD_KV_PAGE_BYTES,
        device=device,
        dtype="torch.bfloat16",
        shape=(page_capacity, PD_PAGE_SIZE, PD_KV_HEADS, PD_HEAD_DIM),
        stride_bytes=(PD_KV_PAGE_BYTES, PD_KV_ROW_BYTES, PD_HEAD_DIM * 2, 2),
        generation=generation,
    )


def _host_descriptor(
    region_id: int,
    tensor: Any,
    generation: int,
    *,
    expected_shape: tuple[int, int],
) -> PdRegionDescriptor:
    address = int(tensor.data_ptr())
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    length_bytes = int(tensor.numel()) * int(tensor.element_size())
    device = _device_name(tensor.device)
    if (
        address <= 0
        or address % 64 != 0
        or device != "cpu:0"
        or str(tensor.dtype) != "torch.uint8"
        or shape != expected_shape
        or stride != (expected_shape[1], 1)
        or length_bytes != expected_shape[0] * expected_shape[1]
        or not tensor.is_contiguous()
        or not tensor.is_pinned()
    ):
        raise ValueError("PD_UNSUPPORTED")
    return PdRegionDescriptor(
        region_id=region_id,
        address=address,
        length_bytes=length_bytes,
        device=device,
        dtype="torch.uint8",
        shape=shape,
        stride_bytes=stride,
        generation=generation,
    )


def _device_name(device: Any) -> str:
    device_type = str(device.type)
    index = getattr(device, "index", None)
    return f"{device_type}:{0 if index is None else int(index)}"
