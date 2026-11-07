"""Caller-owned nvCOMP decoding for direct weight deltas.

Preparation allocates small metadata and workspace; decoded output slots are bound
only after generation pauses.
The hot path calls the public C API directly: no nvCOMP Python array finalizers,
implicit size discovery, host scalar reads or wrapper-added synchronization.
The nvCOMP DE backend and Bitcomp batched decoder may wait for earlier work on
the calling stream before submitting decompression, despite the Async API name.
CUDA decoding uses two compressed device slots so H2D can overlap decode/apply.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np
import torch

from sglang.srt.weight_sync.gpu_delta import profiling


class _SnappyOptions(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("sort_before_hw_decompress", ctypes.c_int),
        ("reserved", ctypes.c_char * 56),
    ]


class _Lz4Options(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("sort_before_hw_decompress", ctypes.c_int),
        ("data_type", ctypes.c_int),
        ("bitshuffle_mode", ctypes.c_int),
        ("reserved", ctypes.c_char * 48),
    ]


class _BitcompOptions(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("reserved", ctypes.c_char * 60)]


_DECOMPRESS_OPTIONS = {
    "snappy": ("Snappy", _SnappyOptions),
    "lz4": ("LZ4", _Lz4Options),
    "bitcomp": ("Bitcomp", _BitcompOptions),
}
_MAX_FRAME_BYTES = 4 << 20
_INT64_MAX = np.iinfo(np.int64).max


def _require_hardware_device(device: torch.device, algorithm: str) -> int:
    from sglang.srt.weight_sync.gpu_delta.memory import _driver

    driver = _driver()
    status, mask = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK,
        device.index,
    )
    if status:
        raise RuntimeError(f"GPU-delta DE device admission failed: CUDA {status}")
    status, maximum = driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH,
        device.index,
    )
    if status:
        raise RuntimeError(f"GPU-delta DE device admission failed: CUDA {status}")
    algorithms = driver.CUmemDecompressAlgorithm
    required = {
        "Snappy": algorithms.CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY,
        "LZ4": algorithms.CU_MEM_DECOMPRESS_ALGORITHM_LZ4,
    }[algorithm]
    if not mask & required:
        raise RuntimeError(f"Device has no hardware {algorithm} decompressor")
    if maximum <= 0:
        raise RuntimeError("Device has no positive hardware decompression limit")
    return maximum


class _Alignments(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_size_t),
        ("output", ctypes.c_size_t),
        ("temp", ctypes.c_size_t),
    ]


@dataclass
class DecodeWorkspace:
    temporary: torch.Tensor
    actual_sizes: torch.Tensor
    statuses: torch.Tensor


@cache
def decoder_backend(device, inner_codec):
    """One backend choice for the decoder and its engine-host publication."""
    hardware = inner_codec in {"snappy", "lz4"} and torch.cuda.get_device_capability(
        device
    ) in {(10, 0), (10, 3)}
    return "hardware" if hardware else "cuda"


class NvcompDecoder:
    """DE for Snappy/LZ4 on SM100/103; CUDA decoding for other combinations."""

    def __init__(self, device: torch.device, inner_codec: str):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Delta decoder requires an explicit CUDA device")
        if inner_codec not in _DECOMPRESS_OPTIONS:
            raise ValueError("GPU delta inner codec must be snappy, lz4 or bitcomp")
        self.inner_codec = inner_codec
        self._algorithm, options_type = _DECOMPRESS_OPTIONS[inner_codec]
        sorting = os.environ.get("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", "0")
        if sorting not in {"0", "1"}:
            raise ValueError("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS must be 0 or 1")
        cuda_major = torch.version.cuda.split(".")[0]
        package = f"nvidia-libnvcomp-cu{cuda_major}"
        distribution = importlib.metadata.distribution(package)
        major_version = int(distribution.version.split(".")[0])
        if major_version != 6 or ctypes.sizeof(ctypes.c_size_t) != 8:
            raise RuntimeError("Direct GPU deltas require the 64-bit nvCOMP 6 ABI")
        self.backend = decoder_backend(self.device, inner_codec)
        hardware = self.backend == "hardware"
        self._options = options_type()
        self._options.backend = 1 if hardware else 2
        if inner_codec in {"snappy", "lz4"}:
            self._options.sort_before_hw_decompress = int(sorting) if hardware else 0
        if self._algorithm == "LZ4":
            self._options.data_type = 0  # NVCOMP_TYPE_CHAR
            self._options.bitshuffle_mode = 0  # NVCOMP_BITSHUFFLE_NONE
        self.maximum_chunk_bytes = (
            _require_hardware_device(self.device, self._algorithm)
            if hardware
            else _INT64_MAX
        )
        self._library = ctypes.CDLL(
            str(distribution.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.6"))
        )
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        self._temporary = self._bind(
            "GetTempSize",
            [size, size, options_type, ctypes.POINTER(size), size, pointer],
        )
        required_alignments = self._bind(
            "GetRequiredAlignments", [options_type, ctypes.POINTER(_Alignments)]
        )
        self._decode = self._bind(
            "Async",
            [
                pointer,
                pointer,
                pointer,
                pointer,
                size,
                pointer,
                size,
                pointer,
                options_type,
                pointer,
                pointer,
            ],
        )
        self.alignments = _Alignments()
        self._check(required_alignments(self._options, ctypes.byref(self.alignments)))
        self._temporary_sizes: dict[tuple[int, int, int, int], int] = {}

    def _bind(self, suffix, arguments):
        function = getattr(
            self._library,
            "nvcompBatched" + self._algorithm + "Decompress" + suffix,
        )
        function.argtypes, function.restype = arguments, ctypes.c_int
        return function

    def _check(self, status):
        if status != 0:
            raise RuntimeError(
                f"nvCOMP {self.inner_codec}/{self.backend} failed: status={status}"
            )

    def temporary_bytes(self, geometry: tuple[int, int, int], cuda_stream: int) -> int:
        if not geometry[0]:
            return 0
        key = (*geometry, cuda_stream)
        cached = self._temporary_sizes.get(key)
        if cached is not None:
            return cached
        size = ctypes.c_size_t()
        with torch.cuda.device(self.device):
            self._check(
                self._temporary(
                    geometry[0],
                    geometry[1],
                    self._options,
                    ctypes.byref(size),
                    geometry[2],
                    cuda_stream,
                )
            )
        self._temporary_sizes[key] = size.value
        return size.value

    def _empty(self, shape, dtype):
        if self.backend == "hardware":
            from sglang.srt.weight_sync.gpu_delta.memory import device_empty

            return device_empty(shape, dtype=dtype, device=self.device)
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _record_stream(self, tensor, stream):
        if self.backend == "hardware":
            from sglang.srt.weight_sync.gpu_delta.memory import record_device_stream

            record_device_stream(tensor, stream)
        else:
            tensor.record_stream(stream)

    def _allocate_workspace(self, geometry, cuda_stream) -> DecodeWorkspace:
        """One decode stream shares scratch; statuses/sizes belong to two slots."""

        maximum_count = max((batch[0] for batch in geometry), default=0)
        temporary = max(
            (self.temporary_bytes(batch, cuda_stream) for batch in geometry), default=0
        )
        workspace = DecodeWorkspace(
            self._empty((temporary,), torch.uint8),
            self._empty((2, maximum_count), torch.int64),
            self._empty((2, maximum_count), torch.int32),
        )
        if temporary and workspace.temporary.data_ptr() % self.alignments.temp:
            raise ValueError("Misaligned decoder workspace")
        return workspace

    def _frame_geometry(self, table, counts, input_base, input_bytes):
        """Validate numeric rows and derive workspace/output bounds in one pass."""
        if (
            table.dtype != np.int64
            or table.shape != (4, sum(counts))
            or not table.flags.c_contiguous
        ):
            raise ValueError("Expected contiguous int64(4,N) frame metadata")
        if not 0 <= input_base <= _INT64_MAX - input_bytes:
            raise ValueError("Input pointer arithmetic exceeds signed metadata")
        bounds, remainders, geometry = [0, 0], [None, None], []
        maximum = self.maximum_chunk_bytes
        offset = 0
        for index, count in enumerate(counts):
            inputs, encoded, decoded, outputs = table[:, offset : offset + count]
            offset += count
            slot = index % 2
            if not count:
                geometry.append((0, 0, 0))
                continue
            # HARDWARE does not check oversized buffers. Check actual lengths,
            # not the compressor's worst-case allocation: compressible 4 MiB
            # frames can fit the device limit.
            if (
                np.any(decoded <= 0)
                or np.any(decoded > min(_MAX_FRAME_BYTES, maximum))
                or np.any(encoded <= 0)
                or np.any(encoded > maximum)
            ):
                raise ValueError(
                    "GPU-delta frames require positive lengths, <=4 MiB "
                    f"output and both lengths <= device limit {maximum}"
                )
            if np.any(inputs < 0) or np.any(inputs > input_bytes - encoded):
                raise ValueError("Encoded frame outside input allocation")
            if np.any(outputs < 0) or np.any(outputs > _INT64_MAX - decoded):
                raise ValueError("Output arithmetic exceeds signed metadata")
            if np.any(outputs[1:] < outputs[:-1] + decoded[:-1]):
                raise ValueError("Overlapping or unordered decoded frames")
            if np.any((inputs + input_base) % self.alignments.input):
                raise ValueError("Misaligned encoded frame")
            remainder = outputs % self.alignments.output
            first = int(remainder[0])
            if np.any(remainder != first) or remainders[slot] not in (None, first):
                raise ValueError("Misaligned decoded frame")
            remainders[slot] = first
            bounds[slot] = max(bounds[slot], int(outputs[-1] + decoded[-1]))
            if count > _INT64_MAX // int(decoded.max()):
                raise ValueError("Decoded geometry sum exceeds signed metadata")
            geometry.append((count, int(decoded.max()), int(decoded.sum())))
        return geometry, bounds, remainders

    @profiling.annotate("gpu_delta.prepare.decoder_metadata")
    def prepare_batches(
        self,
        table: np.ndarray,
        counts: Sequence[int],
        host_input: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> PreparedDecodePlan:
        """Prepare numeric metadata and small workspace without decoded output.

        Rows contain relative input offsets, encoded sizes, decoded sizes and
        relative output offsets. The admitted host arena must stay immutable
        until decoding completes. Size rows and DE input pointers are uploaded now;
        ``bind_outputs`` fills output pointers and CUDA input pointers while paused.
        """
        if (
            host_input.device.type != "cpu"
            or host_input.dtype != torch.uint8
            or not host_input.is_contiguous()
        ):
            raise ValueError("nvCOMP input must be a contiguous host arena byte view")
        if stream.device != self.device:
            raise ValueError("Decoder stream/device mismatch")
        input_base = host_input.data_ptr()
        geometry, bounds, remainders = self._frame_geometry(
            table, counts, input_base, host_input.numel()
        )
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            workspace = self._allocate_workspace(geometry, stream.cuda_stream)
            host = torch.empty(
                table.shape, dtype=torch.int64, device="cpu", pin_memory=True
            )
            host_rows = host.numpy()
            np.add(table[0], input_base, out=host_rows[0])
            host_rows[1:3] = table[1:3]
            metadata = self._empty(host.shape, torch.int64)
            if metadata.numel():
                first = 0 if self.backend == "hardware" else 1
                metadata[first:3].copy_(host[first:3], non_blocking=True)
        return PreparedDecodePlan(
            self,
            counts,
            host_input,
            workspace,
            host,
            metadata,
            stream,
            table[3].copy(),
            bounds,
            remainders,
            table[:2],
        )


def _pack_inputs(inputs, sizes, alignment):
    """Merge adjacent frames into bounded H2D spans, retaining frame alignment."""
    if not len(inputs):
        return np.empty(0, dtype=np.int64), (), 0
    alignment = max(16, alignment)  # Publication frames carry 16-byte padding.
    order = np.argsort(inputs)
    starts = inputs[order]
    ends = np.maximum.accumulate(starts + sizes[order])
    boundaries = np.r_[True, starts[1:] > ends[:-1] + alignment - 1]
    first = np.flatnonzero(boundaries)
    last = np.r_[first[1:] - 1, len(starts) - 1]
    lengths = ends[last] - starts[first]
    padded = (lengths + alignment - 1) // alignment * alignment
    destinations = np.r_[0, np.cumsum(padded[:-1])]
    groups = np.cumsum(boundaries) - 1
    offsets = np.empty_like(inputs)
    offsets[order] = starts - starts[first][groups] + destinations[groups]
    ranges = tuple(
        (int(source), int(destination), int(length))
        for source, destination, length in zip(starts[first], destinations, lengths)
    )
    return offsets, ranges, int(destinations[-1] + lengths[-1])


class PreparedDecodePlan:
    """Publication metadata and output leases, bound only while paused."""

    def __init__(
        self,
        decoder,
        counts,
        host_input,
        workspace,
        host_metadata,
        metadata,
        stream,
        output_offsets,
        output_bounds,
        output_remainders,
        input_frames=None,
    ):
        self.decoder, self.stream = decoder, stream
        self.workspace = workspace
        self.host_metadata, self.metadata = host_metadata, metadata
        self.output_bounds, self.output_remainders = output_bounds, output_remainders
        # Every batch retains this same list. Binding fills it only after all
        # output checks pass; dropping the plan alone cannot free in-flight slots.
        self.decoded_slots = []
        self.input_slots = []
        self.input_bytes = self.input_capacity_bytes = 0
        self.input_bounds = [0, 0]
        self.batches = []
        temporary = workspace.temporary
        offset = 0
        for index, count in enumerate(counts):
            slot = index % 2
            rows = metadata[:, offset : offset + count]
            host_rows = host_metadata[:, offset : offset + count]
            statuses = workspace.statuses[slot, :count]
            actual_sizes = workspace.actual_sizes[slot, :count]
            expected_sizes = rows[2]
            arguments = (
                rows[0].data_ptr(),
                rows[1].data_ptr(),
                expected_sizes.data_ptr(),
                actual_sizes.data_ptr(),
                count,
                temporary.data_ptr(),
                temporary.numel(),
                rows[3].data_ptr(),
                decoder._options,
                statuses.data_ptr(),
                stream.cuda_stream,
            )
            if decoder.backend == "cuda":
                inputs, sizes = input_frames[:, offset : offset + count]
                input_offsets, copy_ranges, capacity = _pack_inputs(
                    inputs, sizes, decoder.alignments.input
                )
                self.input_bounds[slot] = max(self.input_bounds[slot], capacity)
                self.input_bytes += sum(length for _, _, length in copy_ranges)
            else:
                input_offsets, copy_ranges = None, ()
            self.batches.append(
                PreparedDecode(
                    decoder,
                    host_input,
                    self.decoded_slots,
                    workspace,
                    host_rows,
                    rows,
                    stream,
                    statuses,
                    actual_sizes,
                    expected_sizes,
                    arguments,
                    output_offsets[offset : offset + count],
                    host_rows[3].numpy(),
                    self.input_slots,
                    input_offsets,
                    copy_ranges,
                    [],
                )
            )
            offset += count
        self.record_stream(stream)

    def record_stream(self, stream):
        """Register every consumer before it can enqueue storage readers."""
        if stream.device != self.decoder.device:
            raise ValueError("Decoder consumer stream/device mismatch")
        for tensor in (
            self.metadata,
            self.workspace.temporary,
            self.workspace.actual_sizes,
            self.workspace.statuses,
            *self.decoded_slots,
            *self.input_slots,
        ):
            self.decoder._record_stream(tensor, stream)

    @profiling.annotate("gpu_delta.pause.bind_decoder_outputs")
    def bind_outputs(
        self, decoded_slots: Sequence[torch.Tensor]
    ) -> list[PreparedDecode]:
        """Bind paused allocations: output row 3 and CUDA input row 0.

        Slot bounds and relative pointer alignment were computed during prepare.
        The caller waits for prior apply readers before reusing each output and
        its status/actual-size row. Scratch is reused on the selected decode stream.
        """
        from sglang.srt.weight_sync.gpu_delta.memory import require_de_capable

        if len(decoded_slots) != 2:
            raise ValueError("Decoded output count differs from prepared slots")
        for tensor in decoded_slots:
            if (
                tensor.device != self.decoder.device
                or tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "nvCOMP outputs must be contiguous bytes on the decoder device"
                )
        bases = [tensor.data_ptr() for tensor in decoded_slots]
        capacities = [tensor.numel() for tensor in decoded_slots]
        if (
            capacities[0]
            and capacities[1]
            and bases[0] < bases[1] + capacities[1]
            and bases[1] < bases[0] + capacities[0]
        ):
            raise ValueError("Decoded output slots must not overlap")
        for base, capacity, bound, remainder in zip(
            bases, capacities, self.output_bounds, self.output_remainders
        ):
            if capacity < bound:
                raise ValueError("Out-of-bounds decoded frame")
            if (
                remainder is not None
                and (base + remainder) % self.decoder.alignments.output
            ):
                raise ValueError("Misaligned decoded frame")
        for tensor in decoded_slots:
            self.decoder._record_stream(tensor, self.stream)
        with torch.cuda.device(self.decoder.device), torch.cuda.stream(self.stream):
            if self.decoder.backend == "hardware":
                for base, capacity in zip(bases, capacities):
                    if capacity:
                        require_de_capable(base)
            else:
                self.input_slots[:] = [
                    self.decoder._empty(size, torch.uint8) for size in self.input_bounds
                ]
                self.input_capacity_bytes = sum(self.input_bounds)
                for index, batch in enumerate(self.batches):
                    slot = self.input_slots[index % 2]
                    np.add(
                        batch.input_offsets,
                        slot.data_ptr(),
                        out=batch.host_metadata[0].numpy(),
                    )
                    batch.input_copies[:] = [
                        (
                            slot[destination : destination + length],
                            batch.host_input[source : source + length],
                        )
                        for source, destination, length in batch.copy_ranges
                    ]
                if self.metadata.numel():
                    self.metadata[0].copy_(self.host_metadata[0], non_blocking=True)
            for index, batch in enumerate(self.batches):
                np.add(
                    batch.output_offsets,
                    bases[index % 2],
                    out=batch.output_pointers,
                )
            if self.metadata.numel():
                self.metadata[3].copy_(self.host_metadata[3], non_blocking=True)
        self.decoded_slots[:] = decoded_slots
        return self.batches


@dataclass
class PreparedDecode:
    decoder: NvcompDecoder
    host_input: torch.Tensor
    decoded_slots: list[torch.Tensor]
    workspace: DecodeWorkspace
    host_metadata: torch.Tensor
    metadata: torch.Tensor
    stream: torch.cuda.Stream
    statuses: torch.Tensor
    actual_sizes: torch.Tensor
    expected_sizes: torch.Tensor
    _arguments: tuple
    output_offsets: np.ndarray
    output_pointers: np.ndarray
    input_slots: list[torch.Tensor]
    input_offsets: np.ndarray | None
    copy_ranges: tuple
    input_copies: list

    @profiling.annotate("gpu_delta.pause.h2d_input")
    def copy_input(self, stream) -> None:
        if self.input_copies:
            self.decoder._record_stream(self.input_copies[0][0], stream)
        with torch.cuda.stream(stream):
            for destination, source in self.input_copies:
                destination.copy_(source, non_blocking=True)

    @profiling.annotate("gpu_delta.pause.nvcomp_enqueue")
    def enqueue(self) -> None:
        """Launch only; caller checks statuses/sizes on device before applying weights.

        This object and all storage must outlive the submitted work. There is no
        explicit synchronization here; nvCOMP Bitcomp may synchronize internally
        to inspect headers. The receiver queues the next H2D transfer first.
        The session owns the final fence and calling-stream context. Foreign
        allocation finalizers also fence their recorded consumers before freeing.
        """
        if not self.statuses.numel():
            return
        self.decoder._check(self.decoder._decode(*self._arguments))
