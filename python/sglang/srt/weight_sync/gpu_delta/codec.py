"""Caller-owned nvCOMP decoding for direct weight deltas.

Preparation allocates small metadata and workspace; decoded output slots are bound
only after generation pauses.
The hot path calls the public C API directly: no nvCOMP Python array finalizers,
implicit size discovery, host scalar reads or wrapper-added synchronization.
The nvCOMP DE backend may wait for earlier work on the calling stream before
submitting decompression, even though the public API is named Async.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
import os
from collections.abc import Sequence
from dataclasses import dataclass

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


_DECOMPRESS_OPTIONS = {
    "snappy": ("Snappy", _SnappyOptions),
    "lz4": ("LZ4", _Lz4Options),
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


class NvcompDecoder:
    """One device's fixed inner-codec hardware decoder.

    Missing libraries and unsupported hardware fail admission. There is no CPU
    decoder, algorithm substitution or software fallback.
    """

    def __init__(self, device: torch.device, inner_codec: str):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Delta decoder requires an explicit CUDA device")
        if inner_codec not in _DECOMPRESS_OPTIONS:
            raise ValueError("GPU delta inner codec must be snappy or lz4")
        self.inner_codec = inner_codec
        self._algorithm, options_type = _DECOMPRESS_OPTIONS[inner_codec]
        sorting = os.environ.get("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", "0")
        if sorting not in {"0", "1"}:
            raise ValueError("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS must be 0 or 1")
        cuda_major = torch.version.cuda.split(".")[0]
        package = f"nvidia-libnvcomp-cu{cuda_major}"
        distribution = importlib.metadata.distribution(package)
        version = tuple(int(v) for v in distribution.version.split(".")[:2])
        if not (5, 3) <= version < (6, 0) or ctypes.sizeof(ctypes.c_size_t) != 8:
            raise RuntimeError("Direct GPU deltas require the 64-bit nvCOMP 5.3+ ABI")
        self.backend = "hardware"
        self._options = options_type()
        # Explicit backend selection: DEFAULT can silently select software.
        self._options.backend = 1
        self._options.sort_before_hw_decompress = int(sorting)
        if self._algorithm == "LZ4":
            self._options.data_type = 0  # NVCOMP_TYPE_CHAR
            self._options.bitshuffle_mode = 0  # NVCOMP_BITSHUFFLE_NONE
        self.maximum_chunk_bytes = _require_hardware_device(
            self.device, self._algorithm
        )
        self._library = ctypes.CDLL(
            str(distribution.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.5"))
        )
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        self._temporary = self._bind(
            "GetTempSizeAsync", [size, size, options_type, ctypes.POINTER(size), size]
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
        self._temporary_sizes: dict[tuple[int, int, int], int] = {}

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

    def temporary_bytes(self, geometry: tuple[int, int, int]) -> int:
        if not geometry[0]:
            return 0
        cached = self._temporary_sizes.get(geometry)
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
                )
            )
        self._temporary_sizes[geometry] = size.value
        return size.value

    def _allocate_workspace(self, geometry) -> DecodeWorkspace:
        """One DE stream shares scratch; statuses/sizes belong to two slots."""
        from sglang.srt.weight_sync.gpu_delta.memory import (
            device_empty,
            require_de_capable,
        )

        maximum_count = max((batch[0] for batch in geometry), default=0)
        temporary = max((self.temporary_bytes(batch) for batch in geometry), default=0)
        workspace = DecodeWorkspace(
            device_empty((temporary,), dtype=torch.uint8, device=self.device),
            device_empty((2, maximum_count), dtype=torch.int64, device=self.device),
            device_empty((2, maximum_count), dtype=torch.int32, device=self.device),
        )
        if temporary and workspace.temporary.data_ptr() % self.alignments.temp:
            raise ValueError("Misaligned decoder workspace")
        for tensor in (
            workspace.temporary,
            workspace.actual_sizes,
            workspace.statuses,
        ):
            if tensor.numel():
                require_de_capable(tensor.data_ptr())
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
            if count > _INT64_MAX // maximum:
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
        until the DE stream drains. Only input/size rows are uploaded now;
        ``bind_outputs`` fills output pointers under the serving pause.
        """
        from sglang.srt.weight_sync.gpu_delta.memory import (
            device_empty,
            require_de_capable,
        )

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
            workspace = self._allocate_workspace(geometry)
            host = torch.empty(
                table.shape, dtype=torch.int64, device="cpu", pin_memory=True
            )
            host_rows = host.numpy()
            np.add(table[0], input_base, out=host_rows[0])
            host_rows[1:3] = table[1:3]
            metadata = device_empty(host.shape, dtype=torch.int64, device=self.device)
            if metadata.numel():
                require_de_capable(metadata.data_ptr())
                metadata[:3].copy_(host[:3], non_blocking=True)
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
        )


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
    ):
        self.decoder, self.stream = decoder, stream
        self.workspace = workspace
        self.host_metadata, self.metadata = host_metadata, metadata
        self.output_bounds, self.output_remainders = output_bounds, output_remainders
        # Every batch retains this same list. Binding fills it only after all
        # output checks pass; dropping the plan alone cannot free in-flight slots.
        self.decoded_slots = []
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
                )
            )
            offset += count
        self.record_stream(stream)

    def record_stream(self, stream):
        """Register every consumer before it can enqueue storage readers."""
        from sglang.srt.weight_sync.gpu_delta.memory import record_device_stream

        if stream.device != self.decoder.device:
            raise ValueError("Decoder consumer stream/device mismatch")
        for tensor in (
            self.metadata,
            self.workspace.temporary,
            self.workspace.actual_sizes,
            self.workspace.statuses,
            *self.decoded_slots,
        ):
            record_device_stream(tensor, stream)

    @profiling.annotate("gpu_delta.pause.bind_decoder_outputs")
    def bind_outputs(
        self, decoded_slots: Sequence[torch.Tensor]
    ) -> list[PreparedDecode]:
        """Check paused allocations, fill output pointers, upload just row 3.

        Slot bounds and relative pointer alignment were computed during prepare.
        The caller waits for prior apply readers before reusing each output and
        its status/actual-size row. Temporary storage is reused on the DE stream.
        """
        from sglang.srt.weight_sync.gpu_delta.memory import (
            record_device_stream,
            require_de_capable,
        )

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
            record_device_stream(tensor, self.stream)
        with torch.cuda.device(self.decoder.device), torch.cuda.stream(self.stream):
            for base, capacity in zip(bases, capacities):
                if capacity:
                    require_de_capable(base)
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

    @profiling.annotate("gpu_delta.pause.nvcomp_enqueue")
    def enqueue(self) -> None:
        """Launch only; caller checks statuses/sizes on device before applying weights.

        This object and all storage must outlive the submitted work. There is no
        normal-path synchronization here; the session owns the final fence and
        enters the captured DE stream/device context around submission. Foreign
        allocation finalizers also fence their recorded consumers before freeing.
        """
        if not self.statuses.numel():
            return
        self.decoder._check(self.decoder._decode(*self._arguments))
