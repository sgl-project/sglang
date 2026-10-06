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
    "snappy-zstd": ("Snappy", _SnappyOptions),
    "lz4-zstd": ("LZ4", _Lz4Options),
}
_MAX_FRAME_BYTES = 4 << 20


def _require_hardware_device(device: torch.device, algorithm: str) -> int:
    from sglang.srt.weight_sync.gpu_delta_memory import _driver

    driver = _driver()
    mask, maximum = ctypes.c_int(), ctypes.c_int()
    for attribute, result in ((136, mask), (137, maximum)):
        status = driver.cuDeviceGetAttribute(
            ctypes.byref(result), attribute, device.index
        )
        if status:
            raise RuntimeError(f"GPU-delta DE device admission failed: CUDA {status}")
    # CUDA's CUmemDecompressAlgorithm bits, not compute-capability inference.
    if not mask.value & {"Snappy": 1 << 1, "LZ4": 1 << 2}[algorithm]:
        raise RuntimeError(f"Device has no hardware {algorithm} decompressor")
    if maximum.value <= 0:
        raise RuntimeError("Device has no positive hardware decompression limit")
    return maximum.value


class _Alignments(ctypes.Structure):
    _fields_ = [
        ("input", ctypes.c_size_t),
        ("output", ctypes.c_size_t),
        ("temp", ctypes.c_size_t),
    ]


@dataclass(frozen=True)
class DecodeFrame:
    input_offset: int
    encoded_bytes: int
    output_offset: int
    decoded_bytes: int


@dataclass
class DecodeWorkspace:
    temporary: torch.Tensor
    actual_sizes: torch.Tensor
    statuses: torch.Tensor


def _require_hardware_allocator(device: torch.device) -> None:
    # Native caching alone is insufficient: expandable_segments uses CUDA VMM
    # without the hardware-decompression allocation flag. Inspect effective
    # settings, not only the environment (PyTorch can change them at runtime).
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise RuntimeError("GPU deltas require native CUDA allocations")
    snapshot = torch.cuda.memory._snapshot()
    expandable = snapshot.get("allocator_settings", {}).get("expandable_segments")
    if expandable is not False:
        raise RuntimeError(
            "Hardware decoding requires verifiable expandable_segments:False; "
            "PyTorch expandable VMM buffers are not admitted"
        )
    if any(
        segment.get("device") == device.index and segment.get("is_expandable", False)
        for segment in snapshot.get("segments", [])
    ):
        raise RuntimeError(
            "Hardware decoding cannot reuse existing PyTorch expandable VMM segments; "
            "start the process with expandable_segments:False"
        )


class NvcompDecoder:
    """One device's fixed inner-codec hardware decoder.

    Missing libraries and unsupported hardware fail admission. There is no CPU
    decoder, algorithm substitution or software fallback.
    """

    def __init__(self, device: torch.device, codec: str):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Delta decoder requires an explicit CUDA device")
        if codec not in _DECOMPRESS_OPTIONS:
            raise ValueError("GPU delta codec must be snappy-zstd or lz4-zstd")
        self.codec = codec
        self._algorithm, options_type = _DECOMPRESS_OPTIONS[codec]
        sorting = os.environ.get("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS", "0")
        if sorting not in {"0", "1"}:
            raise ValueError("GPU_DELTA_SORT_BEFORE_HW_DECOMPRESS must be 0 or 1")
        cuda_major = torch.version.cuda.split(".")[0]
        package = f"nvidia-libnvcomp-cu{cuda_major}"
        distribution = importlib.metadata.distribution(package)
        version = tuple(int(v) for v in distribution.version.split(".")[:2])
        if not (5, 3) <= version < (6, 0) or ctypes.sizeof(ctypes.c_size_t) != 8:
            raise RuntimeError("Direct GPU deltas require the 64-bit nvCOMP 5.3+ ABI")
        self.version = distribution.version
        self.backend = "hardware"
        self._options = options_type()
        # Explicit backend selection: DEFAULT can silently select software.
        self._options.backend = 1
        self._options.sort_before_hw_decompress = int(sorting)
        if codec == "lz4-zstd":
            self._options.data_type = 0  # NVCOMP_TYPE_CHAR
            self._options.bitshuffle_mode = 0  # NVCOMP_BITSHUFFLE_NONE
        self.maximum_chunk_bytes = _require_hardware_device(
            self.device, self._algorithm
        )
        _require_hardware_allocator(self.device)
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
                f"nvCOMP {self.codec}/{self.backend} failed: status={status}"
            )

    def temporary_bytes(self, frames: Sequence[DecodeFrame]) -> int:
        if not frames:
            return 0
        geometry = (
            len(frames),
            max(f.decoded_bytes for f in frames),
            sum(f.decoded_bytes for f in frames),
        )
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

    def allocate_workspace(
        self, batches: Sequence[Sequence[DecodeFrame]], slot_count: int = 2
    ) -> DecodeWorkspace:
        """Allocate small workspace during prepare; retain the decoder to cache queries.

        DE submissions use one stream and share temporary storage. Status and
        actual-size rows belong to each decoded slot until apply consumes them.
        """
        from sglang.srt.weight_sync.gpu_delta_memory import require_de_capable

        if slot_count not in (2, 3, 4):
            raise ValueError("nvCOMP requires 2, 3 or 4 decoded output slots")
        maximum_count = max((len(batch) for batch in batches), default=0)
        temporary = max((self.temporary_bytes(batch) for batch in batches), default=0)
        with torch.cuda.device(self.device):
            workspace = DecodeWorkspace(
                torch.empty(temporary, dtype=torch.uint8, device=self.device),
                torch.empty(
                    (slot_count, maximum_count), dtype=torch.int64, device=self.device
                ),
                torch.empty(
                    (slot_count, maximum_count), dtype=torch.int32, device=self.device
                ),
            )
            for tensor in (
                workspace.temporary,
                workspace.actual_sizes,
                workspace.statuses,
            ):
                if tensor.numel():
                    require_de_capable(tensor.data_ptr())
        return workspace

    def prepare_batches(
        self,
        batches: Sequence[Sequence[DecodeFrame]],
        host_input: torch.Tensor,
        workspace: DecodeWorkspace,
        stream: torch.cuda.Stream,
    ) -> PreparedDecodePlan:
        """Prepare immutable host-input metadata without allocating decoded output.

        The host arena admits its allocation's hardware-DE capability once per
        capacity generation. Its CPU byte view is device-accessible and must stay
        immutable until the captured DE stream drains. Input and size rows are
        uploaded now; the output pointer row is filled by ``bind_outputs`` under
        the serving pause. No decompression is submitted during preparation.
        All DE submissions share the captured stream and temporary workspace;
        status validation and weight writes belong to the separate apply stream.
        ``workspace`` must be allocated for these batches with ``allocate_workspace``.
        """
        from sglang.srt.weight_sync.gpu_delta_memory import require_de_capable

        if (
            host_input.device.type != "cpu"
            or host_input.dtype != torch.uint8
            or not host_input.is_contiguous()
        ):
            raise ValueError("nvCOMP input must be a contiguous host arena byte view")
        if (
            workspace.temporary.device != self.device
            or workspace.temporary.dtype != torch.uint8
            or not workspace.temporary.is_contiguous()
        ):
            raise ValueError(
                "nvCOMP workspace must be contiguous bytes on the decoder device"
            )
        if stream.device != self.device:
            raise ValueError("Decoder stream/device mismatch")
        maximum_count = max(map(len, batches), default=0)
        for tensor, dtype in (
            (workspace.statuses, torch.int32),
            (workspace.actual_sizes, torch.int64),
        ):
            if (
                tensor.device != self.device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
                or tensor.ndim != 2
                or tensor.shape[0] not in (2, 3, 4)
                or tensor.shape[1] < maximum_count
            ):
                raise ValueError("Invalid per-slot status/actual-size capacity")
        slot_count = workspace.statuses.shape[0]
        if workspace.actual_sizes.shape[0] != slot_count:
            raise ValueError("Status and actual-size slot counts differ")
        if (
            workspace.temporary.numel()
            and workspace.temporary.data_ptr() % self.alignments.temp
        ):
            raise ValueError("Misaligned decoder workspace")
        input_base, input_bytes = host_input.data_ptr(), host_input.numel()
        output_bounds, output_remainders = [0] * slot_count, [None] * slot_count
        for index, frames in enumerate(batches):
            slot = index % slot_count
            prior_output_end = 0
            for frame in frames:
                # HARDWARE does not check oversized buffers in nvCOMP. Check
                # actual lengths, not the compressor's worst-case allocation:
                # a compressible 4 MiB frame can fit a 4 MiB device limit.
                if (
                    not 0 < frame.decoded_bytes <= _MAX_FRAME_BYTES
                    or frame.decoded_bytes > self.maximum_chunk_bytes
                    or not 0 < frame.encoded_bytes <= self.maximum_chunk_bytes
                ):
                    raise ValueError(
                        "GPU-delta frames require positive lengths, <=4 MiB "
                        f"output and both lengths <= device limit {self.maximum_chunk_bytes}"
                    )
                if not 0 <= frame.input_offset <= input_bytes - frame.encoded_bytes:
                    raise ValueError("Encoded frame outside input allocation")
                if frame.output_offset < prior_output_end:
                    raise ValueError("Overlapping or unordered decoded frames")
                if (input_base + frame.input_offset) % self.alignments.input:
                    raise ValueError("Misaligned encoded frame")
                remainder = frame.output_offset % self.alignments.output
                if output_remainders[slot] is None:
                    output_remainders[slot] = remainder
                elif remainder != output_remainders[slot]:
                    raise ValueError("Misaligned decoded frame")
                prior_output_end = frame.output_offset + frame.decoded_bytes
            output_bounds[slot] = max(output_bounds[slot], prior_output_end)
        all_frames = [frame for frames in batches for frame in frames]
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            host = torch.empty(
                (4, len(all_frames)), dtype=torch.int64, device="cpu", pin_memory=True
            )
            host[:3].numpy()[:] = [
                [input_base + f.input_offset for f in all_frames],
                [f.encoded_bytes for f in all_frames],
                [f.decoded_bytes for f in all_frames],
            ]
            metadata = torch.empty(host.shape, dtype=torch.int64, device=self.device)
            if metadata.numel():
                require_de_capable(metadata.data_ptr())
                metadata[:3].copy_(host[:3], non_blocking=True)
        return PreparedDecodePlan(
            self,
            batches,
            host_input,
            workspace,
            host,
            metadata,
            stream,
            np.fromiter((frame.output_offset for frame in all_frames), dtype=np.int64),
            output_bounds,
            output_remainders,
        )


class PreparedDecodePlan:
    """Publication metadata and output leases, bound only while paused."""

    def __init__(
        self,
        decoder,
        batches,
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
        self.host_metadata, self.metadata = host_metadata, metadata
        self.output_bounds, self.output_remainders = output_bounds, output_remainders
        self.slot_count = len(output_bounds)
        # Every batch retains this same list. Binding fills it only after all
        # output checks pass; dropping the plan alone cannot free in-flight slots.
        self.decoded_slots = []
        self.batches = []
        offset = 0
        temporary_pointer, temporary_bytes = (
            workspace.temporary.data_ptr(),
            workspace.temporary.numel(),
        )
        for index, frames in enumerate(batches):
            count, slot = len(frames), index % self.slot_count
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
                temporary_pointer,
                temporary_bytes,
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

    def bind_outputs(
        self, decoded_slots: Sequence[torch.Tensor]
    ) -> list[PreparedDecode]:
        """Check paused allocations, fill output pointers, upload just row 3.

        Slot bounds and relative pointer alignment were computed during prepare.
        The caller waits for prior apply readers before reusing each output and
        its status/actual-size row; DE uses one stream and temporary workspace.
        """
        from sglang.srt.weight_sync.gpu_delta_memory import require_de_capable

        if len(decoded_slots) != self.slot_count:
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
        ranges = sorted(
            (base, base + size) for base, size in zip(bases, capacities) if size
        )
        if any(
            start < previous_end
            for (_, previous_end), (start, _) in zip(ranges, ranges[1:])
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
        with torch.cuda.device(self.decoder.device), torch.cuda.stream(self.stream):
            for base, capacity in zip(bases, capacities):
                if capacity:
                    require_de_capable(base)
            for index, batch in enumerate(self.batches):
                np.add(
                    batch.output_offsets,
                    bases[index % self.slot_count],
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

    def enqueue(self) -> None:
        """Launch only; caller checks statuses/sizes on device before applying weights.

        This object and all storage must outlive the submitted work. There is no
        implicit synchronization in destruction; the session owns the final fence
        and enters the captured DE stream/device context around submission.
        """
        if not self.statuses.numel():
            return
        self.decoder._check(self.decoder._decode(*self._arguments))
