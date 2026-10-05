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
from collections.abc import Sequence
from dataclasses import dataclass

import torch


class _SnappyOptions(ctypes.Structure):
    _fields_ = [
        ("backend", ctypes.c_int),
        ("sort_before_hw_decompress", ctypes.c_int),
        ("reserved", ctypes.c_char * 56),
    ]


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
        raise RuntimeError("Snappy deltas require native CUDA allocations")
    snapshot = torch.cuda.memory._snapshot()
    expandable = snapshot.get("allocator_settings", {}).get("expandable_segments")
    if expandable is not False:
        raise RuntimeError(
            "Snappy hardware decoding requires verifiable expandable_segments:False; "
            "PyTorch expandable VMM buffers are not admitted"
        )
    if any(
        segment.get("device") == device.index and segment.get("is_expandable", False)
        for segment in snapshot.get("segments", [])
    ):
        raise RuntimeError(
            "Snappy hardware decoding cannot reuse existing PyTorch expandable VMM segments; "
            "start the process with expandable_segments:False"
        )


class NvcompDecoder:
    """One device's qualified Snappy hardware decoder.

    Missing libraries and unsupported hardware fail admission. There is no CPU
    decoder, algorithm substitution or software Snappy fallback.
    """

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("Delta decoder requires an explicit CUDA device")
        cuda_major = torch.version.cuda.split(".")[0]
        package = f"nvidia-libnvcomp-cu{cuda_major}"
        distribution = importlib.metadata.distribution(package)
        version = tuple(int(v) for v in distribution.version.split(".")[:2])
        if not (5, 3) <= version < (6, 0) or ctypes.sizeof(ctypes.c_size_t) != 8:
            raise RuntimeError("Direct GPU deltas require the 64-bit nvCOMP 5.3+ ABI")
        self.version = distribution.version
        self.backend = "hardware"
        self._options = _SnappyOptions()
        # Explicit backend selection: DEFAULT can silently select software.
        self._options.backend = 1
        if torch.cuda.get_device_capability(self.device)[0] < 10:
            raise RuntimeError("Snappy deltas require Blackwell hardware decompression")
        _require_hardware_allocator(self.device)
        self._library = ctypes.CDLL(
            str(distribution.locate_file("nvidia/libnvcomp/lib64/libnvcomp.so.5"))
        )
        pointer, size = ctypes.c_void_p, ctypes.c_size_t
        self._temporary = self._bind(
            "GetTempSizeAsync", [size, size, _SnappyOptions, ctypes.POINTER(size), size]
        )
        self._align = self._bind(
            "GetRequiredAlignments", [_SnappyOptions, ctypes.POINTER(_Alignments)]
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
                _SnappyOptions,
                pointer,
                pointer,
            ],
        )
        self.alignments = _Alignments()
        self._check(self._align(self._options, ctypes.byref(self.alignments)))
        self._temporary_sizes: dict[tuple[int, int, int], int] = {}

    def _bind(self, suffix, arguments):
        function = getattr(
            self._library,
            "nvcompBatchedSnappyDecompress" + suffix,
        )
        function.argtypes, function.restype = arguments, ctypes.c_int
        return function

    def _check(self, status):
        if status != 0:
            raise RuntimeError(f"nvCOMP snappy/{self.backend} failed: status={status}")

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
        self, batches: Sequence[Sequence[DecodeFrame]]
    ) -> DecodeWorkspace:
        """Allocate small workspace during prepare; retain the decoder to cache queries.

        DE submissions use one stream and share temporary storage. Status and
        actual-size rows belong to the two decoded slots until apply consumes them.
        """
        from sglang.srt.weight_sync.gpu_delta_memory import require_de_capable

        maximum_count = max((len(batch) for batch in batches), default=0)
        temporary = max((self.temporary_bytes(batch) for batch in batches), default=0)
        with torch.cuda.device(self.device):
            workspace = DecodeWorkspace(
                torch.empty(temporary, dtype=torch.uint8, device=self.device),
                torch.empty((2, maximum_count), dtype=torch.int64, device=self.device),
                torch.empty((2, maximum_count), dtype=torch.int32, device=self.device),
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
                or tensor.shape[0] != 2
                or tensor.shape[1] < maximum_count
            ):
                raise ValueError("Invalid per-slot status/actual-size capacity")
        if (
            workspace.temporary.numel()
            and workspace.temporary.data_ptr() % self.alignments.temp
        ):
            raise ValueError("Misaligned decoder workspace")
        input_base, input_bytes = host_input.data_ptr(), host_input.numel()
        for frames in batches:
            prior_output_end = 0
            for frame in frames:
                if not 0 < frame.decoded_bytes <= 1 << 20 or frame.encoded_bytes <= 0:
                    raise ValueError(
                        "Direct-delta frames require positive lengths and <=1 MiB output"
                    )
                if not 0 <= frame.input_offset <= input_bytes - frame.encoded_bytes:
                    raise ValueError("Encoded frame outside input allocation")
                if frame.output_offset < prior_output_end:
                    raise ValueError("Overlapping or unordered decoded frames")
                if (input_base + frame.input_offset) % self.alignments.input:
                    raise ValueError("Misaligned encoded frame")
                prior_output_end = frame.output_offset + frame.decoded_bytes
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
            self, batches, host_input, workspace, host, metadata, stream
        )


@dataclass
class PreparedDecodePlan:
    decoder: NvcompDecoder
    batches: Sequence[Sequence[DecodeFrame]]
    host_input: torch.Tensor
    workspace: DecodeWorkspace
    host_metadata: torch.Tensor
    metadata: torch.Tensor
    stream: torch.cuda.Stream

    def bind_outputs(
        self, decoded_slots: Sequence[torch.Tensor]
    ) -> list[PreparedDecode]:
        """Bind two paused output allocations and upload only their pointer row.

        Batch i uses output slot i % 2 and the matching status/actual-size row.
        The caller waits for prior apply readers before reusing a slot; all DE
        submissions share the captured stream and one temporary workspace.
        """
        from sglang.srt.weight_sync.gpu_delta_memory import require_de_capable

        if len(decoded_slots) != 2:
            raise ValueError("nvCOMP requires two decoded output slots")
        for tensor in decoded_slots:
            if (
                tensor.device != self.decoder.device
                or tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "nvCOMP outputs must be contiguous bytes on the decoder device"
                )
        first, second = decoded_slots
        if max(first.data_ptr(), second.data_ptr()) < min(
            first.data_ptr() + first.numel(), second.data_ptr() + second.numel()
        ):
            raise ValueError("Decoded output slots must not overlap")
        output_pointers = []
        for index, frames in enumerate(self.batches):
            decoded = decoded_slots[index % 2]
            for frame in frames:
                if frame.output_offset + frame.decoded_bytes > decoded.numel():
                    raise ValueError("Out-of-bounds decoded frame")
                pointer = decoded.data_ptr() + frame.output_offset
                if pointer % self.decoder.alignments.output:
                    raise ValueError("Misaligned decoded frame")
                output_pointers.append(pointer)
        with torch.cuda.device(self.decoder.device), torch.cuda.stream(self.stream):
            for tensor in decoded_slots:
                if tensor.numel():
                    require_de_capable(tensor.data_ptr())
            if output_pointers:
                self.host_metadata[3].numpy()[:] = output_pointers
                self.metadata[3].copy_(self.host_metadata[3], non_blocking=True)
        plans, offset = [], 0
        for index, frames in enumerate(self.batches):
            count = len(frames)
            rows = self.metadata[:, offset : offset + count]
            slot = index % 2
            statuses = self.workspace.statuses[slot, :count]
            actual_sizes = self.workspace.actual_sizes[slot, :count]
            expected_sizes = rows[2]
            arguments = (
                rows[0].data_ptr(),
                rows[1].data_ptr(),
                expected_sizes.data_ptr(),
                actual_sizes.data_ptr(),
                count,
                self.workspace.temporary.data_ptr(),
                self.workspace.temporary.numel(),
                rows[3].data_ptr(),
                self.decoder._options,
                statuses.data_ptr(),
                self.stream.cuda_stream,
            )
            plans.append(
                PreparedDecode(
                    self.decoder,
                    self.host_input,
                    decoded_slots[slot],
                    self.workspace,
                    self.host_metadata[:, offset : offset + count],
                    rows,
                    self.stream,
                    statuses,
                    actual_sizes,
                    expected_sizes,
                    arguments,
                )
            )
            offset += count
        return plans


@dataclass
class PreparedDecode:
    decoder: NvcompDecoder
    host_input: torch.Tensor
    decoded: torch.Tensor
    workspace: DecodeWorkspace
    host_metadata: torch.Tensor
    metadata: torch.Tensor
    stream: torch.cuda.Stream
    statuses: torch.Tensor
    actual_sizes: torch.Tensor
    expected_sizes: torch.Tensor
    _arguments: tuple

    def enqueue(self) -> None:
        """Launch only; caller checks statuses/sizes on device before applying weights.

        This object and all storage must outlive the submitted work. There is no
        implicit synchronization in destruction; the session owns the final fence
        and enters the captured DE stream/device context around submission.
        """
        if not self.statuses.numel():
            return
        self.decoder._check(self.decoder._decode(*self._arguments))
