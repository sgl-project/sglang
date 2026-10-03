"""Caller-owned, asynchronous nvCOMP decoding for direct weight deltas.

Metadata, workspace and output allocations are prepared before pausing generation.
The hot path calls the public C API directly: no nvCOMP Python array finalizers,
implicit size discovery, host scalar reads or device-wide synchronization.
"""

from __future__ import annotations

import ctypes
import importlib.metadata
from dataclasses import dataclass
from typing import Sequence

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
            "CUDA VMM buffers are not admitted"
        )
    if any(
        segment.get("device") == device.index and segment.get("is_expandable", False)
        for segment in snapshot.get("segments", [])
    ):
        raise RuntimeError(
            "Snappy hardware decoding cannot reuse existing CUDA VMM segments; "
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
        size = ctypes.c_size_t()
        with torch.cuda.device(self.device):
            self._check(
                self._temporary(
                    len(frames),
                    max(f.decoded_bytes for f in frames),
                    self._options,
                    ctypes.byref(size),
                    sum(f.decoded_bytes for f in frames),
                )
            )
        return size.value

    def allocate_workspace(
        self, batches: Sequence[Sequence[DecodeFrame]]
    ) -> DecodeWorkspace:
        """Reserve one reusable arena for sequential tensor decoding during preparation."""
        maximum_count = max((len(batch) for batch in batches), default=0)
        temporary = max((self.temporary_bytes(batch) for batch in batches), default=0)
        with torch.cuda.device(self.device):
            return DecodeWorkspace(
                torch.empty(temporary, dtype=torch.uint8, device=self.device),
                torch.empty(maximum_count, dtype=torch.int64, device=self.device),
                torch.empty(maximum_count, dtype=torch.int32, device=self.device),
            )

    def prepare(
        self,
        frames: Sequence[DecodeFrame],
        encoded: torch.Tensor,
        decoded: torch.Tensor,
        workspace: DecodeWorkspace,
        stream: torch.cuda.Stream,
    ) -> PreparedDecode:
        """Freeze pointers/lengths and upload small metadata before the serving pause.

        A plan can reuse an encoded tensor slot and decoded scratch. Its caller
        must order slot reuse after all previous consumers on the chosen stream.
        ``workspace`` must be allocated for these batches with ``allocate_workspace``.
        """
        frames = tuple(frames)
        for tensor in (encoded, decoded, workspace.temporary):
            if (
                tensor.device != self.device
                or tensor.dtype != torch.uint8
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "nvCOMP buffers must be contiguous bytes on the decoder device"
                )
        if stream.device != self.device:
            raise ValueError("Decoder stream/device mismatch")
        if (
            len(frames) > workspace.statuses.numel()
            or len(frames) > workspace.actual_sizes.numel()
        ):
            raise ValueError("Insufficient per-frame status capacity")
        if (
            workspace.temporary.numel()
            and workspace.temporary.data_ptr() % self.alignments.temp
        ):
            raise ValueError("Misaligned decoder workspace")
        prior_output_end = 0
        for frame in frames:
            if not 0 < frame.decoded_bytes <= 1 << 20 or frame.encoded_bytes <= 0:
                raise ValueError(
                    "Direct-delta frames require positive lengths and <=1 MiB output"
                )
            if not 0 <= frame.input_offset <= encoded.numel() - frame.encoded_bytes:
                raise ValueError("Encoded frame outside input allocation")
            if (
                not prior_output_end
                <= frame.output_offset
                <= decoded.numel() - frame.decoded_bytes
            ):
                raise ValueError(
                    "Overlapping, unordered or out-of-bounds decoded frames"
                )
            if (encoded.data_ptr() + frame.input_offset) % self.alignments.input:
                raise ValueError("Misaligned encoded frame")
            if (decoded.data_ptr() + frame.output_offset) % self.alignments.output:
                raise ValueError("Misaligned decoded frame")
            prior_output_end = frame.output_offset + frame.decoded_bytes
        with torch.cuda.device(self.device), torch.cuda.stream(stream):
            host = torch.empty(
                (4, len(frames)), dtype=torch.int64, device="cpu", pin_memory=True
            )
            host.numpy()[:] = [
                [encoded.data_ptr() + f.input_offset for f in frames],
                [f.encoded_bytes for f in frames],
                [f.decoded_bytes for f in frames],
                [decoded.data_ptr() + f.output_offset for f in frames],
            ]
            metadata = host.to(self.device, non_blocking=True)
            ready = torch.cuda.Event()
            ready.record(stream)
        return PreparedDecode(
            self, frames, encoded, decoded, workspace, host, metadata, ready
        )


@dataclass
class PreparedDecode:
    decoder: NvcompDecoder
    frames: tuple[DecodeFrame, ...]
    encoded: torch.Tensor
    decoded: torch.Tensor
    workspace: DecodeWorkspace
    host_metadata: torch.Tensor
    metadata: torch.Tensor
    ready: torch.cuda.Event

    @property
    def statuses(self):
        return self.workspace.statuses[: len(self.frames)]

    @property
    def actual_sizes(self):
        return self.workspace.actual_sizes[: len(self.frames)]

    def enqueue(self, stream: torch.cuda.Stream) -> None:
        """Launch only; caller checks statuses/sizes on device before applying weights.

        This object and all storage must outlive the submitted work. There is no
        implicit synchronization in destruction; the session owns the final fence.
        """
        if stream.device != self.decoder.device:
            raise ValueError("Decoder stream/device mismatch")
        if not self.frames:
            return
        stream.wait_event(self.ready)
        metadata, workspace = self.metadata, self.workspace
        with torch.cuda.device(self.decoder.device):
            self.decoder._check(
                self.decoder._decode(
                    metadata[0].data_ptr(),
                    metadata[1].data_ptr(),
                    metadata[2].data_ptr(),
                    workspace.actual_sizes.data_ptr(),
                    len(self.frames),
                    workspace.temporary.data_ptr(),
                    workspace.temporary.numel(),
                    metadata[3].data_ptr(),
                    self.decoder._options,
                    workspace.statuses.data_ptr(),
                    stream.cuda_stream,
                )
            )
