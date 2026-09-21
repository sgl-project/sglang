"""Thread-local GPU transforms; no request, cache or transport ownership.

Methods finish their CUDA work before returning. Call them only from transfer
or restoration workers, never from scheduler polling. This deliberately simple
prototype does not claim fully asynchronous nvCOMP Python API submission.
"""

from __future__ import annotations

import importlib.metadata
import threading

from sglang.srt.kv_compression.types import (
    NVCOMP_VERSION,
    BufferDrainError,
)


class NvcompLZ4Backend:
    def __init__(self, device, stream):
        import torch
        from nvidia import nvcomp

        cuda_major = torch.version.cuda.split(".")[0]
        installed = importlib.metadata.version(f"nvidia-nvcomp-cu{cuda_major}")
        if installed != NVCOMP_VERSION:
            raise RuntimeError(
                f"P/D prototype requires nvCOMP {NVCOMP_VERSION}, got {installed}"
            )
        self.nvcomp = nvcomp
        self.stream = stream
        self.thread_id = threading.get_ident()
        self.impl = nvcomp.Codec(
            algorithm="LZ4",
            device_id=device.index,
            cuda_stream=stream.cuda_stream,
            uncomp_chunk_size=65536,
            bitstream_kind=nvcomp.BitstreamKind.NVCOMP_NATIVE,
        )

    def _array(self, tensor):
        import torch

        if threading.get_ident() != self.thread_id:
            raise RuntimeError("An nvCOMP instance must stay on its creating thread")
        if (
            tensor.dtype != torch.uint8
            or not tensor.is_cuda
            or not tensor.is_contiguous()
        ):
            raise ValueError("Compression requires contiguous CUDA uint8 buffers")
        return self.nvcomp.as_array(tensor, cuda_stream=self.stream.cuda_stream)

    def max_output_bytes(self, src) -> int:
        # The 5.3 release note and Python reference use different spellings.
        bound = getattr(self.impl, "get_max_comp_buffer_size", None)
        if bound is None:
            bound = getattr(self.impl, "get_max_compressed_buffer_size")
        return int(bound(self._array(src)))

    def drain(self):
        try:
            self.stream.synchronize()
        except Exception as exc:
            raise BufferDrainError("CUDA stream did not drain") from exc

    def compress_into(self, src, dst) -> int:
        if dst.numel() < self.max_output_bytes(src):
            raise ValueError("Compression output capacity is too small")
        try:
            result = self.impl.encode(self._array(src), out=self._array(dst))
            self.stream.synchronize()
            length = int(result.buffer_size)
            if not 0 < length <= dst.numel():
                raise ValueError("nvCOMP returned an invalid encoded length")
            return length
        finally:
            # Keep Python wrappers and borrowed buffers alive even on failure.
            self.drain()

    def decompress_into(self, src, dst) -> None:
        try:
            source = self._array(src)
            expected = int(self.impl.get_uncomp_buffer_size(source))
            if expected != dst.numel():
                raise ValueError("nvCOMP header disagrees with destination length")
            self.impl.decode(source, out=self._array(dst))
        finally:
            self.drain()

    def compress_batch(self, sources, outputs):
        if len(sources) != len(outputs) or not sources:
            raise ValueError("Invalid compression batch")
        try:
            result = self.impl.encode(
                [self._array(t) for t in sources],
                out=[self._array(t) for t in outputs],
            )
            self.drain()
            lengths = [int(t.buffer_size) for t in result]
            if len(lengths) != len(outputs) or any(
                n <= 0 or n > dst.numel() for n, dst in zip(lengths, outputs)
            ):
                raise ValueError("Invalid encoded lengths")
            return lengths
        finally:
            self.drain()

    def decompress_batch(self, sources, outputs):
        if len(sources) != len(outputs) or not sources:
            raise ValueError("Invalid decompression batch")
        try:
            arrays = [self._array(t) for t in sources]
            if any(
                int(self.impl.get_uncomp_buffer_size(src)) != dst.numel()
                for src, dst in zip(arrays, outputs)
            ):
                raise ValueError("Encoded page disagrees with its declared size")
            self.impl.decode(arrays, out=[self._array(t) for t in outputs])
        finally:
            self.drain()
