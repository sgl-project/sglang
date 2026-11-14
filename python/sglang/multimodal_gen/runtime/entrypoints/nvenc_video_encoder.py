# SPDX-License-Identifier: Apache-2.0
"""NVENC MP4 saving with reusable per-configuration sessions."""

from __future__ import annotations

import io
import logging
import math
import os
import threading
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

import torch

logger = logging.getLogger(__name__)
_MAX_IDLE_SESSIONS = 4
_idle_sessions: OrderedDict[tuple[int, int, int, int, int], _NvencSession] = (
    OrderedDict()
)
_cache_lock = threading.Lock()


@lru_cache(maxsize=None)
def _device_has_nvenc(pid: int, gpu_id: int) -> bool:
    try:
        import PyNvVideoCodec as nvc

        caps = nvc.GetEncoderCaps(gpu_id, "h264")
        supported = int(caps.get("num_encoder_engines", 0)) > 0
    except Exception as exc:
        message = str(exc).splitlines()
        logger.warning(
            "Cannot query NVENC on CUDA device %d; using CPU encoding: %s",
            gpu_id,
            message[0] if message else type(exc).__name__,
        )
        return False
    if not supported:
        logger.warning("CUDA device %d has no H.264 NVENC; using CPU encoding", gpu_id)
    return supported


def supports_nvenc(device: torch.device) -> bool:
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    gpu_id = device.index if device.index is not None else torch.cuda.current_device()
    return _device_has_nvenc(os.getpid(), gpu_id)


class _NvencSession:
    def __init__(self, key: tuple[int, int, int, int, int]):
        import PyNvVideoCodec as nvc

        self.key = key
        self.nvc = nvc
        gpu_id, width, height, fps, cq = key
        with torch.cuda.device(gpu_id):
            self.stream = torch.cuda.Stream()
            self.encoder = nvc.CreateEncoder(
                width,
                height,
                "ABGR",
                False,
                gpu_id=gpu_id,
                cudastream=self.stream.cuda_stream,
                codec="h264",
                preset="P4",
                tuning_info="high_quality",
                rc="vbr",
                cq=cq,
                bf=0,
                fps=fps,
                gop=250,
            )
        self.frame_count = 0


def _acquire_session(key: tuple[int, int, int, int, int]):
    with _cache_lock:
        session = _idle_sessions.pop(key, None)
    if session is not None:
        return session, True
    return _NvencSession(key), False


def _release_session(session: _NvencSession) -> None:
    with _cache_lock:
        # Concurrent videos with the same settings may each create a session.
        # Keep one idle session per key and evict the least recently used key.
        if session.key in _idle_sessions:
            return
        _idle_sessions[session.key] = session
        if len(_idle_sessions) > _MAX_IDLE_SESSIONS:
            _idle_sessions.popitem(last=False)


class NvencVideoEncoder:
    """Save one complete CUDA RGB video using an idle NVENC session if available."""

    @classmethod
    def open(
        cls,
        save_file_path: str,
        *,
        device: torch.device,
        height: int,
        width: int,
        fps: int,
        num_frames: int,
        element_size: int = 4,
        audio: Any = None,
        audio_sample_rate: int | None = None,
        output_compression: int | None = None,
        x264_preset: str | None = None,
        max_queued_frames: int = 0,
    ) -> NvencVideoEncoder | None:
        if (
            device.type != "cuda"
            or os.path.splitext(save_file_path)[1].lower() != ".mp4"
            or audio is not None
            or x264_preset is not None
        ):
            return None
        quality = output_compression / 10 if output_compression is not None else 5
        if not 1 <= quality <= 10:
            return None
        cq = int((1 - quality / 10) * 51)
        if cq == 0:  # NVENC CQ=0 is automatic quality, not x264 lossless CRF=0.
            return None
        if not supports_nvenc(device):
            return None
        gpu_id = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        key = (gpu_id, width, height, fps, cq)
        session, cache_hit = _acquire_session(key)
        try:
            return cls(
                save_file_path,
                session,
                cache_hit=cache_hit,
                max_queued_frames=max_queued_frames,
            )
        except BaseException:
            try:
                os.unlink(save_file_path)
            except FileNotFoundError:
                pass
            raise

    def __init__(
        self,
        save_file_path: str,
        session: _NvencSession,
        *,
        cache_hit: bool,
        max_queued_frames: int = 0,
    ):
        import av

        self.save_file_path = save_file_path
        self.cache_hit = cache_hit
        self._session = session
        self._finished = False
        self._frame_index = 0
        self._first_packet_timestamp = None
        gpu_id, width, height, fps, _ = session.key
        self._device = torch.device("cuda", gpu_id)
        self._width = width
        self._height = height
        headers = session.encoder.GetSequenceParams()
        with av.open(io.BytesIO(headers), format="h264") as source:
            self._bitstream_filter = av.BitStreamFilterContext(
                "h264_metadata=video_full_range_flag=0:"
                "colour_primaries=1:transfer_characteristics=1:"
                "matrix_coefficients=1",
                source.streams.video[0],
            )
        headers = bytes(self._bitstream_filter.filter(av.Packet(headers))[0])
        timebase = math.lcm(90000, fps)
        nvc = session.nvc
        self._muxer = nvc.FFmpegMuxer(
            file_path=save_file_path,
            media_format=nvc.GetMediaFormat(save_file_path),
            codec="h264",
            width=width,
            height=height,
            fps_num=fps,
            fps_den=1,
            timebase_num=1,
            timebase_den=timebase,
            extradata=headers,
        )
        self._muxer.SetUniformPtsIncrement(timebase // fps)
        self._max_queued_frames = max_queued_frames
        self._pending = deque()
        self._queued_frames = 0
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="nvenc-video")
            if max_queued_frames > 0
            else None
        )

    def write(self, frames: torch.Tensor) -> None:
        """Submit [3, T, H, W] CUDA RGB frames, either uint8 or in [0, 1]."""
        if self._finished:
            raise RuntimeError("NVENC video encoder is closed")
        if frames.dim() == 3:
            frames = frames.unsqueeze(1)
        if (
            frames.dim() != 4
            or frames.device != self._device
            or int(frames.shape[0]) != 3
            or tuple(frames.shape[-2:]) != (self._height, self._width)
        ):
            raise ValueError(
                f"expected [3, T, {self._height}, {self._width}] on "
                f"{self._device}, got {tuple(frames.shape)} on {frames.device}"
            )
        if self._executor is None:
            self._write_frames(frames)
            return

        # Capture the producer's exact point of completion before the VAE
        # starts its next chunk. The worker must not wait on a later stream tail.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(self._device))
        count = int(frames.shape[1])
        while self._pending and self._queued_frames + count > self._max_queued_frames:
            future, completed_count = self._pending.popleft()
            self._queued_frames -= completed_count
            future.result()
        self._pending.append(
            (self._executor.submit(self._write_frames, frames, ready), count)
        )
        self._queued_frames += count

    def _write_frames(
        self, frames: torch.Tensor, ready: torch.cuda.Event | None = None
    ) -> None:
        session = self._session
        stream = session.stream
        if ready is None:
            stream.wait_stream(torch.cuda.current_stream(self._device))
        else:
            stream.wait_event(ready)
        frames.record_stream(stream)
        frame_bytes = 3 * self._height * self._width * frames.element_size()
        chunk_frames = max(1, min(4, (64 * 1024 * 1024) // frame_bytes))
        with torch.cuda.device(self._device), torch.cuda.stream(stream):
            for chunk in frames.split(chunk_frames, dim=1):
                rgb = (
                    chunk
                    if chunk.dtype == torch.uint8
                    else (chunk * 255).clamp_(0, 255).to(torch.uint8)
                )
                rgb = rgb.permute(1, 2, 3, 0)
                rgba = torch.empty(
                    (*rgb.shape[:-1], 4), dtype=torch.uint8, device=self._device
                )
                rgba[..., :3].copy_(rgb)
                rgba[..., 3].fill_(255)
                for frame in rgba:
                    picture = session.nvc.NV_ENC_PIC_PARAMS()
                    picture.inputTimeStamp = session.frame_count
                    if self._frame_index == 0 and session.frame_count:
                        picture.encodePicFlags = session.nvc.NV_ENC_PIC_FLAGS.FORCEIDR
                    self._mux_packets(session.encoder.Encode(frame, picture))
                    session.frame_count += 1
                    self._frame_index += 1
                rgba.record_stream(stream)

    def _mux_packets(self, packets) -> None:
        import av

        for packet in packets:
            if self._first_packet_timestamp is None:
                self._first_packet_timestamp = packet["timestamp"]
            for filtered in self._bitstream_filter.filter(av.Packet(packet["data"])):
                self._muxer.MuxVideoPacket(
                    bytes(filtered),
                    packet["picture_type"],
                    packet["timestamp"] - self._first_packet_timestamp,
                )

    def close(self) -> None:
        if self._finished:
            return
        try:
            if self._executor is None:
                self._finish_encode()
            else:
                while self._pending:
                    future, count = self._pending.popleft()
                    self._queued_frames -= count
                    future.result()
                self._executor.submit(self._finish_encode).result()
        except BaseException:
            self.abort()
            raise
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._executor = None

    def _finish_encode(self) -> None:
        self._mux_packets(self._session.encoder.EndEncode())
        self._muxer.Finalize()
        self._session.stream.synchronize()
        self._finished = True
        self._muxer = None
        session, self._session = self._session, None
        _release_session(session)

    def abort(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._queued_frames = 0
        if self._finished:
            return
        self._finished = True
        # An interrupted encoding leaves unknown buffered frames; discard it.
        self._muxer = None
        session, self._session = self._session, None
        if session is not None:
            try:
                session.stream.synchronize()
            except Exception:
                pass
        try:
            os.unlink(self.save_file_path)
        except FileNotFoundError:
            pass
