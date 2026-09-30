# SPDX-License-Identifier: Apache-2.0
"""H.264-encode decoded video chunks while later chunks are still decoding.

A decoding stage pushes each finished temporal chunk and hands the encoder to
the worker's direct MP4 save through ``OutputBatch.streamed_video``; the save
then only muxes the audio track. The pushed frames must equal the final output
tensor, or the save falls back to encoding that tensor."""

from __future__ import annotations

import os
import queue
import subprocess
import tempfile
import threading
from typing import Any, Optional

import torch

from sglang.multimodal_gen.runtime.entrypoints import utils as _utils
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class StreamingVideoEncoder:
    """libx264 over a raw rgb24 pipe, fed from pinned host chunks by a thread."""

    def __init__(
        self,
        *,
        save_file_path: str,
        width: int,
        height: int,
        fps: int,
        crf: int,
    ) -> None:
        self.save_file_path = save_file_path
        self.shape = (height, width)
        self.num_frames = 0
        self.video_path = f"{save_file_path}.video.mp4"
        self._error: Optional[BaseException] = None
        self._closed = False
        self._stderr = tempfile.TemporaryFile()
        os.makedirs(os.path.dirname(save_file_path) or ".", exist_ok=True)
        self._process = subprocess.Popen(
            [
                _utils._resolve_ffmpeg_exe(),
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-s",
                f"{width}x{height}",
                "-pix_fmt",
                "rgb24",
                "-r",
                f"{fps:.02f}",
                "-i",
                "pipe:0",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                _utils.X264_PRESET,
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-threads",
                str(_utils._x264_auto_thread_count(height)),
                "-v",
                "warning",
                self.video_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
        )
        self._chunks: queue.Queue = queue.Queue()
        self._writer = threading.Thread(target=self._write_chunks, daemon=True)
        self._writer.start()

    def push(self, frames: torch.Tensor) -> None:
        """Queue float frames [3, T, H, W] in [0, 1] from the current CUDA stream."""
        if self._error is not None:
            return
        if frames.shape[0] != 3 or tuple(frames.shape[-2:]) != self.shape:
            self._error = ValueError(f"unexpected chunk shape {tuple(frames.shape)}")
            return
        # the direct CUDA save converts the final tensor the same way
        rgb = (frames.float() * 255).clamp_(0, 255).to(torch.uint8)
        rgb = rgb.permute(1, 2, 3, 0).contiguous()
        host = torch.empty(rgb.shape, dtype=torch.uint8, pin_memory=True)
        host.copy_(rgb, non_blocking=True)
        copied = torch.cuda.Event()
        copied.record()
        self.num_frames += int(rgb.shape[0])
        self._chunks.put((host, copied))

    def _write_chunks(self) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        while (chunk := self._chunks.get()) is not None:
            if self._error is not None:
                continue
            host, copied = chunk
            try:
                copied.synchronize()
                stdin.write(host.numpy().data)
            except OSError as exc:
                self._error = exc

    def _close_video(self) -> bool:
        self._closed = True
        self._chunks.put(None)
        self._writer.join()
        try:
            self._process.stdin.close()
        except OSError as exc:
            self._error = self._error or exc
        if self._process.wait() and self._error is None:
            self._stderr.seek(0)
            self._error = RuntimeError(self._stderr.read().decode(errors="replace"))
        return self._error is None

    def abort(self) -> None:
        if self._closed:
            return
        self._error = RuntimeError("aborted")
        self._process.kill()
        self._close_video()
        _remove(self.video_path)

    def finish(
        self,
        *,
        save_file_path: str,
        num_frames: int,
        audio: Any,
        audio_sample_rate: Optional[int],
        fps: int,
    ) -> bool:
        """Mux the audio into the streamed video; False leaves no output file."""
        if save_file_path != self.save_file_path or num_frames != self.num_frames:
            self._error = self._error or ValueError(
                f"streamed {self.num_frames} frames for {self.save_file_path}, "
                f"saving {num_frames} to {save_file_path}"
            )
        if not self._close_video():
            logger.warning(
                "Streaming video encode failed (%s); encoding the output.", self._error
            )
            _remove(self.video_path)
            return False
        audio_np = _utils._normalize_audio_to_numpy(audio)
        if audio_np is None:
            os.replace(self.video_path, save_file_path)
            return True
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            _utils.scipy_wavfile.write(
                wav_path,
                _utils._pick_audio_sample_rate(
                    audio_np=audio_np,
                    audio_sample_rate=audio_sample_rate,
                    fps=fps,
                    num_frames=num_frames,
                ),
                audio_np,
            )
            subprocess.run(
                [
                    _utils._resolve_ffmpeg_exe(),
                    "-y",
                    "-i",
                    self.video_path,
                    "-i",
                    wav_path,
                    "-c:v",
                    "copy",
                    "-acodec",
                    "aac",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-v",
                    "warning",
                    save_file_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            logger.warning("Streaming video mux failed (%s); encoding the output.", exc)
            _remove(save_file_path)
            return False
        finally:
            _remove(self.video_path)
            _remove(wav_path)
        return True


def _remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def start_streaming_video_encoder(
    *,
    save_file_path: str,
    width: int,
    height: int,
    fps: int,
    output_compression: Optional[int],
) -> Optional[StreamingVideoEncoder]:
    """An encoder for ``save_file_path``, or None where the direct save would
    not produce an MP4 this encoder can reproduce."""
    crf = _utils._x264_crf(output_compression)
    if (
        crf is None
        or width % 16
        or height % 16
        or _utils.scipy_wavfile is None
        or os.path.splitext(save_file_path)[1].lower() != ".mp4"
    ):
        return None
    try:
        return StreamingVideoEncoder(
            save_file_path=save_file_path,
            width=width,
            height=height,
            fps=fps,
            crf=crf,
        )
    except OSError as exc:
        logger.warning("Streaming video encode unavailable: %s", exc)
        return None
