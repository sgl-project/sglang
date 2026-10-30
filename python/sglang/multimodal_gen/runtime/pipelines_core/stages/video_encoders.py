# SPDX-License-Identifier: Apache-2.0
"""Registry for the video encoders that consume frames while a decode is running.

:class:`EncodeWhileDecoding` forwards each run of finished frames to an encoder
resolved from this registry:

* :class:`VideoEncoder` is the protocol an encoder implements.
* :func:`register_video_encoder` adds a factory under a name.
* entry points published in the ``sglang.video_encoders`` group are registered on
  first lookup, so a package can supply an encoder without patching SGLang.
* :func:`set_default_video_encoder` selects the encoder used when a caller does
  not name one; :data:`DEFAULT_VIDEO_ENCODER` is the built-in ffmpeg encoder.

A factory receives the keyword arguments the stage has (``save_file_path``,
``device``, ``height``, ``width``, ``fps``, ``num_frames``, and the audio and
compression settings) and returns an encoder, or ``None`` when the output cannot
use that path, in which case the stage saves the frames after the decode instead.
"""

from __future__ import annotations

import threading
from importlib.metadata import entry_points
from typing import Any, Callable, Optional, Protocol, runtime_checkable

import torch

#: Entry-point group discovery looks at.
VIDEO_ENCODER_ENTRY_POINT_GROUP = "sglang.video_encoders"

#: Name used when the caller does not ask for a specific encoder. The built-in
#: encoder shells out to ffmpeg (libx264) and stages frames through CUDA-registered
#: host memory, so the name describes the process, not a hardware encoder.
DEFAULT_VIDEO_ENCODER = "ffmpeg"


@runtime_checkable
class VideoEncoder(Protocol):
    """Encodes the frames a decode produces, as it produces them.

    ``write`` is called once per run of finished frames, so one clip arrives as a
    sequence of calls whose ``T`` varies; concatenating the arguments in call
    order reproduces the clip in decode order. Each call carries a ``[3, T, H, W]``
    float tensor in ``[0, 1]`` on the device that produced it, and the producer
    does not synchronise first: an implementation that copies to the host must do
    so on the producing device's current stream.

    One encoder serves one clip. A caller encoding several clips creates one
    encoder per clip.

    ``close`` returns the path written, or ``None`` when the encoder wrote the
    path it was opened with. The stage reports exactly one path per clip, so an
    encoder that produces several files must return the one that represents the
    clip.
    """

    def write(self, frames: torch.Tensor) -> None:
        """Queue frames for encoding. Must not block the producing stream."""

    def close(self) -> str | None:
        """Finish writing.

        Return the path actually written when it differs from the one the caller
        asked for, or ``None`` to mean "the path you passed to the factory is the
        output" (what the built-in encoder does). One path per request: the
        pipeline requires ``len(output_file_paths) == len(requests)``.
        """

    def abort(self) -> None:
        """Discard any partial output. Safe to call after ``close``."""


VideoEncoderFactory = Callable[..., Optional[VideoEncoder]]

_REGISTRY: dict[str, VideoEncoderFactory] = {}
#: Set by ``set_default_video_encoder``; ``None`` means DEFAULT_VIDEO_ENCODER.
_DEFAULT_OVERRIDE: str | None = None
_REGISTRY_LOCK = threading.Lock()
_ENTRY_POINTS_LOADED = False


def _default_factory() -> VideoEncoderFactory:
    def open_default(**kwargs: Any) -> Optional[VideoEncoder]:
        from sglang.multimodal_gen.runtime.entrypoints.utils import CudaVideoEncoder

        return CudaVideoEncoder.open(**kwargs)

    return open_default


def _load_entry_points() -> None:
    """Register encoders advertised by installed packages. Runs once."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        discovered = entry_points(group=VIDEO_ENCODER_ENTRY_POINT_GROUP)
    except Exception:  # pragma: no cover - defensive, older importlib
        return
    for entry_point in discovered:
        try:
            _REGISTRY.setdefault(entry_point.name, entry_point.load())
        except Exception:  # pragma: no cover - a broken plugin must not break import
            continue


def register_video_encoder(
    name: str, factory: VideoEncoderFactory, *, replace: bool = False
) -> None:
    """Install ``factory`` under ``name``.

    ``replace=False`` keeps an already registered encoder, so a plugin cannot
    silently shadow a built-in one.
    """
    if not name:
        raise ValueError("video encoder name must be non-empty")
    with _REGISTRY_LOCK:
        if name in _REGISTRY and not replace:
            raise ValueError(
                f"video encoder {name!r} is already registered; "
                "pass replace=True to override it"
            )
        _REGISTRY[name] = factory


def set_default_video_encoder(name: str | None) -> None:
    """Choose the encoder used when a caller does not name one.

    This is the selection point for embedders: there is no per-request argument,
    because the pipeline writes exactly one file per request. Passing ``None``
    restores the built-in default.
    """
    global _DEFAULT_OVERRIDE
    if name is None:
        _DEFAULT_OVERRIDE = None
        return
    get_video_encoder_factory(name)  # validates, raises for unknown names
    _DEFAULT_OVERRIDE = name


def default_video_encoder() -> str:
    """The name ``open_video_encoder(None, ...)`` resolves to."""
    return _DEFAULT_OVERRIDE or DEFAULT_VIDEO_ENCODER


def available_video_encoders() -> list[str]:
    with _REGISTRY_LOCK:
        _load_entry_points()
        return sorted(_REGISTRY)


def get_video_encoder_factory(name: str | None = None) -> VideoEncoderFactory:
    """Resolve ``name`` (default: :data:`DEFAULT_VIDEO_ENCODER`) to a factory."""
    resolved_name = name or default_video_encoder()
    with _REGISTRY_LOCK:
        _load_entry_points()
        factory = _REGISTRY.get(resolved_name)
    if factory is not None:
        return factory
    if resolved_name == DEFAULT_VIDEO_ENCODER:
        return _default_factory()
    raise ValueError(
        f"unknown video encoder {resolved_name!r}; "
        f"available: {', '.join(available_video_encoders() + [DEFAULT_VIDEO_ENCODER])}"
    )


def open_video_encoder(
    name: str | None = None, **kwargs: Any
) -> Optional[VideoEncoder]:
    """Build the encoder for ``name``, or ``None`` if it cannot handle this output."""
    return get_video_encoder_factory(name)(**kwargs)
