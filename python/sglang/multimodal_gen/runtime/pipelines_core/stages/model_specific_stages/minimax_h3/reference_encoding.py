# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 ref2va reference-material encoding.

Encoding recipes for user-provided reference materials:

- image reference: independent 2048px short-edge resize with upscale enabled,
  LANCZOS, and nearest-32 dimensions, then the SAME keyframe tokenizer recipe
  as fl2va (seed-42 sampled encode, normalize, [1,2,2] patchify);
- audio reference: the audio material chain (pure
  audio is losslessly normalized
  to stereo; video soundtracks are extracted as 44.1 kHz stereo), then a
  single resample to 32 kHz,
  audio VAE posterior MEAN (encoder -> optional pre_block -> mean_proj; no
  sampling), canonical [2, T, 32], normalize with loader-injected audio stats,
  channel-major rows.
"""

from __future__ import annotations

import functools
import math
import mmap
import os
import socket
import subprocess
import sys
from typing import Any

import torch

from sglang.multimodal_gen.configs.models.vaes.minimax_h3_audio import (
    MiniMaxH3AudioVAEArchConfig,
)
from sglang.multimodal_gen.configs.models.vaes.minimax_h3_video import (
    MiniMaxH3VideoVAEArchConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_world_group
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
    MINIMAX_H3_SUPPORTED_FPS,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.keyframe_encoding import (
    _cached_latent_mean_std,
    minimax_h3_scoped_encode_rng,
)

MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE = 2048
MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE = 32
MINIMAX_H3_AUDIO_SAMPLE_RATE = 32000
MINIMAX_H3_AUDIO_CHANNELS = 2


class _AudioVAEDeterminismContext:
    """Scoped determinism config for the audio encode.

    Disables TF32, forces deterministic algorithms, DISABLES cuDNN entirely
    for the encode (convs run on the fallback kernels), and pins SDP to the
    math backend. This configuration is required for a deterministic encode.
    Everything is restored on exit
    so the decode path keeps its own configuration.

    Reentrant via a shared depth counter: a caller encoding several reference
    materials in one request (audio_encoding.py's per-material loop) can wrap
    the whole loop in one of these: only the outermost enter/exit actually
    touches torch.backends, and each per-material call's own nested
    with-block becomes a no-op increment/decrement instead of redundantly
    saving and restoring the same flags per material.
    """

    _depth = 0
    _saved: tuple | None = None

    def __enter__(self):
        if _AudioVAEDeterminismContext._depth == 0:
            b = torch.backends
            _AudioVAEDeterminismContext._saved = (
                b.cuda.matmul.allow_tf32,
                b.cudnn.allow_tf32,
                b.cudnn.benchmark,
                b.cudnn.deterministic,
                b.cudnn.enabled,
                b.cuda.flash_sdp_enabled(),
                b.cuda.mem_efficient_sdp_enabled(),
                b.cuda.math_sdp_enabled(),
            )
            b.cuda.matmul.allow_tf32 = False
            b.cudnn.allow_tf32 = False
            b.cudnn.benchmark = False
            b.cudnn.deterministic = True
            b.cudnn.enabled = False
            b.cuda.enable_flash_sdp(False)
            b.cuda.enable_mem_efficient_sdp(False)
            b.cuda.enable_math_sdp(True)
        _AudioVAEDeterminismContext._depth += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        _AudioVAEDeterminismContext._depth -= 1
        if _AudioVAEDeterminismContext._depth == 0:
            b = torch.backends
            (
                b.cuda.matmul.allow_tf32,
                b.cudnn.allow_tf32,
                b.cudnn.benchmark,
                b.cudnn.deterministic,
                b.cudnn.enabled,
                flash,
                mem_eff,
                math_sdp,
            ) = _AudioVAEDeterminismContext._saved
            b.cuda.enable_flash_sdp(flash)
            b.cuda.enable_mem_efficient_sdp(mem_eff)
            b.cuda.enable_math_sdp(math_sdp)
            _AudioVAEDeterminismContext._saved = None


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def minimax_h3_resolve_reference_image_shape(
    *,
    width: int | float,
    height: int | float,
) -> dict[str, Any]:
    """Resolve a ref2va image independently from the target canvas.

    The image keeps its display ratio, always targets a 2048px short edge (even
    when that requires upscaling), and rounds both dimensions independently to
    the nearest 32px grid. Unlike target/video ``adapt_shape_v1``, reference
    images have no area-cap branch.
    """

    try:
        source_width = float(width)
        source_height = float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference image width and height must be positive finite numbers"
        ) from exc
    if (
        not math.isfinite(source_width)
        or not math.isfinite(source_height)
        or source_width <= 0.0
        or source_height <= 0.0
    ):
        raise ValueError(
            "reference image width and height must be positive finite numbers"
        )
    if source_width > 4.0 * source_height or source_height > 4.0 * source_width:
        raise ValueError(
            "reference image ratio must be within the inclusive range "
            f"1:4 to 4:1, got {source_width:g}x{source_height:g}"
        )

    scale = MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE / min(source_width, source_height)
    target_width = _nearest_multiple(
        source_width * scale, MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
    )
    target_height = _nearest_multiple(
        source_height * scale, MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
    )
    return {
        "geometry": "reference_image_resolved",
        "shape_policy_version": "reference_image_short_edge_v1",
        "base_short_edge": MINIMAX_H3_REFERENCE_IMAGE_SHORT_EDGE,
        "effective_short_edge": min(target_width, target_height),
        "size_mode": "short_edge",
        "multiple": MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE,
        "rounding": "nearest",
        "allow_upscale": True,
        "width": target_width,
        "height": target_height,
    }


def minimax_h3_resize_reference_image(
    image: Any,
    *,
    target_width: int,
    target_height: int,
) -> Any:
    """Resize a reference image to the shape fixed by pre-queue admission."""

    from PIL import Image

    if target_width <= 0 or target_height <= 0:
        raise ValueError("reference image target dimensions must be positive")
    if (
        target_width % MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
        or target_height % MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE
    ):
        raise ValueError(
            "reference image target dimensions must be aligned to "
            f"{MINIMAX_H3_REFERENCE_IMAGE_MULTIPLE}"
        )
    image = image.convert("RGB")
    if (target_width, target_height) == image.size:
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def _load_waveform(
    path: str,
    *,
    material_chain: str = "audio",
    max_duration_seconds: float | None = None,
    start_time_seconds: float = 0.0,
    source_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Apply the audio material chain.

    Pure-audio references preserve their source rate while normalizing to
    stereo. Video-bearing references first extract 44.1 kHz stereo PCM. The
    audio VAE boundary then performs the single 32 kHz resample below. ffmpeg
    writes bounded interleaved float PCM directly to stdout, avoiding a
    temporary lossless file plus a second decode.
    """

    import numpy as np

    if max_duration_seconds is not None:
        max_duration_seconds = float(max_duration_seconds)
        if not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0:
            raise ValueError("reference audio duration bound must be positive")
    start_time_seconds = float(start_time_seconds)
    if not math.isfinite(start_time_seconds) or start_time_seconds < 0:
        raise ValueError("reference audio start time must be non-negative")

    if material_chain == "audio":
        if source_sample_rate is None or int(source_sample_rate) <= 0:
            raise ValueError("reference audio sample rate must be positive")
        source_rate = int(source_sample_rate)
    elif material_chain in {
        "video.reference_preserve",
        "video_audio.reference_preserve",
    }:
        source_rate = 44100
    else:
        raise ValueError(
            f"unsupported MiniMax H3 audio material chain {material_chain!r}"
        )

    command = [
        "ffmpeg",
        "-v",
        "error",
    ]
    if start_time_seconds > 0:
        command += ["-ss", f"{start_time_seconds:.9g}"]
    command += [
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        str(MINIMAX_H3_AUDIO_CHANNELS),
    ]
    if material_chain != "audio":
        command += ["-ar", str(source_rate)]
    if max_duration_seconds is not None:
        command += ["-t", f"{max_duration_seconds:.9g}"]
    command += ["-f", "f32le", "pipe:1"]
    decoded = subprocess.run(command, check=True, capture_output=True)
    payload = decoded.stdout
    if not isinstance(payload, bytes):
        raise TypeError("ffmpeg float PCM output must be bytes")
    frame_bytes = MINIMAX_H3_AUDIO_CHANNELS * torch.float32.itemsize
    if len(payload) % frame_bytes:
        raise ValueError(
            "ffmpeg returned a partial reference-audio sample frame: "
            f"{len(payload)} bytes"
        )
    waveform = torch.from_numpy(
        np.frombuffer(payload, dtype=np.float32)
        .reshape(-1, MINIMAX_H3_AUDIO_CHANNELS)
        .T.copy()
    )
    return waveform, source_rate


@functools.lru_cache(maxsize=8)
def _audio_resampler(source_rate: int):
    import torchaudio

    return torchaudio.transforms.Resample(source_rate, MINIMAX_H3_AUDIO_SAMPLE_RATE)


@torch.inference_mode()
def minimax_h3_encode_reference_audio_rows(
    audio_vae: Any,
    audio_path: str,
    arch_config: MiniMaxH3AudioVAEArchConfig,
    *,
    material_chain: str = "audio",
    max_duration_seconds: float | None = None,
    start_time_seconds: float = 0.0,
    source_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Encode a reference audio file into normalized channel-major rows.

    Returns {"rows": [2*T, 32] fp32 cpu, "ref_audio_t": T,
    "duration_seconds": float}.
    """
    model = audio_vae
    device = next(model.parameters()).device
    waveform, source_rate = _load_waveform(
        audio_path,
        material_chain=material_chain,
        max_duration_seconds=max_duration_seconds,
        start_time_seconds=start_time_seconds,
        source_sample_rate=source_sample_rate,
    )
    if waveform.numel() == 0:
        raise ValueError(f"reference audio is empty: {audio_path}")
    if int(source_rate) != MINIMAX_H3_AUDIO_SAMPLE_RATE:
        waveform = _audio_resampler(int(source_rate))(waveform)
    waveform = waveform.to(device)

    with _AudioVAEDeterminismContext():
        audio_data = model.preprocess(
            waveform.unsqueeze(1), MINIMAX_H3_AUDIO_SAMPLE_RATE
        )
        z = model.encoder(audio_data)
        if bool(getattr(model, "attn_proj", False)):
            z = model.pre_block(z.transpose(1, 2)).transpose(1, 2)
        if not hasattr(model, "mean_proj"):
            raise AttributeError(
                "audio VAE model must expose mean_proj for deterministic mean encoding"
            )
        latent = model.mean_proj(z).float()  # [2, 32, T] or [2, T, 32]
    if latent.ndim != 3:
        raise ValueError(f"expected 3D audio latent, got {list(latent.shape)}")
    latent_channels = arch_config.latent_channels
    if int(latent.shape[-1]) != latent_channels:
        if int(latent.shape[1]) != latent_channels:
            raise ValueError(f"cannot canonicalize audio latent {list(latent.shape)}")
        latent = latent.transpose(1, 2).contiguous()  # -> [2, T, 32]
    latent = latent.cpu()

    mean, std = _cached_latent_mean_std(
        tuple(arch_config.latents_mean),
        tuple(arch_config.latents_std),
        (1, 1, latent_channels),
    )
    latent.sub_(mean).div_(std)
    rows = latent.reshape(-1, latent_channels).to(torch.float32).contiguous()
    ref_audio_t = int(latent.shape[1])
    return {
        "rows": rows,
        "ref_audio_t": ref_audio_t,
        "duration_seconds": float(waveform.shape[-1])
        / float(MINIMAX_H3_AUDIO_SAMPLE_RATE),
    }


MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY = "minimax_h3_prepared_reference_image"


def minimax_h3_decode_reference_video_frames(
    video_path: str,
    *,
    target_width: int,
    target_height: int,
    target_frame_count: int,
    fps: float = MINIMAX_H3_SUPPORTED_FPS,
    start_time_seconds: float = 0.0,
    share_across_replicas: bool = False,
) -> Any:
    """Decode, transform, and truncate a reference video in one ffmpeg pass.

    ffmpeg applies display rotation, CFR sampling, direct Lanczos scaling, and
    square-pixel normalization before writing a bounded RGB24 stream.
    The returned array is shared by Qwen and the visual VAE, so conditioning
    never passes through a lossy x264 intermediate or a second video decode.
    """
    import numpy as np

    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be positive")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target reference-video dimensions must be positive")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("reference-video fps must be positive")
    start_time_seconds = float(start_time_seconds)
    if not math.isfinite(start_time_seconds) or start_time_seconds < 0:
        raise ValueError("reference-video start time must be non-negative")

    filters = (
        f"fps={float(fps):g},"
        f"scale={target_width}:{target_height}:flags=lanczos,"
        "setsar=1"
    )
    command = ["ffmpeg", "-v", "error"]
    if start_time_seconds > 0:
        # Input seeking remains accurate while transcoding and avoids decoding
        # the unused prefix of a long reference into RGB frames.
        command += ["-ss", f"{start_time_seconds:.9g}"]
    command += [
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filters,
        "-frames:v",
        str(target_frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
    ]
    frame_bytes = target_width * target_height * 3
    if share_across_replicas:
        payload, payload_size = _decode_reference_video_shared(command)
    else:
        payload, payload_size = _decode_reference_video_local(command)

    if payload_size <= 0:
        raise ValueError(f"reference video has no frames: {video_path}")
    if payload_size % frame_bytes:
        raise ValueError(
            "ffmpeg returned a partial reference-video frame: "
            f"{payload_size} bytes for {target_width}x{target_height} RGB24"
        )
    frame_count = payload_size // frame_bytes
    return np.frombuffer(payload, dtype=np.uint8).reshape(
        frame_count, target_height, target_width, 3
    )


def _decode_reference_video_local(command: list[str]) -> tuple[Any, int]:
    """Write one worker's RGB stream without a large stdout aggregation."""

    # Linux workers can let ffmpeg write the exact RGB24 stream into an
    # anonymous file descriptor. Mapping that output avoids communicate()'s
    # chunk list and final bytes join for a several-hundred-MiB reference.
    output_fd = -1
    if sys.platform.startswith("linux"):
        try:
            output_fd = os.memfd_create(
                "sglang-h3-reference-video",
                flags=os.MFD_CLOEXEC,
            )
        except OSError:
            output_fd = -1

    payload: Any = b""
    payload_size = 0
    if output_fd >= 0:
        try:
            payload_size = _write_reference_video_to_fd(command, output_fd)
            if payload_size > 0:
                payload = mmap.mmap(
                    output_fd,
                    payload_size,
                    access=mmap.ACCESS_WRITE,
                )
        finally:
            os.close(output_fd)
    else:
        decoded = subprocess.run(
            [*command, "pipe:1"],
            check=True,
            capture_output=True,
        )
        payload = decoded.stdout
        if not isinstance(payload, bytes):
            raise TypeError("ffmpeg RGB24 output must be bytes")
        payload_size = len(payload)
    return payload, payload_size


def _write_reference_video_to_fd(command: list[str], output_fd: int) -> int:
    subprocess.run(
        [*command, f"pipe:{output_fd}"],
        check=True,
        pass_fds=(output_fd,),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return os.lseek(output_fd, 0, os.SEEK_CUR)


def _all_gather_world_objects(group: Any, value: Any) -> list[Any]:
    values = [None] * group.world_size
    torch.distributed.all_gather_object(
        values,
        value,
        group=group.cpu_group,
    )
    return values


@functools.lru_cache(maxsize=1)
def _reference_video_host_leader() -> int:
    group = get_world_group()
    hostnames = _all_gather_world_objects(group, socket.gethostname())
    return hostnames.index(hostnames[group.rank_in_group])


def _decode_reference_video_shared(command: list[str]) -> tuple[Any, int]:
    """Decode once per host and map the same RGB pages on its worker ranks."""
    group = get_world_group()
    if (
        group.world_size <= 1
        or not sys.platform.startswith("linux")
        or not os.path.isdir("/proc/self/fd")
    ):
        return _decode_reference_video_local(command)

    leader = _reference_video_host_leader()
    is_leader = group.rank_in_group == leader

    leader_fd = -1
    payload_size = 0
    owner_exception = None
    leader_state = None
    if is_leader:
        try:
            try:
                leader_fd = os.memfd_create(
                    "sglang-h3-reference-video-shared",
                    flags=os.MFD_CLOEXEC,
                )
            except OSError:
                # Anonymous file descriptors can be disabled by a container's
                # seccomp policy. Tell every host to use the unchanged local
                # decode path instead of failing a valid request.
                leader_state = (None, 0, None)
            else:
                payload_size = _write_reference_video_to_fd(command, leader_fd)
                leader_state = (
                    f"/proc/{os.getpid()}/fd/{leader_fd}",
                    payload_size,
                    None,
                )
        except Exception as exc:
            owner_exception = exc
            leader_state = (None, 0, f"{type(exc).__name__}: {exc}")

    states = _all_gather_world_objects(group, leader_state)
    host_states = [state for state in states if state is not None]
    owner_error = next(
        (state[2] for state in host_states if state[2] is not None),
        None,
    )
    # Every rank makes the same decision here. In particular, a decode failure
    # on one host must not leave the other hosts entering the mapping collective.
    if owner_error is not None:
        if leader_fd >= 0:
            os.close(leader_fd)
        if owner_exception is not None and owner_error.endswith(str(owner_exception)):
            raise owner_exception
        raise RuntimeError(f"MiniMax H3 shared video decode failed: {owner_error}")
    if any(state[0] is None for state in host_states):
        if leader_fd >= 0:
            os.close(leader_fd)
        return _decode_reference_video_local(command)
    if any(state[1] <= 0 for state in host_states):
        if leader_fd >= 0:
            os.close(leader_fd)
        return b"", 0

    state = states[leader]
    if state is None:
        raise RuntimeError("MiniMax H3 shared video decode returned no descriptor")
    local_path, payload_size, _ = state
    if payload_size <= 0:
        if leader_fd >= 0:
            os.close(leader_fd)
        return b"", 0
    if local_path is None:
        raise RuntimeError("MiniMax H3 shared video decode returned no path")

    mapping = None
    map_error = None
    try:
        map_fd = os.open(local_path, os.O_RDWR)
        try:
            mapping = mmap.mmap(map_fd, payload_size, access=mmap.ACCESS_COPY)
        finally:
            os.close(map_fd)
    except Exception as exc:
        map_error = f"{type(exc).__name__}: {exc}"

    map_errors = _all_gather_world_objects(group, map_error)
    failed = next((error for error in map_errors if error is not None), None)
    if is_leader:
        os.close(leader_fd)
    if failed is not None:
        if mapping is not None:
            mapping.close()
        # /proc fd traversal can be denied by hidepid or a container policy.
        # All ranks fall back together so the optimization never makes a
        # previously valid request fail or changes its RGB bytes.
        return _decode_reference_video_local(command)

    if mapping is None:
        raise RuntimeError("MiniMax H3 shared video mapping returned no payload")
    return mapping, payload_size


MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED = 42
MINIMAX_H3_REFERENCE_VIDEO_PATCH_SIZE = (1, 2, 2)


@torch.inference_mode()
def minimax_h3_encode_reference_video_rows(
    video_vae: Any,
    frames: Any,
    arch_config: MiniMaxH3VideoVAEArchConfig,
) -> tuple[torch.Tensor, int, int, int]:
    """Encode transformed reference-video frames into packed imgvid cond rows.

    Frames come from the request's single ffmpeg transformation pass, then use
    the SAME ``encode_videos`` recipe as the fl2va keyframe sink (fp32 weights,
    configured complete-tile parallelism, torch seed pinned at 42 because the
    encode SAMPLES the DiagonalGaussian, fp16 latent), then normalize
    and [1,2,2]-patchify. The VAE's clip_length=17 / token_drop=3 give the
    17-frames-per-5-latents temporal grouping (107 frames -> 32 latents).

    Returns (rows [n, 96] fp32 cpu, latent_t, latent_h, latent_w).
    """
    import numpy as np

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.packed_tokens import (
        minimax_h3_patchify_video_latent,
    )

    frames = np.asarray(frames)
    if (
        frames.ndim != 4
        or int(frames.shape[-1]) != 3
        or frames.dtype != np.uint8
        or int(frames.shape[0]) <= 0
    ):
        raise ValueError(
            "reference-video frames must be non-empty [T,H,W,3] uint8, got "
            f"shape={list(frames.shape)}, dtype={frames.dtype}"
        )

    parameter = next(video_vae.parameters())
    prev_dtype = parameter.dtype
    if prev_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        with minimax_h3_scoped_encode_rng(
            MINIMAX_H3_REFERENCE_VIDEO_ENCODE_SEED, parameter.device
        ):
            z = video_vae.encode_videos(frames, use_fp16_latent=True)[0]
    finally:
        if prev_dtype != torch.float32:
            video_vae.to(prev_dtype)
    z = z.cpu().float()
    if z.dim() == 4:
        z = z[None]
    latent_channels = arch_config.latent_channels
    if z.dim() != 5 or int(z.shape[1]) != latent_channels:
        raise ValueError(f"unexpected reference video latent shape {list(z.shape)}")
    latent_t, latent_h, latent_w = int(z.shape[2]), int(z.shape[3]), int(z.shape[4])
    mean, std = _cached_latent_mean_std(
        tuple(arch_config.latents_mean),
        tuple(arch_config.latents_std),
        (1, latent_channels, 1, 1, 1),
    )
    z.sub_(mean).div_(std)
    rows = minimax_h3_patchify_video_latent(
        z, patch_size=list(MINIMAX_H3_REFERENCE_VIDEO_PATCH_SIZE)
    )
    return rows.to(torch.float32), latent_t, latent_h, latent_w


MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS = 2.0
MINIMAX_H3_QWEN_TEMPORAL_PATCH = 2


def minimax_h3_sample_reference_video_frames(
    frames: Any,
) -> dict[str, Any]:
    """Sample Qwen frames from the shared transformed RGB array.

    Frame-sampling recipe (24 FPS -> 2 FPS strided view) plus the qwen3
    timestamp rule (indices padded to the
    temporal patch size with the last frame, block ts = mean of the pair at
    sample fps; text is rendered later with ``f"<{ts:.1f} seconds>"``).

    Returns {"frames": np.ndarray TxHxWx3 u8, "block_timestamps": [float]}.
    """
    import numpy as np

    frames = np.asarray(frames)
    if frames.ndim != 4 or int(frames.shape[0]) <= 0:
        raise ValueError(
            "Qwen reference-video sampling requires non-empty [T,H,W,C] frames"
        )
    sample_stride = int(MINIMAX_H3_SUPPORTED_FPS / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS)
    sampled_frames = frames[::sample_stride]
    ts = [
        i / MINIMAX_H3_QWEN_VIDEO_SAMPLE_FPS
        for i in range(int(sampled_frames.shape[0]))
    ]
    pad = (-len(ts)) % MINIMAX_H3_QWEN_TEMPORAL_PATCH
    ts = ts + [ts[-1]] * pad
    block_timestamps = [
        (ts[i] + ts[i + MINIMAX_H3_QWEN_TEMPORAL_PATCH - 1]) / 2
        for i in range(0, len(ts), MINIMAX_H3_QWEN_TEMPORAL_PATCH)
    ]
    return {"frames": sampled_frames, "block_timestamps": block_timestamps}


def _reference_video_materials(plan: Any) -> list[Any]:
    return [
        m
        for m in plan.materials
        if m.material_chain
        in {"video.reference_preserve", "video_audio.reference_preserve"}
    ]


def _reference_video_target_frame_count(
    *,
    plan: Any,
) -> int:
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
        minimax_h3_align_frame_count,
        minimax_h3_frame_count_from_video_latent_t,
    )

    shape = plan.shape
    fps = int(shape["fps"])
    duration = shape.get("duration_seconds")
    if duration is not None:
        return minimax_h3_align_frame_count(int(round(float(duration) * fps)))
    if shape.get("video_latent_t") is not None:
        return minimax_h3_frame_count_from_video_latent_t(int(shape["video_latent_t"]))
    raise ValueError(
        "reference-video preparation requires pre-queue resolved temporal dimensions"
    )


def minimax_h3_prepared_reference_videos(
    batch: Any,
    plan: Any,
    *,
    share_across_replicas: bool = False,
) -> dict[str, Any]:
    """Decode the bounded reference-video RGB frames once per request.

    BOTH the visual-condition tokenizer and Qwen consume the same transformed
    array. Its frame cap comes from the resolved target duration (17n+5 rule).
    The original path travels alongside for direct soundtrack decoding.
    """
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY,
    )

    cached = batch.extra.get(MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY)
    if cached is not None:
        return cached
    videos = _reference_video_materials(plan)
    if not videos:
        raise NotImplementedError(
            "ref2va video preparation requires a video or video_audio reference"
        )

    prepared_videos = []
    for material in videos:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
            minimax_h3_localize_material_uri,
        )

        video_path = minimax_h3_localize_material_uri(
            batch,
            material.uri,
            condition_type=material.condition_type,
            condition_index=int(material.condition_index),
        )
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
        )

        condition_index = int(material.condition_index)
        source_facts = batch.extra.get(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, {}).get(
            condition_index
        )
        resolved_material_shape = batch.extra.get(
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY, {}
        ).get(condition_index)
        if not isinstance(source_facts, dict) or not isinstance(
            resolved_material_shape, dict
        ):
            raise ValueError(
                "reference-video preparation requires cached pre-queue probe "
                f"and shape facts for conditions[{condition_index}]"
            )
        input_has_audio = bool(source_facts.get("has_audio"))
        target_frames = _reference_video_target_frame_count(plan=plan)
        frames = minimax_h3_decode_reference_video_frames(
            video_path,
            target_width=int(resolved_material_shape["width"]),
            target_height=int(resolved_material_shape["height"]),
            target_frame_count=target_frames,
            fps=float(plan.shape["fps"]),
            start_time_seconds=float(material.start_time_seconds),
            share_across_replicas=share_across_replicas,
        )
        prepared_videos.append(
            {
                "frames": frames,
                "original_path": video_path,
                "target_frame_count": target_frames,
                "frame_count": int(frames.shape[0]),
                "condition_index": int(material.condition_index),
                "material_chain": str(material.material_chain),
                "start_time_seconds": float(material.start_time_seconds),
                "input_has_audio": input_has_audio,
                "width": int(resolved_material_shape["width"]),
                "height": int(resolved_material_shape["height"]),
            }
        )
    prepared = {
        key: value for key, value in prepared_videos[0].items() if key != "frames"
    }
    prepared["videos"] = prepared_videos
    batch.extra[MINIMAX_H3_PREPARED_REFERENCE_VIDEO_EXTRA_KEY] = prepared
    return prepared


def minimax_h3_prepared_reference_image(batch: Any, plan: Any) -> dict[str, Any]:
    """Resize ref2va image references to their pre-queue-resolved shapes.

    Qwen (pixel_values) and the visual-condition tokenizer consume the identical
    prepared image. The runtime never recomputes geometry from ``plan.shape``;
    it consumes the per-material width/height admitted before queueing.
    """
    cached = batch.extra.get(MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY)
    if cached is not None:
        return cached
    images = [
        m for m in plan.materials if m.material_chain == "image.reference_preserve"
    ]
    if not images:
        raise ValueError("ref2va requires at least one image reference")
    from PIL import Image, ImageOps

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
        minimax_h3_localize_material_uri,
    )

    prepared_images = []
    for material in images:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.prequeue import (
            MINIMAX_H3_PROBE_FACTS_EXTRA_KEY,
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY,
        )

        condition_index = int(material.condition_index)
        source_facts = batch.extra.get(MINIMAX_H3_PROBE_FACTS_EXTRA_KEY, {}).get(
            condition_index
        )
        resolved_shape = batch.extra.get(
            MINIMAX_H3_RESOLVED_MATERIAL_SHAPES_EXTRA_KEY, {}
        ).get(condition_index)
        if not isinstance(source_facts, dict) or not isinstance(resolved_shape, dict):
            raise ValueError(
                "reference-image preparation requires cached pre-queue probe "
                f"and shape facts for conditions[{condition_index}]"
            )
        image_path = minimax_h3_localize_material_uri(
            batch,
            material.uri,
            condition_type=material.condition_type,
            condition_index=condition_index,
        )
        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image).convert("RGB")
        expected_size = (
            int(resolved_shape["width"]),
            int(resolved_shape["height"]),
        )
        prepared_image = minimax_h3_resize_reference_image(
            image,
            target_width=expected_size[0],
            target_height=expected_size[1],
        )
        if prepared_image.size != expected_size:
            raise ValueError(
                "reference image preparation disagrees with pre-queue shape: "
                f"expected={expected_size}, actual={prepared_image.size}"
            )
        prepared_images.append(
            {
                "image": prepared_image,
                "condition_index": condition_index,
            }
        )
    prepared = {
        # single-image consumers keep the existing keys
        "image": prepared_images[0]["image"],
        "condition_index": prepared_images[0]["condition_index"],
        "images": prepared_images,
    }
    batch.extra[MINIMAX_H3_PREPARED_REFERENCE_IMAGE_EXTRA_KEY] = prepared
    return prepared


__all__ = [
    "minimax_h3_decode_reference_video_frames",
    "minimax_h3_encode_reference_audio_rows",
    "minimax_h3_encode_reference_video_rows",
    "minimax_h3_prepared_reference_image",
    "minimax_h3_prepared_reference_videos",
    "minimax_h3_resolve_reference_image_shape",
    "minimax_h3_sample_reference_video_frames",
]
