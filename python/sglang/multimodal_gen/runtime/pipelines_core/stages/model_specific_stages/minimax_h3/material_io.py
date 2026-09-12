# SPDX-License-Identifier: Apache-2.0
"""Request-owned material URI localization for the MiniMax H3 pipeline.

The canonical MiniMax H3 contract intentionally carries semantic URIs rather
than worker-local paths.  Direct media consumers (Pillow, ffmpeg and
torchaudio) cannot consume every URI scheme in that contract, so localization
belongs at the model-specific material boundary.  Materialized sources and
derived work directories are registered on ``Req.extra`` and explicitly
released by the encoder stages.
"""

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

MINIMAX_H3_MATERIAL_CACHE_EXTRA_KEY = "minimax_h3_material_localization"
MINIMAX_H3_MATERIAL_PROBE_EXTRA_KEY = "minimax_h3_material_probe_facts"
MINIMAX_H3_TEMP_DIRS_EXTRA_KEY = "minimax_h3_request_temp_dirs"
MINIMAX_H3_HTTP_READ_CHUNK_BYTES = 1024 * 1024
MINIMAX_H3_BASE64_DECODE_CHUNK_CHARS = 1024 * 1024
MINIMAX_H3_BASE64_HEADER_MAX_CHARS = 4 * 1024
MINIMAX_H3_TAR_HEADER_MAX_ENCODED_CHARS = 64 * 1024

_BASE64_ALPHABET = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=_-"
)

_DEFAULT_SUFFIX_BY_TYPE = {
    "image": ".png",
    "video": ".mp4",
    "video_audio": ".mp4",
    "audio": ".wav",
}
_SUFFIX_BY_MEDIA_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/flac": ".flac",
}


def minimax_h3_register_temp_dir(batch: Any, path: str, *, owner: str) -> str:
    """Register one request-owned directory and return *path* unchanged."""

    registry = batch.extra.setdefault(MINIMAX_H3_TEMP_DIRS_EXTRA_KEY, {})
    paths = registry.setdefault(str(owner), [])
    normalized = str(path)
    if normalized not in paths:
        paths.append(normalized)
    return normalized


def minimax_h3_cleanup_temp_dirs(
    batch: Any, *, owners: Iterable[str] | None = None
) -> None:
    """Remove registered request directories, tolerating repeated cleanup."""

    registry = batch.extra.get(MINIMAX_H3_TEMP_DIRS_EXTRA_KEY)
    selected = (
        list(registry)
        if owners is None and isinstance(registry, dict)
        else [str(owner) for owner in (owners or ())]
    )
    if isinstance(registry, dict):
        for owner in selected:
            paths = registry.pop(owner, [])
            if isinstance(paths, (list, tuple)):
                for path in paths:
                    shutil.rmtree(str(path), ignore_errors=True)
        if not registry:
            batch.extra.pop(MINIMAX_H3_TEMP_DIRS_EXTRA_KEY, None)
    if owners is None or "material" in selected:
        batch.extra.pop(MINIMAX_H3_MATERIAL_CACHE_EXTRA_KEY, None)
        batch.extra.pop(MINIMAX_H3_MATERIAL_PROBE_EXTRA_KEY, None)


def _base64_uri_payload_start(uri: str) -> tuple[int, str | None]:
    media_type = None
    if uri.startswith("data:"):
        separator = uri.find(",")
        if separator < 0:
            raise ValueError("data URI must contain a comma separator")
        if separator > MINIMAX_H3_BASE64_HEADER_MAX_CHARS:
            raise ValueError("data URI header is too large")
        header = uri[:separator]
        if ";base64" not in header:
            raise ValueError("data URI must use ;base64 encoding")
        media_type = header[5:].split(";", 1)[0].lower() or None
        payload_start = separator + 1
    elif uri.startswith("base64://"):
        payload_start = len("base64://")
        separator = uri.find(",", payload_start)
        if separator >= 0:
            if separator - payload_start > MINIMAX_H3_BASE64_HEADER_MAX_CHARS:
                raise ValueError("base64 URI header is too large")
            header = uri[payload_start:separator]
            media_type = header.split(";", 1)[0].lower() or None
            payload_start = separator + 1
    else:  # pragma: no cover - guarded by the caller
        raise ValueError("not a base64 material URI")
    return payload_start, media_type


def _iter_base64_payload_bytes(uri: str, payload_start: int):
    """Yield validated, unquoted base64 bytes without copying the payload."""

    index = payload_start
    while index < len(uri):
        character = uri[index]
        if character == "%":
            if index + 2 >= len(uri):
                raise ValueError("material URI has an invalid percent escape")
            try:
                value = int(uri[index + 1 : index + 3], 16)
            except ValueError as exc:
                raise ValueError("material URI has an invalid percent escape") from exc
            index += 3
            character = chr(value)
        else:
            index += 1

        if character.isspace():
            continue
        if len(character) != 1 or ord(character) > 127:
            raise ValueError("material URI base64 payload must be ASCII")
        value = ord(character)
        if value not in _BASE64_ALPHABET:
            raise ValueError(
                f"material URI has an invalid base64 character {character!r}"
            )
        yield value


def _parse_tar_member_uri(uri: str) -> tuple[Path, int, int, str | None]:
    if uri.startswith("tar+offset://"):
        prefix = "tar+offset://"
    elif uri.startswith("tar+b64header://"):
        prefix = "tar+b64header://"
    else:
        raise ValueError("unsupported tar material URI")
    try:
        tar_path, encoded_header = uri[len(prefix) :].rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(
            "tar material URI must contain '<tar_path>:<encoded_header>'"
        ) from exc
    if len(encoded_header) > MINIMAX_H3_TAR_HEADER_MAX_ENCODED_CHARS:
        raise ValueError("tar material URI encoded header is too large")
    padded = encoded_header + "=" * (-len(encoded_header) % 4)
    try:
        header = json.loads(
            base64.b64decode(
                padded.encode("ascii"), altchars=b"-_", validate=True
            ).decode("utf-8")
        )
    except Exception as exc:
        raise ValueError("tar material URI has an invalid encoded header") from exc
    if not isinstance(header, dict):
        raise ValueError("tar material URI header must be a JSON object")
    if header.get("schema") != "sglang.tar_member_ref/v1":
        raise ValueError(
            f"unsupported tar material header schema: {header.get('schema')!r}"
        )
    try:
        offset = int(header["offset_data"])
        size = int(header["size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "tar material header requires integer offset_data and size"
        ) from exc
    if offset < 0 or size < 0:
        raise ValueError("tar material offset_data and size must be non-negative")
    return (
        Path(tar_path).expanduser(),
        offset,
        size,
        str(header.get("member") or "") or None,
    )


def _safe_suffix(value: str | None) -> str | None:
    if not value:
        return None
    suffix = Path(urllib.parse.urlsplit(value).path).suffix.lower()
    if suffix and len(suffix) <= 10 and suffix[1:].isalnum():
        return suffix
    return None


def _checked_material_file(path: Path, *, label: str) -> str:
    """Validate that a localized source exists and is non-empty."""

    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{label} is empty: {path}")
    return str(path)


def _parse_frame_rate(value: Any) -> float:
    if value in {None, "", "N/A", "0/0"}:
        return 0.0
    raw = str(value)
    try:
        if "/" in raw:
            numerator, denominator = raw.split("/", 1)
            denominator_value = float(denominator)
            parsed = float(numerator) / denominator_value if denominator_value else 0.0
        else:
            parsed = float(raw)
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _parse_display_ratio(value: Any) -> float:
    if value in {None, "", "N/A", "0:1", "0/1"}:
        return 0.0
    raw = str(value).strip()
    separator = ":" if ":" in raw else "/" if "/" in raw else None
    try:
        if separator is None:
            ratio = float(raw)
        else:
            numerator, denominator = raw.split(separator, 1)
            ratio = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0
    return ratio if math.isfinite(ratio) and ratio > 0 else 0.0


def _stream_rotation_degrees(stream: dict[str, Any]) -> float:
    values: list[Any] = []
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        values.extend(
            item.get("rotation") for item in side_data if isinstance(item, dict)
        )
    tags = stream.get("tags")
    if isinstance(tags, dict):
        values.append(tags.get("rotate"))
    for value in values:
        if value in {None, "", "N/A"}:
            continue
        try:
            rotation = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(rotation):
            return rotation % 360.0
    return 0.0


def _display_geometry(stream: dict[str, Any]) -> tuple[float, float, float, float]:
    """Return square-pixel display width/height, SAR, and rotation."""

    coded_width = int(stream.get("width") or 0)
    coded_height = int(stream.get("height") or 0)
    sar = _parse_display_ratio(stream.get("sample_aspect_ratio")) or 1.0
    dar = _parse_display_ratio(stream.get("display_aspect_ratio"))
    physical_height = float(coded_height)
    physical_width = dar * physical_height if dar > 0.0 else float(coded_width) * sar
    rotation = _stream_rotation_degrees(stream)
    quarter_turns = round(rotation / 90.0)
    if abs(rotation - quarter_turns * 90.0) <= 1e-6:
        if quarter_turns % 2:
            return physical_height, physical_width, sar, rotation
        return physical_width, physical_height, sar, rotation
    radians = math.radians(rotation)
    cosine = abs(math.cos(radians))
    sine = abs(math.sin(radians))
    display_width = physical_width * cosine + physical_height * sine
    display_height = physical_width * sine + physical_height * cosine
    return display_width, display_height, sar, rotation


_FFPROBE_STREAM_ENTRIES = (
    "stream=codec_type,width,height,duration,sample_rate,channels,"
    "avg_frame_rate,r_frame_rate,nb_frames,sample_aspect_ratio,display_aspect_ratio"
    ":stream_tags=rotate"
)

# ffprobe gained the per-stream "stream_side_data" section in 6.0 and rejects
# the whole -show_entries spec without it. Older builds (e.g. Ubuntu 22.04's
# 4.4) report rotation through the "rotate" stream tag, which
# _stream_rotation_degrees already reads, so drop the section on fallback.
_FFPROBE_ENTRY_VARIANTS = (
    f"{_FFPROBE_STREAM_ENTRIES}:stream_side_data=rotation:format=format_name,duration",
    f"{_FFPROBE_STREAM_ENTRIES}:format=format_name,duration",
)
_ffprobe_entries: str | None = None


def _ffprobe_media(path: str) -> dict[str, Any]:
    global _ffprobe_entries

    variants = (
        (_ffprobe_entries,) if _ffprobe_entries is not None else _FFPROBE_ENTRY_VARIANTS
    )

    last_error: Exception | None = None
    for entries in variants:
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-protocol_whitelist",
                    "file",
                    "-format_whitelist",
                    "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,wav,mp3,flac,ogg",
                    "-show_entries",
                    entries,
                    "-of",
                    "json",
                    "-i",
                    path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            # Only an unknown section name is worth retrying; a genuinely bad
            # input must fail on the first variant.
            if "No match for section" not in (exc.stderr or ""):
                raise
            last_error = exc
            continue
        _ffprobe_entries = entries
        return json.loads(result.stdout)

    raise last_error  # type: ignore[misc]


def _validate_localized_media(
    path: str,
    *,
    condition_type: str,
) -> dict[str, Any]:
    """Probe one localized source and return facts used by MiniMax H3 admission.

    This is deliberately a model-facing validity check: it verifies that the
    source is non-empty, parseable and contains the stream type requested by
    the condition.  Generic transport/resource ceilings do not belong here;
    the model's shape and temporal contracts are resolved separately.
    """

    if condition_type == "image":
        try:
            from PIL import Image, ImageOps

            with Image.open(path) as image:
                coded_width, coded_height = image.size
                image_format = str(image.format or "").upper()
                if coded_width <= 0 or coded_height <= 0:
                    raise ValueError("image has no positive dimensions")
                display_image = ImageOps.exif_transpose(image)
                width, height = display_image.size
        except Exception as exc:
            raise ValueError("MiniMax H3 image material is invalid") from exc
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("MiniMax H3 image material uses an unsupported format")
        if width <= 0 or height <= 0:
            raise ValueError(
                "MiniMax H3 image material has no positive display geometry"
            )
        return {
            "condition_type": "image",
            "coded_width": int(coded_width),
            "coded_height": int(coded_height),
            "display_width": int(width),
            "display_height": int(height),
            "image_format": image_format,
            "exif_transposed": (coded_width, coded_height) != (width, height),
        }

    if condition_type not in {"audio", "video", "video_audio"}:
        raise ValueError(f"unsupported MiniMax H3 condition type {condition_type!r}")
    try:
        payload = _ffprobe_media(path)
    except Exception as exc:
        raise ValueError("MiniMax H3 media material is invalid") from exc

    streams = payload.get("streams") or []
    format_names = set(
        str((payload.get("format") or {}).get("format_name") or "").split(",")
    )
    allowed_formats = {
        "mov",
        "mp4",
        "m4a",
        "3gp",
        "3g2",
        "mj2",
        "matroska",
        "webm",
        "wav",
        "mp3",
        "flac",
        "ogg",
    }
    if not format_names or not format_names.issubset(allowed_formats):
        raise ValueError("MiniMax H3 media container format is not allowed")
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if condition_type in {"video", "video_audio"} and not video_streams:
        raise ValueError("MiniMax H3 video material has no video stream")
    if condition_type in {"audio", "video_audio"} and not audio_streams:
        raise ValueError("MiniMax H3 audio material has no audio stream")

    primary_video: dict[str, Any] | None = None
    for stream in video_streams:
        try:
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "MiniMax H3 video material has invalid dimensions"
            ) from exc
        if width <= 0 or height <= 0:
            raise ValueError("MiniMax H3 video material has no positive dimensions")
        fps = _parse_frame_rate(stream.get("avg_frame_rate")) or _parse_frame_rate(
            stream.get("r_frame_rate")
        )
        if fps <= 0:
            raise ValueError("MiniMax H3 video material has no usable frame rate")
        if primary_video is None:
            primary_video = stream

    for stream in audio_streams:
        try:
            sample_rate = int(stream.get("sample_rate") or 0)
            channels = int(stream.get("channels") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("MiniMax H3 audio material has invalid metadata") from exc
        if sample_rate <= 0:
            raise ValueError("MiniMax H3 audio material has no usable sample rate")
        if channels <= 0:
            raise ValueError("MiniMax H3 audio material has no usable channel count")

    durations: list[float] = []
    for value in [
        (payload.get("format") or {}).get("duration"),
        *(stream.get("duration") for stream in streams),
    ]:
        if value in {None, "", "N/A"}:
            continue
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            durations.append(duration)
    if not durations:
        raise ValueError("MiniMax H3 media material has no positive duration")

    duration_seconds = max(durations)
    facts: dict[str, Any] = {
        "condition_type": condition_type,
        "duration_seconds": duration_seconds,
        "has_audio": bool(audio_streams),
    }
    if primary_video is not None:
        coded_width = int(primary_video.get("width") or 0)
        coded_height = int(primary_video.get("height") or 0)
        display_width, display_height, sar, rotation = _display_geometry(primary_video)
        fps = _parse_frame_rate(primary_video.get("avg_frame_rate"))
        if fps <= 0:
            fps = _parse_frame_rate(primary_video.get("r_frame_rate"))
        raw_count = primary_video.get("nb_frames")
        try:
            frame_count = int(raw_count)
        except (TypeError, ValueError):
            frame_count = max(1, int(round(duration_seconds * fps)))
        if frame_count <= 0:
            frame_count = max(1, int(round(duration_seconds * fps)))
        try:
            video_duration_seconds = float(primary_video.get("duration"))
        except (TypeError, ValueError):
            video_duration_seconds = 0.0
        if not math.isfinite(video_duration_seconds) or video_duration_seconds <= 0:
            video_duration_seconds = frame_count / fps
        facts.update(
            {
                "coded_width": coded_width,
                "coded_height": coded_height,
                "display_width": display_width,
                "display_height": display_height,
                "sample_aspect_ratio": str(
                    primary_video.get("sample_aspect_ratio") or "1:1"
                ),
                "sample_aspect_ratio_value": sar,
                "display_aspect_ratio": display_width / display_height,
                "rotation_degrees": rotation,
                "fps": fps,
                "frame_count": frame_count,
                "video_duration_seconds": video_duration_seconds,
            }
        )
    if audio_streams:
        facts["audio_sample_rate"] = int(audio_streams[0].get("sample_rate") or 0)
        facts["audio_channels"] = int(audio_streams[0].get("channels") or 0)
        try:
            audio_duration_seconds = float(audio_streams[0].get("duration"))
        except (TypeError, ValueError):
            audio_duration_seconds = 0.0
        facts["audio_duration_seconds"] = (
            audio_duration_seconds
            if math.isfinite(audio_duration_seconds) and audio_duration_seconds > 0
            else duration_seconds
        )
    return facts


def _validate_material_once(
    batch: Any,
    uri: str,
    path: str,
    *,
    condition_type: str,
) -> dict[str, Any]:
    probe_facts = batch.extra.setdefault(MINIMAX_H3_MATERIAL_PROBE_EXTRA_KEY, {})
    key = (uri, condition_type)
    cached = probe_facts.get(key)
    if isinstance(cached, dict):
        return cached
    facts = _validate_localized_media(path, condition_type=condition_type)
    if not isinstance(facts, dict):
        raise ValueError("MiniMax H3 material probe did not return facts")
    probe_facts[key] = facts
    return facts


def _material_output_paths(
    batch: Any,
    *,
    condition_type: str,
    condition_index: int,
    source_name: str | None = None,
    media_type: str | None = None,
) -> tuple[Path, Path]:
    suffix = (
        _safe_suffix(source_name)
        or _SUFFIX_BY_MEDIA_TYPE.get(str(media_type or "").lower())
        or _DEFAULT_SUFFIX_BY_TYPE.get(condition_type, ".bin")
    )
    output_path = (
        Path(_material_workdir(batch)) / f"condition_{int(condition_index):04d}{suffix}"
    )
    return output_path, output_path.with_name(output_path.name + ".partial")


def _material_workdir(batch: Any) -> str:
    registry = batch.extra.setdefault(MINIMAX_H3_TEMP_DIRS_EXTRA_KEY, {})
    material_dirs = registry.get("material")
    if isinstance(material_dirs, list) and material_dirs:
        return str(material_dirs[0])
    return minimax_h3_register_temp_dir(
        batch,
        tempfile.mkdtemp(prefix="minimax_h3_material_"),
        owner="material",
    )


def _decode_base64_chunk(encoded: bytes | bytearray) -> bytes:
    try:
        return base64.b64decode(encoded, altchars=b"-_", validate=True)
    except Exception as exc:
        raise ValueError("material URI has an invalid base64 payload") from exc


def _stream_base64_material(
    batch: Any,
    uri: str,
    *,
    condition_type: str,
    condition_index: int,
) -> str:
    payload_start, media_type = _base64_uri_payload_start(uri)

    output_path, partial_path = _material_output_paths(
        batch,
        condition_type=condition_type,
        condition_index=condition_index,
        media_type=media_type,
    )
    total = 0
    encoded_size = 0
    padding = 0
    saw_padding = False
    encoded_chunk = bytearray()
    try:
        with partial_path.open("wb") as output:
            for value in _iter_base64_payload_bytes(uri, payload_start):
                encoded_size += 1
                if value == ord("="):
                    saw_padding = True
                    padding += 1
                    if padding > 2:
                        raise ValueError("material URI has invalid base64 padding")
                elif saw_padding:
                    raise ValueError("material URI has data after base64 padding")
                encoded_chunk.append(value)
                if len(encoded_chunk) == MINIMAX_H3_BASE64_DECODE_CHUNK_CHARS:
                    decoded_chunk = _decode_base64_chunk(encoded_chunk)
                    total += len(decoded_chunk)
                    output.write(decoded_chunk)
                    encoded_chunk.clear()
            if encoded_size == 0:
                raise ValueError("material URI base64 payload is empty")
            if encoded_size % 4 == 1 or (padding and encoded_size % 4):
                raise ValueError("material URI has an invalid base64 payload length")
            if encoded_chunk:
                encoded_chunk.extend(b"=" * (-len(encoded_chunk) % 4))
                decoded_chunk = _decode_base64_chunk(encoded_chunk)
                total += len(decoded_chunk)
                output.write(decoded_chunk)
        decoded_size = (encoded_size * 3) // 4 - padding
        if decoded_size <= 0:
            raise ValueError("material URI decoded payload is empty")
        if total != decoded_size:
            raise ValueError(
                f"MiniMax H3 base64 decoded size {total} != expected {decoded_size}"
            )
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    return str(output_path)


def _stream_tar_member_material(
    batch: Any,
    uri: str,
    *,
    condition_type: str,
    condition_index: int,
) -> str:
    source_path, offset, size, member = _parse_tar_member_uri(uri)
    if size <= 0:
        raise ValueError("tar material payload is empty")
    if not source_path.is_file():
        raise FileNotFoundError(
            f"tar material source does not exist or is not a file: {source_path}"
        )
    source_size = source_path.stat().st_size
    if offset > source_size or size > source_size - offset:
        available = max(0, source_size - offset)
        raise ValueError(
            f"tar material payload is truncated: expected {size} bytes, "
            f"only {available} available"
        )

    output_path, partial_path = _material_output_paths(
        batch,
        condition_type=condition_type,
        condition_index=condition_index,
        source_name=member,
    )
    remaining = size
    try:
        with source_path.open("rb") as source, partial_path.open("wb") as output:
            source.seek(offset)
            while remaining:
                chunk = source.read(min(MINIMAX_H3_HTTP_READ_CHUNK_BYTES, remaining))
                if not chunk:
                    raise ValueError(
                        f"tar material payload is truncated with {remaining} bytes left"
                    )
                if len(chunk) > remaining:
                    raise ValueError("tar material reader returned too many bytes")
                output.write(chunk)
                remaining -= len(chunk)
        partial_path.replace(output_path)
    except Exception:
        partial_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    return str(output_path)


def _http_media_type(response: Any) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    get_content_type = getattr(headers, "get_content_type", None)
    if callable(get_content_type):
        value = get_content_type()
    else:
        value = headers.get("Content-Type") or headers.get("content-type")
        if isinstance(value, str):
            value = value.split(";", 1)[0].strip()
    return str(value).lower() if value else None


def _stream_http_material(
    batch: Any,
    uri: str,
    *,
    condition_type: str,
    condition_index: int,
    timeout_s: float,
) -> str:
    # Use the repository's legacy urllib behavior.  Model-specific material
    # localization intentionally does not perform the shared public SSRF
    # policy or a cumulative request deadline.
    with urllib.request.urlopen(uri, timeout=timeout_s) as response:
        media_type = _http_media_type(response)
        suffix = (
            _safe_suffix(uri)
            or _SUFFIX_BY_MEDIA_TYPE.get(str(media_type or "").lower())
            or _DEFAULT_SUFFIX_BY_TYPE.get(condition_type, ".bin")
        )
        output_path = (
            Path(_material_workdir(batch))
            / f"condition_{int(condition_index):04d}{suffix}"
        )
        partial_path = output_path.with_name(output_path.name + ".partial")
        total = 0
        try:
            with partial_path.open("wb") as output:
                while True:
                    chunk = response.read(MINIMAX_H3_HTTP_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    if not isinstance(chunk, bytes):
                        raise TypeError(
                            "HTTP material response.read() must return bytes, got "
                            f"{type(chunk).__name__}"
                        )
                    total += len(chunk)
                    output.write(chunk)
            if total == 0:
                raise ValueError(f"HTTP material body is empty: {uri}")
            partial_path.replace(output_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            raise
    return str(output_path)


def minimax_h3_localize_material_uri(
    batch: Any,
    uri: str,
    *,
    condition_type: str,
    condition_index: int,
    timeout_s: float = 120.0,
) -> str:
    """Return a local path for a canonical condition URI.

    Local paths and local ``file://`` URIs are validated and returned without
    copying. HTTP(S), base64/data and direct tar-member URIs are materialized
    once per request and cached for all MiniMax H3 consumers.
    """

    if not isinstance(uri, str) or not uri:
        raise ValueError("condition URI must be a non-empty string")
    parsed = None
    for special_scheme in ("data", "base64", "tar+offset", "tar+b64header"):
        if uri.startswith(special_scheme + ":"):
            scheme = special_scheme
            break
    else:
        parsed = urllib.parse.urlsplit(uri)
        scheme = parsed.scheme

    if scheme == "file":
        assert parsed is not None
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"file URI host must be local, got {parsed.netloc!r}")
        output_path = _checked_material_file(
            Path(urllib.parse.unquote(parsed.path)),
            label="MiniMax H3 material source",
        )
        _validate_material_once(batch, uri, output_path, condition_type=condition_type)
        return output_path
    if not scheme:
        output_path = _checked_material_file(
            Path(uri).expanduser(),
            label="MiniMax H3 material source",
        )
        _validate_material_once(batch, uri, output_path, condition_type=condition_type)
        return output_path
    if scheme == "s3":
        raise NotImplementedError(
            "MiniMax H3 s3:// material URIs require a configured artifact resolver"
        )

    cache = batch.extra.setdefault(MINIMAX_H3_MATERIAL_CACHE_EXTRA_KEY, {})
    cached = cache.get(uri)
    if isinstance(cached, str):
        cached_path = Path(cached)
        if cached_path.exists():
            output_path = _checked_material_file(
                cached_path,
                label="cached MiniMax H3 material",
            )
            _validate_material_once(
                batch, uri, output_path, condition_type=condition_type
            )
            return output_path
        cache.pop(uri, None)

    if scheme in {"http", "https"}:
        output_path = _stream_http_material(
            batch,
            uri,
            condition_type=condition_type,
            condition_index=condition_index,
            timeout_s=timeout_s,
        )
        try:
            _validate_material_once(
                batch, uri, output_path, condition_type=condition_type
            )
        except Exception:
            Path(output_path).unlink(missing_ok=True)
            raise
        cache[uri] = output_path
        return output_path

    if scheme in {"data", "base64"}:
        output_path = _stream_base64_material(
            batch,
            uri,
            condition_type=condition_type,
            condition_index=condition_index,
        )
        try:
            _validate_material_once(
                batch, uri, output_path, condition_type=condition_type
            )
        except Exception:
            Path(output_path).unlink(missing_ok=True)
            raise
        cache[uri] = output_path
        return output_path

    if scheme in {"tar+offset", "tar+b64header"}:
        output_path = _stream_tar_member_material(
            batch,
            uri,
            condition_type=condition_type,
            condition_index=condition_index,
        )
        try:
            _validate_material_once(
                batch, uri, output_path, condition_type=condition_type
            )
        except Exception:
            Path(output_path).unlink(missing_ok=True)
            raise
        cache[uri] = output_path
        return output_path

    raise NotImplementedError(
        f"MiniMax H3 material localization does not support URI scheme {scheme!r}"
    )


def minimax_h3_probe_material(
    batch: Any,
    uri: str,
    *,
    condition_type: str,
    condition_index: int,
) -> dict[str, Any]:
    """Localize and return cached display-geometry facts for one condition."""

    path = minimax_h3_localize_material_uri(
        batch,
        uri,
        condition_type=condition_type,
        condition_index=condition_index,
    )
    key = (uri, condition_type)
    facts = batch.extra.get(MINIMAX_H3_MATERIAL_PROBE_EXTRA_KEY, {}).get(key)
    if not isinstance(facts, dict) or not facts:
        raise RuntimeError(
            "MiniMax H3 material localization completed without cached probe facts"
        )
    return {"local_path": path, **facts}


__all__ = [
    "MINIMAX_H3_MATERIAL_CACHE_EXTRA_KEY",
    "MINIMAX_H3_MATERIAL_PROBE_EXTRA_KEY",
    "MINIMAX_H3_TEMP_DIRS_EXTRA_KEY",
    "minimax_h3_cleanup_temp_dirs",
    "minimax_h3_localize_material_uri",
    "minimax_h3_probe_material",
    "minimax_h3_register_temp_dir",
]
