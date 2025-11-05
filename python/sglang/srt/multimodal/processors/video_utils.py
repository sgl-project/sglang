import base64 as _pybase64
import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Union

import requests

try:
    import pybase64

    _b64 = pybase64
except Exception:
    _b64 = _pybase64


@dataclass
class VideoInput:
    path: str
    cleanup: bool = False
    frame_count_limit: Optional[int] = None


def _ramdisk_tmpdir() -> str:
    shm = "/dev/shm"
    if os.path.isdir(shm) and os.access(shm, os.W_OK):
        return shm
    return tempfile.gettempdir()


def _mkstemp_path(suffix: str = ".mp4") -> str:
    tmpdir = _ramdisk_tmpdir()
    fd, path = tempfile.mkstemp(prefix="mmvid_", suffix=suffix, dir=tmpdir)
    os.close(fd)
    return path


def _write_bytes_to_file(b: bytes, path: str) -> None:
    with open(path, "wb") as f:
        f.write(b)


def _is_data_uri(s: str) -> bool:
    return s.startswith("data:")


def _maybe_guess_suffix_from_url(url: str) -> str:
    _, ext = os.path.splitext(url)
    if ext and len(ext) <= 5:
        return ext
    return ".mp4"


def _download_url_to_file(url: str, path: str, timeout: int = 10) -> None:
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def make_video_input(
    video_file: "Union[str, bytes]",
    *,
    frame_count_limit: Optional[int] = None,
    request_timeout_env: str = "REQUEST_TIMEOUT",
) -> VideoInput:
    """
    Based on original load_video logic, refactor to return VideoInput.
    Note: Don't remove temp file here, the subprocess is responsible to delete the temp file.
    """
    if isinstance(video_file, (bytes, bytearray)):
        path = _mkstemp_path(".mp4")
        _write_bytes_to_file(video_file, path)
        return VideoInput(path=path, cleanup=True, frame_count_limit=frame_count_limit)

    if isinstance(video_file, str):
        # 1) http(s)://
        if video_file.startswith(("http://", "https://")):
            timeout = int(os.getenv(request_timeout_env, "10"))
            suffix = _maybe_guess_suffix_from_url(video_file)
            path = _mkstemp_path(suffix)
            _download_url_to_file(video_file, path, timeout=timeout)
            return VideoInput(
                path=path, cleanup=True, frame_count_limit=frame_count_limit
            )

        # 2) data:URI
        if _is_data_uri(video_file):
            try:
                _, encoded = video_file.split(",", 1)
            except ValueError:
                raise ValueError("Invalid data URI for video.")
            video_bytes = _b64.b64decode(encoded, validate=True)
            path = _mkstemp_path(".mp4")
            _write_bytes_to_file(video_bytes, path)
            return VideoInput(
                path=path, cleanup=True, frame_count_limit=frame_count_limit
            )

        # 3) local file
        if os.path.isfile(video_file):
            return VideoInput(
                path=video_file, cleanup=False, frame_count_limit=frame_count_limit
            )

        # 4) fallback base64
        try:
            video_bytes = _b64.b64decode(video_file, validate=True)
        except Exception:
            raise ValueError(
                f"Unsupported video input string: not a file, URL, data URI, or valid base64."
            )
        path = _mkstemp_path(".mp4")
        _write_bytes_to_file(video_bytes, path)
        return VideoInput(path=path, cleanup=True, frame_count_limit=frame_count_limit)

    raise ValueError(f"Unsupported video input type: {type(video_file)}")
