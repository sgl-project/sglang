# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible Audio Speech API.

``POST /v1/audio/speech`` matches OpenAI TTS: JSON (or multipart) in, raw
audio bytes out. SGLang extensions cover LongCat-AudioDiT voice cloning and
sampling controls, the same way ``/v1/images`` and ``/v1/videos`` add fields
on top of the OpenAI request body.

Metadata / content routes under ``/v1/audio/speech/{id}`` follow the image
and mesh stores so a completed clip can be re-fetched.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, Optional, Union

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response

from sglang.multimodal_gen.configs.sample.sampling_params import generate_request_id
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    AudioSpeechListResponse,
    AudioSpeechRequest,
    AudioSpeechResponse,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.storage import cloud_storage
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AUDIO_STORE
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    add_common_data_to_response,
    build_sampling_params,
    process_generation_batch,
    temp_dir_if_disabled,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import async_scheduler_client
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.observability.trace import extract_trace_headers

logger = init_logger(__name__)
router = APIRouter(prefix="/v1/audio", tags=["audio"])

OPENAI_SPEECH_FORMATS = ("mp3", "opus", "aac", "flac", "wav", "pcm")
FFMPEG_ONLY_FORMATS = ("mp3", "opus", "aac")
FFMPEG_TIMEOUT_S = 60.0
PROMPT_AUDIO_MAX_BYTES = 32 * 1024 * 1024
PROMPT_AUDIO_LOCAL_PATH_MSG = (
    "HTTP serving does not accept local filesystem paths for "
    "prompt_audio_path. Upload clone audio via multipart field "
    "'prompt_audio', or pass an http(s) URL. Local paths are only "
    "supported on the CLI."
)
MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _get_extra_field(request: AudioSpeechRequest, field_name: str) -> Any:
    extra = request.model_extra or {}
    value = extra.get(field_name)
    if value is not None:
        return value
    extra_body = extra.get("extra_body")
    if isinstance(extra_body, dict):
        return extra_body.get(field_name)
    return None


def resolve_speech_text(req: AudioSpeechRequest) -> str:
    text = req.input if req.input is not None else req.prompt
    if extra := _get_extra_field(req, "input"):
        text = text or extra
    if extra := _get_extra_field(req, "prompt"):
        text = text or extra
    if not text or not str(text).strip():
        raise _bad_request("Field 'input' is required")
    return str(text)


def normalize_response_format(fmt: Optional[str]) -> str:
    value = (fmt or "wav").lower()
    if value == "wave":
        value = "wav"
    if value not in OPENAI_SPEECH_FORMATS:
        raise _bad_request(
            f"response_format must be one of {list(OPENAI_SPEECH_FORMATS)}, got {fmt!r}"
        )
    return value


def normalize_speed(speed: Optional[float]) -> float:
    value = 1.0 if speed is None else float(speed)
    if not 0.25 <= value <= 4.0:
        raise _bad_request("speed must be between 0.25 and 4.0")
    return value


def normalize_duration_seconds(duration_seconds: Optional[float]) -> Optional[float]:
    if duration_seconds is None:
        return None
    try:
        from sglang.multimodal_gen.configs.sample.longcat_audiodit import (
            _require_positive_duration_seconds,
        )

        _require_positive_duration_seconds(duration_seconds)
    except ValueError as e:
        raise _bad_request(str(e)) from e
    return float(duration_seconds)


def require_audio_model(server_args) -> None:
    task_type = server_args.pipeline_config.task_type
    if not task_type.is_audio_gen():
        raise HTTPException(
            status_code=400,
            detail=(
                "POST /v1/audio/speech requires an audio generation model "
                f"(got task_type={task_type.name})"
            ),
        )


def _voice_label(voice: Any) -> Optional[str]:
    if isinstance(voice, str):
        return voice
    if isinstance(voice, dict):
        voice_id = voice.get("id")
        return str(voice_id) if voice_id is not None else None
    return None


def encode_speech_audio(
    wav_path: str,
    *,
    response_format: str,
    speed: float,
    sample_rate: Optional[int] = None,
) -> tuple[bytes, str]:
    """Convert a saved WAV to the OpenAI response_format. Returns (bytes, media_type)."""
    import soundfile as sf

    fmt = normalize_response_format(response_format)
    speed = normalize_speed(speed)
    audio, sr = sf.read(wav_path, dtype="float32")
    if sample_rate:
        sr = int(sample_rate)

    if speed != 1.0:
        try:
            import librosa
        except ImportError as e:
            raise _bad_request(
                "speed != 1.0 requires librosa; install sglang[diffusion] "
                "or set speed=1.0"
            ) from e

        if audio.ndim == 1:
            audio = librosa.effects.time_stretch(audio, rate=speed)
        else:
            stretched = [
                librosa.effects.time_stretch(audio[:, ch], rate=speed)
                for ch in range(audio.shape[1])
            ]
            audio = np.stack(stretched, axis=1)

    if fmt == "pcm":
        pcm = np.clip(audio, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype("<i2")
        return pcm.tobytes(), MEDIA_TYPES["pcm"]

    if fmt == "wav" and speed == 1.0:
        with open(wav_path, "rb") as f:
            return f.read(), MEDIA_TYPES["wav"]

    if fmt in ("wav", "flac"):
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format=fmt.upper())
        return buf.getvalue(), MEDIA_TYPES[fmt]

    if fmt in FFMPEG_ONLY_FORMATS:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise _bad_request(
                f"response_format={fmt!r} requires ffmpeg on PATH; "
                "use wav, flac, or pcm"
            )
        with tempfile.TemporaryDirectory(prefix="sglang_tts_") as tmp:
            src = os.path.join(tmp, "input.wav")
            dst = os.path.join(tmp, f"output.{fmt}")
            sf.write(src, audio, sr, format="WAV")
            codec = {"mp3": "libmp3lame", "opus": "libopus", "aac": "aac"}[fmt]
            cmd = [ffmpeg, "-y", "-i", src, "-c:a", codec, dst]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=FFMPEG_TIMEOUT_S,
                )
            except subprocess.TimeoutExpired as e:
                raise _bad_request(
                    f"ffmpeg timed out encoding {fmt} after {FFMPEG_TIMEOUT_S:.0f}s"
                ) from e
            except subprocess.CalledProcessError as e:
                stderr = e.stderr or ""
                raise _bad_request(
                    f"ffmpeg failed to encode {fmt}: {stderr[-400:]}"
                ) from e
            with open(dst, "rb") as f:
                return f.read(), MEDIA_TYPES[fmt]

    raise _bad_request(f"Unsupported response_format {fmt!r}")


def _is_http_url(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


async def _download_prompt_audio(url: str, target_path: str) -> str:
    import httpx

    os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
    if not os.path.splitext(target_path)[1]:
        target_path = f"{target_path}.wav"
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url, timeout=30.0) as response:
                response.raise_for_status()
                total = 0
                with open(target_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > PROMPT_AUDIO_MAX_BYTES:
                            raise _bad_request(
                                "prompt_audio_path download exceeds "
                                f"{PROMPT_AUDIO_MAX_BYTES} bytes"
                            )
                        f.write(chunk)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as e:
        raise _bad_request(
            f"Failed to download prompt_audio_path: HTTP {e.response.status_code}"
        ) from e
    except httpx.RequestError as e:
        raise _bad_request(f"Failed to download prompt_audio_path: {e}") from e
    return target_path


async def _save_prompt_audio(
    source: Union[UploadFile, bytes, str],
    target_path: str,
) -> str:
    from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
        _save_upload_to_path,
    )

    if isinstance(source, str):
        if _is_http_url(source):
            return await _download_prompt_audio(source, target_path)
        raise _bad_request(PROMPT_AUDIO_LOCAL_PATH_MSG)
    return await _save_upload_to_path(source, target_path)


async def _resolve_prompt_audio_ref(
    source: Optional[str], target_path: str
) -> Optional[str]:
    if not source:
        return None
    if _is_http_url(source):
        return await _download_prompt_audio(source, target_path)
    raise _bad_request(PROMPT_AUDIO_LOCAL_PATH_MSG)


def _form_optional(form, key: str) -> Any:
    value = form.get(key)
    return None if value in (None, "") else value


def _form_float(form, key: str) -> Optional[float]:
    value = _form_optional(form, key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise _bad_request(f"Field {key!r} must be a number") from e


def _form_int(form, key: str) -> Optional[int]:
    value = _form_optional(form, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise _bad_request(f"Field {key!r} must be an integer") from e


def _safe_upload_filename(filename: Optional[str], default: str = "prompt.wav") -> str:
    """Keep upload names inside the destination directory."""
    name = os.path.basename(filename or "")
    return default if name in ("", ".", "..") else name


async def parse_speech_request(
    request: Request, request_id: str
) -> tuple[AudioSpeechRequest, Optional[str]]:
    content_type = request.headers.get("content-type", "").lower()
    uploads_dir = os.path.join("outputs", "uploads")

    if "multipart/form-data" in content_type:
        form = await request.form()
        req = AudioSpeechRequest(
            input=_form_optional(form, "input"),
            prompt=_form_optional(form, "prompt"),
            model=_form_optional(form, "model"),
            voice=_form_optional(form, "voice") or "alloy",
            instructions=_form_optional(form, "instructions"),
            response_format=_form_optional(form, "response_format") or "wav",
            speed=_form_float(form, "speed"),
            stream_format=_form_optional(form, "stream_format") or "audio",
            prompt_text=_form_optional(form, "prompt_text"),
            prompt_audio_path=_form_optional(form, "prompt_audio_path"),
            guidance_method=_form_optional(form, "guidance_method"),
            guidance_scale=_form_float(form, "guidance_scale"),
            num_inference_steps=_form_int(form, "num_inference_steps"),
            duration_seconds=_form_float(form, "duration_seconds"),
            seed=_form_int(form, "seed"),
            generator_device=_form_optional(form, "generator_device"),
        )
        prompt_audio = form.get("prompt_audio")
        if prompt_audio is not None and hasattr(prompt_audio, "read"):
            os.makedirs(uploads_dir, exist_ok=True)
            filename = _safe_upload_filename(getattr(prompt_audio, "filename", None))
            saved = await _save_prompt_audio(
                prompt_audio,
                os.path.join(uploads_dir, f"{request_id}_{filename}"),
            )
            return req, saved
        extra_path = req.prompt_audio_path or _get_extra_field(req, "prompt_audio_path")
        return req, await _resolve_prompt_audio_ref(
            extra_path,
            os.path.join(uploads_dir, f"{request_id}_prompt"),
        )

    try:
        body = await request.json()
    except Exception as e:
        raise _bad_request("JSON object expected") from e
    if not isinstance(body, dict):
        raise _bad_request("JSON object expected")
    try:
        req = AudioSpeechRequest(**body)
    except Exception as e:
        raise _bad_request(f"Invalid request body: {e}") from e

    extra_path = req.prompt_audio_path or _get_extra_field(req, "prompt_audio_path")
    prompt_audio_path = await _resolve_prompt_audio_ref(
        extra_path,
        os.path.join(uploads_dir, f"{request_id}_prompt"),
    )
    return req, prompt_audio_path


def _sampling_kwargs_from_speech_request(
    request_id: str,
    req: AudioSpeechRequest,
    text: str,
    prompt_audio_path: Optional[str],
    output_dir: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prompt": text,
        "output_file_name": f"{request_id}.wav",
        "output_path": output_dir,
        "save_output": True,
        "seed": req.seed,
        "prompt_text": req.prompt_text or _get_extra_field(req, "prompt_text"),
        "prompt_audio_path": prompt_audio_path,
        "guidance_method": req.guidance_method
        or _get_extra_field(req, "guidance_method"),
        "guidance_scale": (
            req.guidance_scale
            if req.guidance_scale is not None
            else _get_extra_field(req, "guidance_scale")
        ),
        "num_inference_steps": (
            req.num_inference_steps
            if req.num_inference_steps is not None
            else _get_extra_field(req, "num_inference_steps")
        ),
        "duration_seconds": normalize_duration_seconds(
            req.duration_seconds
            if req.duration_seconds is not None
            else _get_extra_field(req, "duration_seconds")
        ),
    }
    if req.generator_device is not None:
        kwargs["generator_device"] = req.generator_device
    return kwargs


@router.post("/speech")
async def create_speech(request: Request):
    server_args = get_global_server_args()
    require_audio_model(server_args)
    request_id = generate_request_id()

    req, uploaded_prompt_audio = await parse_speech_request(request, request_id)

    stream_fmt = (req.stream_format or "audio").lower()
    if stream_fmt == "sse":
        raise _bad_request("stream_format='sse' is not supported")
    if stream_fmt not in ("audio",):
        raise _bad_request("stream_format must be 'audio' or 'sse'")

    text = resolve_speech_text(req)
    fmt = normalize_response_format(req.response_format)
    speed_value = normalize_speed(req.speed)
    # Cloning uses the dedicated SGLang fields (multipart upload or
    # http(s) prompt_audio_path). Local filesystem paths are CLI-only.
    # OpenAI ``voice`` is a label (alloy / Voices id), never a filesystem path.
    prompt_audio_file = uploaded_prompt_audio

    with temp_dir_if_disabled(server_args.output_path) as output_dir:
        sampling = build_sampling_params(
            request_id,
            **_sampling_kwargs_from_speech_request(
                request_id, req, text, prompt_audio_file, output_dir
            ),
        )
        trace_headers = extract_trace_headers(request.headers)
        batch = prepare_request(
            server_args=server_args,
            sampling_params=sampling,
            external_trace_header=trace_headers,
        )
        save_file_path_list, result = await process_generation_batch(
            async_scheduler_client, batch
        )
        if not save_file_path_list or not save_file_path_list[0]:
            raise RuntimeError("Audio generation returned no output file")
        wav_path = save_file_path_list[0]
        audio_bytes, media_type = encode_speech_audio(
            wav_path,
            response_format=fmt,
            speed=speed_value,
            sample_rate=result.audio_sample_rate,
        )

        is_persistent = server_args.output_path is not None
        stored_path = wav_path
        if fmt != "wav":
            encoded_path = os.path.splitext(wav_path)[0] + f".{fmt}"
            with open(encoded_path, "wb") as f:
                f.write(audio_bytes)
            stored_path = encoded_path
        cloud_url = await cloud_storage.upload_and_cleanup(stored_path)
        if cloud_url or not is_persistent:
            stored_path = None

        job: Dict[str, Any] = {
            "id": request_id,
            "object": "audio.speech",
            "model": req.model or (server_args.model_path or ""),
            "status": "completed",
            "created_at": int(time.time()),
            "completed_at": int(time.time()),
            "response_format": fmt,
            "voice": _voice_label(req.voice),
            "url": cloud_url,
            "file_path": stored_path,
            "file_size_bytes": len(audio_bytes),
            "sample_rate": result.audio_sample_rate,
        }
        job = add_common_data_to_response(job, request_id=request_id, result=result)
        await AUDIO_STORE.upsert(request_id, job)

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{request_id}.{fmt}"',
        },
    )


@router.get("/speech", response_model=AudioSpeechListResponse)
async def list_speech(
    after: Optional[str] = Query(None),
    limit: Optional[int] = Query(None, ge=1, le=100),
    order: Optional[str] = Query("desc"),
):
    jobs = await AUDIO_STORE.list_page(after=after, limit=limit, order=order)
    return AudioSpeechListResponse(data=[AudioSpeechResponse(**job) for job in jobs])


@router.get("/speech/{speech_id}", response_model=AudioSpeechResponse)
async def retrieve_speech(speech_id: str = Path(...)):
    job = await AUDIO_STORE.get(speech_id)
    if not job:
        raise HTTPException(status_code=404, detail="Speech not found")
    return AudioSpeechResponse(**job)


@router.delete("/speech/{speech_id}", response_model=AudioSpeechResponse)
async def delete_speech(speech_id: str = Path(...)):
    job = await AUDIO_STORE.pop(speech_id)
    if not job:
        raise HTTPException(status_code=404, detail="Speech not found")
    job["status"] = "deleted"
    return AudioSpeechResponse(**job)


@router.get("/speech/{speech_id}/content")
async def download_speech_content(speech_id: str = Path(...)):
    job = await AUDIO_STORE.get(speech_id)
    if not job:
        raise HTTPException(status_code=404, detail="Speech not found")
    if job.get("url"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Speech has been uploaded to cloud storage. "
                f"Please use the cloud URL: {job.get('url')}"
            ),
        )
    file_path = job.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=(
                "Speech was not persisted on disk (output_path is disabled) "
                "or is still being generated."
            ),
        )
    fmt = (job.get("response_format") or "wav").lower()
    return FileResponse(
        path=file_path,
        media_type=MEDIA_TYPES.get(fmt, "application/octet-stream"),
        filename=os.path.basename(file_path),
    )
