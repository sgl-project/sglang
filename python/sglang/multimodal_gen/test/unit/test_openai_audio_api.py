# SPDX-License-Identifier: Apache-2.0
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.runtime.entrypoints.openai.audio_api import (
    _safe_upload_filename,
    _sampling_kwargs_from_speech_request,
    encode_speech_audio,
    normalize_duration_seconds,
    normalize_response_format,
    normalize_speed,
    require_audio_model,
    resolve_speech_text,
    router,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import AudioSpeechRequest
from sglang.multimodal_gen.runtime.entrypoints.openai.stores import AUDIO_STORE
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch


class TestAudioSpeechProtocol(unittest.TestCase):
    def test_resolve_speech_text_prefers_openai_input(self):
        req = AudioSpeechRequest(input="hello", prompt="ignored")
        self.assertEqual(resolve_speech_text(req), "hello")

    def test_resolve_speech_text_accepts_prompt_alias(self):
        req = AudioSpeechRequest(prompt="from prompt")
        self.assertEqual(resolve_speech_text(req), "from prompt")

    def test_resolve_speech_text_requires_input(self):
        with self.assertRaises(HTTPException) as ctx:
            resolve_speech_text(AudioSpeechRequest())
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("input", ctx.exception.detail)

    def test_normalize_response_format_defaults_to_wav(self):
        self.assertEqual(normalize_response_format(None), "wav")
        self.assertEqual(normalize_response_format("WAV"), "wav")

    def test_normalize_response_format_rejects_unknown(self):
        with self.assertRaises(HTTPException) as ctx:
            normalize_response_format("ogg")
        self.assertEqual(ctx.exception.status_code, 400)

    def test_normalize_speed_bounds(self):
        self.assertEqual(normalize_speed(None), 1.0)
        self.assertEqual(normalize_speed(1.25), 1.25)
        with self.assertRaises(HTTPException) as ctx:
            normalize_speed(8.0)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_normalize_duration_seconds_rejects_non_positive(self):
        self.assertIsNone(normalize_duration_seconds(None))
        self.assertEqual(normalize_duration_seconds(2.5), 2.5)
        with self.assertRaises(HTTPException) as ctx:
            normalize_duration_seconds(0)
        self.assertEqual(ctx.exception.status_code, 400)
        with self.assertRaises(HTTPException) as ctx:
            normalize_duration_seconds(-1.0)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_voice_object_is_label_not_prompt_audio_path(self):
        req = AudioSpeechRequest(input="hello", voice={"id": "/ref.wav"})
        self.assertIsNone(req.prompt_audio_path)
        self.assertEqual(req.voice, {"id": "/ref.wav"})

    def test_require_audio_model_rejects_image_task(self):
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(task_type=ModelTaskType.T2I)
        )
        with self.assertRaises(HTTPException) as ctx:
            require_audio_model(server_args)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("/v1/audio/speech", ctx.exception.detail)

    def test_require_audio_model_accepts_t2a(self):
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(task_type=ModelTaskType.T2A)
        )
        require_audio_model(server_args)

    def test_generator_device_defaults_to_none(self):
        req = AudioSpeechRequest(input="hello")
        self.assertIsNone(req.generator_device)
        kwargs = _sampling_kwargs_from_speech_request(
            "speech-1", req, "hello", None, "/tmp"
        )
        self.assertNotIn("generator_device", kwargs)
        req = AudioSpeechRequest(input="hello", generator_device="cpu")
        kwargs = _sampling_kwargs_from_speech_request(
            "speech-1", req, "hello", None, "/tmp"
        )
        self.assertEqual(kwargs["generator_device"], "cpu")

    def test_safe_upload_filename_strips_path_components(self):
        self.assertEqual(_safe_upload_filename("../../x.wav"), "x.wav")
        self.assertEqual(_safe_upload_filename("/etc/passwd"), "passwd")
        self.assertEqual(_safe_upload_filename(""), "prompt.wav")
        self.assertEqual(_safe_upload_filename(".."), "prompt.wav")
        self.assertEqual(_safe_upload_filename(None), "prompt.wav")


class TestEncodeSpeechAudio(unittest.TestCase):
    def test_wav_and_pcm(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = f"{tmp}/clip.wav"
            samples = np.zeros(2400, dtype=np.float32)
            sf.write(wav_path, samples, 24000)

            wav_bytes, wav_type = encode_speech_audio(
                wav_path, response_format="wav", speed=1.0
            )
            self.assertEqual(wav_type, "audio/wav")
            self.assertEqual(wav_bytes[:4], b"RIFF")

            pcm_bytes, pcm_type = encode_speech_audio(
                wav_path, response_format="pcm", speed=1.0
            )
            self.assertEqual(pcm_type, "audio/pcm")
            self.assertEqual(len(pcm_bytes), 2400 * 2)

    def test_ffmpeg_timeout_returns_400(self):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = f"{tmp}/clip.wav"
            sf.write(wav_path, np.zeros(2400, dtype=np.float32), 24000)
            with (
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.shutil.which",
                    return_value="ffmpeg",
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60),
                ) as mock_run,
            ):
                with self.assertRaises(HTTPException) as ctx:
                    encode_speech_audio(wav_path, response_format="mp3", speed=1.0)
            self.assertEqual(mock_run.call_args.kwargs["timeout"], 60.0)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("timed out", ctx.exception.detail)


def _audio_server_args():
    return SimpleNamespace(
        pipeline_config=SimpleNamespace(task_type=ModelTaskType.T2A),
        model_path="/models/LongCat-AudioDiT-1B",
        output_path=None,
    )


class TestAudioSpeechHttp(unittest.TestCase):
    def test_speech_endpoint_returns_wav_bytes(self):
        import tempfile

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = f"{tmp}/out.wav"
            sf.write(wav_path, np.zeros(800, dtype=np.float32), 24000)

            async def fake_forward(*_args, **_kwargs):
                return [wav_path], OutputBatch(audio_sample_rate=24000)

            with (
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.get_global_server_args",
                    return_value=_audio_server_args(),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.build_sampling_params",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.prepare_request",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.process_generation_batch",
                    new=AsyncMock(side_effect=fake_forward),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.extract_trace_headers",
                    return_value=None,
                ),
            ):
                response = client.post(
                    "/v1/audio/speech",
                    json={
                        "input": "hello",
                        "voice": "alloy",
                        "response_format": "wav",
                    },
                )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("audio/wav"))
        self.assertEqual(response.content[:4], b"RIFF")

    def test_voice_id_is_not_used_as_prompt_audio_path(self):
        import tempfile

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        captured = {}

        def fake_build(request_id, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = f"{tmp}/out.wav"
            sf.write(wav_path, np.zeros(800, dtype=np.float32), 24000)

            async def fake_forward(*_args, **_kwargs):
                return [wav_path], OutputBatch(audio_sample_rate=24000)

            with (
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.get_global_server_args",
                    return_value=_audio_server_args(),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.build_sampling_params",
                    side_effect=fake_build,
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.prepare_request",
                    return_value=SimpleNamespace(),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.process_generation_batch",
                    new=AsyncMock(side_effect=fake_forward),
                ),
                patch(
                    "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.extract_trace_headers",
                    return_value=None,
                ),
            ):
                response = client.post(
                    "/v1/audio/speech",
                    json={"input": "hello", "voice": {"id": "/ref.wav"}},
                )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(captured.get("prompt_audio_path"))

    def test_speech_endpoint_rejects_sse_stream_format(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.get_global_server_args",
            return_value=_audio_server_args(),
        ):
            response = client.post(
                "/v1/audio/speech",
                json={"input": "hello", "stream_format": "sse"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("sse", response.json()["detail"])

    def test_invalid_json_is_not_reported_as_missing_input(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.get_global_server_args",
            return_value=_audio_server_args(),
        ):
            response = client.post(
                "/v1/audio/speech",
                content=b"{not json",
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON object expected", response.json()["detail"])
        self.assertNotIn("input", response.json()["detail"])

    def test_local_prompt_audio_path_is_rejected(self):
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.audio_api.get_global_server_args",
            return_value=_audio_server_args(),
        ):
            response = client.post(
                "/v1/audio/speech",
                json={"input": "hello", "prompt_audio_path": "/etc/passwd"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("local filesystem", response.json()["detail"])

    def test_multipart_invalid_duration_is_400(self):
        from sglang.multimodal_gen.runtime.entrypoints.openai.audio_api import (
            _form_float,
        )

        with self.assertRaises(HTTPException) as ctx:
            _form_float({"duration_seconds": "abc"}, "duration_seconds")
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("duration_seconds", ctx.exception.detail)

    def test_speed_without_librosa_is_400(self):
        import builtins
        import tempfile

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("missing librosa")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = f"{tmp}/clip.wav"
            sf.write(wav_path, np.zeros(2400, dtype=np.float32), 24000)
            with patch("builtins.__import__", side_effect=fake_import):
                with self.assertRaises(HTTPException) as ctx:
                    encode_speech_audio(wav_path, response_format="wav", speed=1.25)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("librosa", ctx.exception.detail)

    def test_speech_store_retrieve_and_content(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            wav_path = f"{tmp}/stored.wav"
            sf.write(wav_path, np.zeros(400, dtype=np.float32), 24000)
            job = {
                "id": "speech-1",
                "object": "audio.speech",
                "model": "longcat",
                "status": "completed",
                "created_at": 1,
                "response_format": "wav",
                "file_path": wav_path,
                "file_size_bytes": 1,
            }
            asyncio.run(AUDIO_STORE.upsert("speech-1", job))
            try:
                app = FastAPI()
                app.include_router(router)
                client = TestClient(app)

                meta = client.get("/v1/audio/speech/speech-1")
                self.assertEqual(meta.status_code, 200)
                self.assertEqual(meta.json()["id"], "speech-1")

                content = client.get("/v1/audio/speech/speech-1/content")
                self.assertEqual(content.status_code, 200)
                self.assertEqual(content.content[:4], b"RIFF")
            finally:
                asyncio.run(AUDIO_STORE.pop("speech-1"))


if __name__ == "__main__":
    unittest.main()
