"""End-to-end tests for LTX-2 dual-VAE DAG disaggregation.

Exercises fan-out from the denoiser to independently scaled video and audio
decoders, terminal output merging, and conditional pruning of the audio branch.

Uses a 2-GPU colocated layout (encoder + both VAE nodes on GPU 0, denoiser on
GPU 1) so it fits the multimodal ``2-gpu`` CI suite alongside Z-Image disagg.

Run directly:

    CUDA_VISIBLE_DEVICES=0,1 MC_FORCE_TCP=1 \\
        pytest -v python/sglang/multimodal_gen/test/single_test_file/test_disagg_ltx2_dag.py
    pytest -v ... -k DualVae
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

import requests

from sglang.multimodal_gen.test.single_test_file.test_disagg_server import (
    HOST,
    DisaggCluster,
    _DisaggTestBase,
)
_LTX23_MODEL = "Lightricks/LTX-2.3"
_TOPOLOGY_DIR = Path(__file__).resolve().parents[2] / "configs" / "disagg_topologies"

# Keep generation short: four nodes each load weights and this runs on CI GPUs.
_LTX2_TEST_VIDEO = {
    "prompt": "A cat walking on grass",
    "seconds": 1,
    "size": "512x320",
    "num_inference_steps": 4,
    "num_frames": 17,
}

_LTX2_DAG_ROLE_MAP = {
    "encoder": "encoder",
    "denoiser": "denoiser",
    "vae_video": "decoder",
    "vae_audio": "decoder",
}

# LTX encoder outputs are much larger than Z-Image; size the Mooncake pool
# accordingly for same-host multi-GPU e2e.
_LTX2_TRANSFER_POOL_BYTES = str(1024 * 1024 * 1024)
_LTX2_TRANSFER_ARGS = [
    "--disagg-transfer-pool-size",
    _LTX2_TRANSFER_POOL_BYTES,
]


def _generate_video(
    api_port: int,
    model: str,
    *,
    generate_audio: bool | None = None,
) -> bytes:
    payload = {"model": model, **_LTX2_TEST_VIDEO}
    if generate_audio is not None:
        payload["generate_audio"] = generate_audio

    resp = requests.post(
        f"http://{HOST}:{api_port}/v1/videos",
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    video_id = resp.json()["id"]

    deadline = time.time() + 900
    status = "queued"
    while time.time() < deadline:
        page = requests.get(
            f"http://{HOST}:{api_port}/v1/videos",
            timeout=30,
        )
        page.raise_for_status()
        item = next(
            (job for job in page.json()["data"] if job["id"] == video_id),
            None,
        )
        status = item.get("status") if item else None
        if status == "completed":
            break
        if status == "failed":
            detail = item.get("error") if item else "unknown"
            raise RuntimeError(f"video job {video_id} failed: {detail}")
        time.sleep(2)

    if status != "completed":
        raise RuntimeError(f"video job {video_id} timed out (last status={status!r})")

    content = requests.get(
        f"http://{HOST}:{api_port}/v1/videos/{video_id}/content",
        timeout=120,
    )
    if content.status_code != 200:
        raise RuntimeError(
            f"download failed ({content.status_code}): {content.text[:500]}"
        )
    data = content.content
    if len(data) < 1_000:
        raise RuntimeError(f"video too small: {len(data)} bytes")
    return data


def _mp4_has_audio(mp4_bytes: bytes) -> bool:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        path = tmp.name
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        os.unlink(path)
    return "audio" in proc.stdout


def _warmup_ltx2_video(cluster: DisaggCluster) -> None:
    _generate_video(cluster.api_port, cluster.model)


class TestDisaggLtx2DualVaeDag(_DisaggTestBase):
    """LTX-2.3 with video/audio VAEs on separate DAG terminal nodes."""

    cluster_name = "ltx2_dual_vae_2gpu"
    model = _LTX23_MODEL
    required_gpus = 2
    gpu_layout = {
        "encoder": [0],
        "denoiser": [1],
        "vae_video": [0],
        "vae_audio": [0],
    }
    dag_topology = _TOPOLOGY_DIR / "ltx2_dual_vae.yaml"
    disagg_role_map = _LTX2_DAG_ROLE_MAP
    warmup_fn = staticmethod(_warmup_ltx2_video)
    startup_timeout = 1800.0
    extra_role_args = {
        "encoder": list(_LTX2_TRANSFER_ARGS),
        "denoiser": [*_LTX2_TRANSFER_ARGS, "--dit-layerwise-offload", "true"],
        "vae_video": list(_LTX2_TRANSFER_ARGS),
        "vae_audio": list(_LTX2_TRANSFER_ARGS),
        "server": ["--disagg-timeout", "900"],
    }

    @classmethod
    def setUpClass(cls) -> None:
        os.environ.setdefault("MC_FORCE_TCP", "1")
        super().setUpClass()

    def test_generates_muxed_av_mp4(self) -> None:
        """Fan-out + dual terminals produce one mp4 with video and audio streams."""
        assert self.cluster is not None
        mp4 = _generate_video(self.cluster.api_port, self.model)
        self.assertGreater(len(mp4), 10_000, f"mp4 too small: {len(mp4)} bytes")
        self.assertTrue(
            _mp4_has_audio(mp4),
            "expected ffmpeg-muxed audio track from the vae_audio terminal",
        )

    def test_stages_are_split_by_plan(self) -> None:
        """Each decoder node runs only the stage the topology assigned it."""
        assert self.cluster is not None
        video_log = self.cluster._logs["vae_video"].read_text(errors="ignore")
        audio_log = self.cluster._logs["vae_audio"].read_text(errors="ignore")
        self.assertIn(
            "DAG node=vae_video: skipping stage LTX2AudioDecodingStage",
            video_log,
        )
        self.assertIn(
            "DAG node=vae_audio: skipping stage LTX2VideoDecodingStage",
            audio_log,
        )
        self.assertIn("['LTX2VideoDecodingStage']", video_log)
        self.assertIn("['LTX2AudioDecodingStage']", audio_log)
        fused = "LTX2AVDecodingStage"
        self.assertNotIn(fused, video_log)
        self.assertNotIn(fused, audio_log)

    def test_denoiser_fan_out_reaches_both_decoders(self) -> None:
        """The denoiser staged once and both terminal decoders consumed their slice."""
        assert self.cluster is not None
        denoiser_log = self.cluster._logs["denoiser"].read_text(errors="ignore")
        video_log = self.cluster._logs["vae_video"].read_text(errors="ignore")
        audio_log = self.cluster._logs["vae_audio"].read_text(errors="ignore")
        self.assertIn("LTX2AVDenoisingStage] finished", denoiser_log)
        self.assertIn("LTX2VideoDecodingStage", video_log)
        self.assertIn("LTX2AudioDecodingStage", audio_log)

    def test_skips_audio_branch_when_disabled(self) -> None:
        """Pruned edges still resolve, so a silent request completes without deadlock."""
        assert self.cluster is not None
        audio_log_before = self.cluster._logs["vae_audio"].read_text(errors="ignore")
        decode_runs_before = audio_log_before.count(
            "Running pipeline stages: ['LTX2AudioDecodingStage']"
        )

        mp4 = _generate_video(
            self.cluster.api_port,
            self.model,
            generate_audio=False,
        )
        self.assertFalse(
            _mp4_has_audio(mp4),
            "generate_audio=false should yield a silent mp4",
        )

        audio_log_after = self.cluster._logs["vae_audio"].read_text(errors="ignore")
        decode_runs_after = audio_log_after.count(
            "Running pipeline stages: ['LTX2AudioDecodingStage']"
        )
        self.assertEqual(
            decode_runs_after,
            decode_runs_before,
            "vae_audio must be skipped, not run, when generate_audio is false",
        )


if __name__ == "__main__":
    unittest.main()
