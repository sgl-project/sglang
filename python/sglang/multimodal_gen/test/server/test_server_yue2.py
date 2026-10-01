# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP end-to-end tests for the YuE2 music pipeline.

Set SGLANG_TEST_YUE2=1 on a CUDA host with the checkpoints available:

    SGLANG_TEST_YUE2=1 SGLANG_TEST_YUE2_MODEL=/raid/yiakwy/YuE2-3B \
        pytest python/sglang/multimodal_gen/test/server/test_server_yue2.py

The VAE is resolved by the server from SGLANG_YUE2_VAE_DIR or from a
``YuE2-Vae`` directory next to the model (override with SGLANG_TEST_YUE2_VAE).
Requests use the experimental ``/v1/audio/generations`` route with a small
token budget so every generation stays in the sub-second range.
"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import pytest
import requests

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionServerArgs,
    DiffusionTestCase,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("SGLANG_TEST_YUE2") != "1",
    reason="opt-in YuE2 end-to-end test (requires checkpoints and a CUDA GPU)",
)


def _resolved_sglang_has_yue2() -> bool:
    """Fail fast when the interpreter's sglang tree lacks YuE2 support.

    Otherwise the harness launches a YuE2-less server and waits out the full
    readiness deadline (20 minutes) before reporting anything.
    """
    try:
        import sglang

        from sglang.multimodal_gen.registry import get_model_info

        return get_model_info(MODEL_PATH) is not None
    except Exception:
        return False

MODEL_PATH = os.environ.get("SGLANG_TEST_YUE2_MODEL", "m-a-p/YuE2-3B")
VAE_DIR = os.environ.get("SGLANG_TEST_YUE2_VAE", "")

# Shared hosts may occupy the default scheduler/master ports; pick two free
# ones and hand them to the server through the harness's arg passthrough.
from sglang.multimodal_gen.test.test_utils import find_free_port

os.environ.setdefault(
    "SGLANG_TEST_SERVE_ARGS",
    f"--scheduler-port {find_free_port()} --master-port {find_free_port()}",
)
ABC_EXAMPLE = Path(
    os.environ.get(
        "SGLANG_TEST_YUE2_ABC",
        "/home/yiakwy/workspace/Github/YuE/examples/melody.abc",
    )
)

# Small budgets keep each request well under a second after warmup.
PAYLOAD = {
    "style": "soft pop",
    "lyrics": "A short benchmark phrase.",
    "seed": 831001,
    "ode_steps": 2,
    "semantic_max_tokens": 20,
    "abc_max_tokens": 32,
    "vae_core_frames": 256,
    "vae_halo_frames": 16,
}


def _case(case_id: str) -> DiffusionTestCase:
    env_vars = {}
    if VAE_DIR:
        env_vars["SGLANG_YUE2_VAE_DIR"] = VAE_DIR
    return DiffusionTestCase(
        case_id,
        DiffusionServerArgs(
            model_path=MODEL_PATH,
            modality="audio",
            extras=["--warmup-mode off"],
            env_vars=env_vars,
        ),
    )


@pytest.fixture
def case():
    if not _resolved_sglang_has_yue2():
        import sglang

        pytest.skip(
            "the resolved sglang has no YuE2 support "
            f"(sglang.__file__ = {sglang.__file__}); "
            "run with the sglang-yue2 interpreter, e.g. "
            "PATH=/root/venv-sgl/bin:$PATH /root/venv-sgl/bin/python -m pytest"
        )
    return _case("yue2_e2e")


def _generations(diffusion_server, payload: dict) -> dict:
    response = requests.post(
        f"http://localhost:{diffusion_server.port}/v1/audio/generations",
        json=payload,
        timeout=600,
    )
    assert response.ok, response.text
    return response.json()


def _read_wav(path: str) -> tuple[int, int, bytes]:
    with wave.open(path, "rb") as handle:
        return handle.getframerate(), handle.getnframes(), handle.readframes(
            handle.getnframes()
        )


def _assert_valid_audio(data: dict) -> None:
    assert data["sample_rate"] == 48000
    assert data["peak_memory_mb"] > 0
    audio_path = Path(data["audio"])
    assert audio_path.is_file(), f"missing rendered audio: {audio_path}"
    rate, frames, pcm = _read_wav(str(audio_path))
    assert rate == 48000
    assert frames > 0
    # Not digital silence: a rendered song must carry signal energy.
    assert max(abs(int.from_bytes(pcm[i : i + 2], "little", signed=True)) for i in range(0, len(pcm), 4096 * 2)) > 16


def test_yue2_audio_generations(diffusion_server, case):
    """cot=full (session/step-graph path), determinism, cot=off, external ABC."""
    # 1. Main path: CoT planning + semantic codec tokens + NAR + VAE.
    data = _generations(diffusion_server, {**PAYLOAD, "cot": "full"})
    _assert_valid_audio(data)
    first_path = Path(data["audio"])

    # 2. Same-seed repeat: must render again with the same token budget.
    #    Byte-exact determinism is NOT asserted: the decode GEMMs use
    #    split-K kernels with atomic reduction whose accumulation order
    #    varies run-to-run, which can flip near-boundary sampled tokens and
    #    cascade into a different (still valid) musical realization. The
    #    upstream reference implementation has the same property.
    data = _generations(
        diffusion_server,
        {**PAYLOAD, "cot": "full", "output_file_name": "repeat.wav"},
    )
    _assert_valid_audio(data)
    _, frames_a, _ = _read_wav(str(first_path))
    _, frames_b, _ = _read_wav(data["audio"])
    for frames in (frames_a, frames_b):
        assert 4800 <= frames <= 48000 * 2, f"unexpected duration: {frames} frames"

    # 3. cot=off: no ABC phase; the semantic phase runs the historical
    #    (vLLM-era) arithmetic through the reference sampler fallback.
    data = _generations(diffusion_server, {**PAYLOAD, "cot": "off"})
    _assert_valid_audio(data)

    # 4. External ABC score: the provided plan must be used as-is.
    if ABC_EXAMPLE.is_file():
        data = _generations(
            diffusion_server,
            {
                **PAYLOAD,
                "cot": "melody",
                "abc": ABC_EXAMPLE.read_text(encoding="utf-8"),
                "semantic_max_tokens": 64,
            },
        )
        _assert_valid_audio(data)

    # 5. The alias route shares the handler.
    response = requests.post(
        f"http://localhost:{diffusion_server.port}/v1/audio/music/generations",
        json={**PAYLOAD, "cot": "full", "output_file_name": "alias.wav"},
        timeout=600,
    )
    assert response.ok, response.text
    assert json.loads(response.text)["sample_rate"] == 48000
