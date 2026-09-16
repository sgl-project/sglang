# SPDX-License-Identifier: Apache-2.0
"""Native FL2VA real audio-video parity and full shared-weight immutability."""

import hashlib
import json
import os
import re
import shlex
import signal
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest
from openai import OpenAI

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.test.server.test_server_utils import get_generate_fn
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionSamplingParams
from sglang.multimodal_gen.test.single_test_file.test_weight_cache_1_gpu import (
    TimedServerManager,
    _assert_mutations_rejected,
    _start_owner,
    _stop_owner,
    _weights_checksum,
)
from sglang.srt.utils.network import get_free_port
from sglang.weight_cache_common.liveness import ProcessIdentity

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA cache adapter"
)

MINIMAX_FLAGS = (
    "--model-variant fl2va --num-gpus 1 --performance-mode manual --attention-backend fa "
    "--use-fsdp-inference false "
    "--component-residency transformer=resident text_encoder=layerwise-offload vae=component-offload"
)


def generate_minimax_h3(context, model, case, *, steps=4):
    params = DiffusionSamplingParams(
        prompt="A small boat sails across a calm lake at sunrise, with gentle water sounds.",
        output_size="1344x768",
        seconds=4,
        output_format="mp4",
        expect_audio_output=True,
        extras={
            "task": "t2va",
            "conditions": [],
            "target": {
                "short_edge": 768,
                "aspect_ratio": "16:9",
                "duration_seconds": 4.0,
            },
            "num_inference_steps": steps,
            "flow_shift": 12.0,
            "audio_flow_shift": 3.0,
            "seed": 42,
        },
    )
    generate = get_generate_fn(model, "video", params)
    with (
        OpenAI(
            base_url=f"http://127.0.0.1:{context.port}/v1",
            api_key="EMPTY",
            timeout=240,
            max_retries=0,
        ) as client,
        patch(
            "sglang.multimodal_gen.test.server.test_server_utils.upload_file_to_slack",
            return_value=False,
        ),
    ):
        # Existing helper validates video and audio streams, frames and AV sync.
        _, content = generate(case, client)
    return content


def test_minimax_h3_weight_cache_recovery(tmp_path):
    model = maybe_download_model(
        os.environ.get(
            "SGLANG_WEIGHT_CACHE_MINIMAX_TEST_MODEL", "MiniMaxAI/MiniMax-H3"
        ),
        allow_patterns=["FL2VA/**"],
        revision="42ed227ee7df40d41602854ae760620d6eb651fe",
    )
    flags = f"{MINIMAX_FLAGS} --warmup-mode off"
    references = {}
    ordinary = TimedServerManager(model, get_free_port(), extra_args=flags).start()
    try:
        checksum = _weights_checksum(ordinary, timeout=240)
        for steps in (4, 8):
            content = generate_minimax_h3(
                ordinary, model, f"h3-reference-{steps}", steps=steps
            )
            references[steps] = content
            (tmp_path / f"reference-{steps}.mp4").write_bytes(content)
    finally:
        (tmp_path / "ordinary.log").write_text(ordinary.stdout_file.read_text())
        ordinary.cleanup()
        ordinary.process.wait(timeout=20)

    with tempfile.TemporaryDirectory(prefix="sgl-h3-wc-") as runtime:
        socket_path = Path(runtime) / "owner.sock"
        env = {"SGLANG_DIFFUSION_WEIGHT_CACHE_DIR": runtime}
        owner = context = None
        with (tmp_path / "owner.log").open("w") as log:
            try:
                owner = _start_owner(
                    model, socket_path, env, log, extra_args=shlex.split(MINIMAX_FLAGS)
                )
                manager = TimedServerManager(
                    model,
                    get_free_port(),
                    env_vars=env,
                    extra_args=f"{flags} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                )
                context = manager.start()
                assert _weights_checksum(context, timeout=240) == checksum
                _assert_mutations_rejected(context)
                for steps in (4, 8):
                    content = generate_minimax_h3(
                        context, model, f"h3-cached-{steps}", steps=steps
                    )
                    assert content == references[steps]
                    (tmp_path / f"cached-{steps}.mp4").write_bytes(content)
                    # Includes rope.inv_freq, re-registered as a parameter by
                    # the ordinary loader. The lazy timestep buffer is local.
                    assert _weights_checksum(context, timeout=240) == checksum
                text = context.stdout_file.read_text()
                (tmp_path / "cached.log").write_text(text)
                assert "Using module transformer already provided" in text
                assert "[ComponentLoader] transformer materialized" not in text
                assert (
                    len(
                        re.findall(
                            r"\[WeightCache\] transformer imported in ([\d.]+)s", text
                        )
                    )
                    == 1
                )
                workers = [
                    ProcessIdentity.read(p.pid)
                    for p in psutil.Process(context.process.pid).children(
                        recursive=True
                    )
                    if p.name().startswith("sgl_diffusion")
                ]
                assert len(workers) == 1
                owner.send_signal(signal.SIGKILL)
                deadline = time.monotonic() + 10
                while any(worker.is_alive() for worker in workers):
                    assert time.monotonic() < deadline, (
                        "Mapped H3 worker survived owner loss"
                    )
                    time.sleep(0.1)
                owner.wait(timeout=20)
                context.cleanup()
                context.process.wait(timeout=20)
                context = None
                # Existing stale-generation cleanup and normal owner shutdown.
                owner = _start_owner(
                    model, socket_path, env, log, extra_args=shlex.split(MINIMAX_FLAGS)
                )
                _stop_owner(owner)
                assert (
                    not socket_path.exists()
                    and not socket_path.with_suffix(".ready").exists()
                )
            finally:
                if context is not None:
                    (tmp_path / "last-client.log").write_text(
                        context.stdout_file.read_text()
                    )
                    context.cleanup()
                    context.process.wait(timeout=20)
                _stop_owner(owner)
    (tmp_path / "parity.json").write_text(
        json.dumps(
            {
                "weight_checksum": checksum,
                "videos": {
                    steps: hashlib.sha256(content).hexdigest()
                    for steps, content in references.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s", *sys.argv[1:]]))
