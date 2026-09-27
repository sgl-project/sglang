# SPDX-License-Identifier: Apache-2.0
"""Native FL2VA real audio-video parity and full shared-weight immutability."""

import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest
import requests
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
)
from sglang.srt.utils.network import get_free_port
from sglang.srt.weight_cache.common.liveness import ProcessIdentity

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


@pytest.mark.parametrize("cache_text_encoder", [False, True], ids=["dit", "dit-te"])
def test_minimax_h3_weight_cache_recovery(tmp_path, cache_text_encoder):
    model = maybe_download_model(
        os.environ.get(
            "SGLANG_TEST_WEIGHT_CACHE_MINIMAX_MODEL", "MiniMaxAI/MiniMax-H3"
        ),
        allow_patterns=["FL2VA/**"],
        revision="42ed227ee7df40d41602854ae760620d6eb651fe",
    )
    component_names = (
        ["transformer", "text_encoder"] if cache_text_encoder else ["transformer"]
    )
    common = MINIMAX_FLAGS
    if cache_text_encoder:
        common = common.replace(
            "text_encoder=layerwise-offload", "text_encoder=resident"
        )
        common += " --weight-cache-components dit text_encoder --weight-cache-max-deliveries 2"
    # Isolate this suite from other GPU tests' default rendezvous port.
    flags = f"{common} --warmup-mode off --master-port {get_free_port()}"

    def checksums(context):
        response = requests.post(
            f"http://127.0.0.1:{context.port}/get_weights_checksum",
            json={"module_names": component_names},
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
        assert set(result) == set(component_names)
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in result.values())
        return result

    guard_env = {}
    events = tmp_path / "guard"
    if cache_text_encoder:
        events.mkdir()
        fixtures = (
            Path(__file__).resolve().parents[5] / "test/manual/weight_cache_read_guard"
        )
        shards = [
            str(path)
            for name in component_names
            for path in (Path(model) / "FL2VA" / name).glob("*.safetensors")
        ]
        assert all(
            any(f"/{name}/" in path for path in shards) for name in component_names
        )
        guard_env = {
            "PYTHONPATH": str(fixtures) + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "WC_TEST_BLOCKED_FILES": json.dumps(shards),
            "WC_TEST_GUARD_LOG_DIR": str(events),
        }
        # Prove the guard rejects an actual tensor retrieval for BOTH components.
        positive = tmp_path / "positive-control"
        positive.mkdir()
        for name in component_names:
            shard = next(path for path in shards if f"/{name}/" in path)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import sys; from safetensors import safe_open\nwith safe_open(sys.argv[1], framework='pt') as f: f.get_tensor(next(iter(f.keys())))",
                    shard,
                ],
                env={**os.environ, **guard_env, "WC_TEST_GUARD_LOG_DIR": str(positive)},
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert (
                result.returncode != 0 and "WC_TEST_CACHED_TENSOR_READ" in result.stderr
            )
    references = {}
    ordinary = TimedServerManager(model, get_free_port(), extra_args=flags).start()
    try:
        checksum = checksums(ordinary)
        for steps in (4, 8):
            content = generate_minimax_h3(
                ordinary, model, f"h3-reference-{steps}", steps=steps
            )
            references[steps] = content
            (tmp_path / f"reference-{steps}.mp4").write_bytes(content)
        assert checksums(ordinary) == checksum
        (tmp_path / "reference-checksums.json").write_text(
            json.dumps(checksum, indent=2)
        )
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
                    model, socket_path, env, log, extra_args=shlex.split(common)
                )
                manager = TimedServerManager(
                    model,
                    get_free_port(),
                    env_vars={**env, **guard_env},
                    extra_args=f"{flags} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                )
                context = manager.start()
                assert checksums(context) == checksum
                _assert_mutations_rejected(context)
                for steps in (4, 8):
                    content = generate_minimax_h3(
                        context, model, f"h3-cached-{steps}", steps=steps
                    )
                    assert content == references[steps]
                    (tmp_path / f"cached-{steps}.mp4").write_bytes(content)
                    # Includes rope.inv_freq, re-registered as a parameter by
                    # the ordinary loader. The lazy timestep buffer is local.
                    assert checksums(context) == checksum
                text = context.stdout_file.read_text()
                (tmp_path / "cached.log").write_text(text)
                assert "Using module transformer already provided" in text
                assert "[ComponentLoader] transformer materialized" not in text
                imported_components = re.escape(",".join(component_names))
                assert (
                    len(
                        re.findall(
                            rf"\[WeightCache\] {imported_components} imported in ([\d.]+)s",
                            text,
                        )
                    )
                    == 1
                )
                if cache_text_encoder:
                    assert "Using module text_encoder already provided" in text
                    assert "[ComponentLoader] text_encoder materialized" not in text
                    # A second worker consumes one whole bundle reservation, not
                    # one reservation per component, and must remain disk-free.
                    context.cleanup()
                    context.process.wait(timeout=20)
                    context = None
                    context = TimedServerManager(
                        model,
                        get_free_port(),
                        env_vars={**env, **guard_env},
                        extra_args=f"{common} --warmup-mode off --master-port {get_free_port()} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                    ).start()
                    assert checksums(context) == checksum
                    recovered = generate_minimax_h3(
                        context, model, "h3-recovered", steps=4
                    )
                    assert recovered == references[4]
                    (tmp_path / "recovered-4.mp4").write_bytes(recovered)
                    assert checksums(context) == checksum
                    (tmp_path / "recovered.log").write_text(
                        context.stdout_file.read_text()
                    )
                    command = [
                        sys.executable,
                        "-m",
                        "sglang.multimodal_gen.runtime.weight_cache.daemon",
                        "--status",
                        "--model-path",
                        model,
                        "--weight-cache-socket",
                        str(socket_path),
                        *shlex.split(common),
                    ]
                    output = subprocess.check_output(
                        command, env={**os.environ, **env}, text=True, timeout=90
                    )
                    status = json.loads(output.strip().splitlines()[-1])
                    (tmp_path / "status.json").write_text(json.dumps(status, indent=2))
                    assert status["components"] == component_names
                    assert status["fetches_remaining"] == 0
                    assert status["active_consumers"] == 1
                    accesses = [
                        json.loads(line)
                        for path in events.glob("*.jsonl")
                        for line in path.read_text().splitlines()
                    ]
                    assert any(event["event"] == "installed" for event in accesses)
                    assert any(
                        event["event"] == "uncached_tensor" for event in accesses
                    )
                    assert not any(
                        event["event"] == "blocked_tensor" for event in accesses
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
                    model, socket_path, env, log, extra_args=shlex.split(common)
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
