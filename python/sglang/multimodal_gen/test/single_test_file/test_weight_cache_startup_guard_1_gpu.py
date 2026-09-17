# SPDX-License-Identifier: Apache-2.0
"""Whole HTTP startup tensor-read guard and concurrent real Wan consumers."""

import json
import os
import shlex
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.test.single_test_file.test_weight_cache_1_gpu import (
    TimedServerManager,
    _assert_parity,
    _generate,
    _start_owner,
    _stop_owner,
    _weights_checksum,
)
from sglang.multimodal_gen.test.single_test_file.test_weight_cache_1_gpu import (
    pytestmark as pytestmark,
)
from sglang.srt.environ import envs
from sglang.srt.utils.network import get_free_port


def _events(directory):
    return [
        json.loads(line)
        for path in directory.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]


def test_complete_warm_start_has_no_cached_tensor_reads_and_two_consumers_are_immutable(
    tmp_path,
):
    model = maybe_download_model(
        envs.SGLANG_TEST_WEIGHT_CACHE_MODEL.get(),
        force_diffusers_model=True,
        revision="0fad780a534b6463e45facd96134c9f345acfa5b",
    )
    fixtures = (
        Path(__file__).resolve().parents[5] / "test/manual/weight_cache_read_guard"
    )
    assert (fixtures / "sitecustomize.py").exists()
    shards = sorted(
        str(path) for path in (Path(model) / "transformer").glob("*.safetensors")
    )
    assert shards
    guard_env = {
        "PYTHONPATH": str(fixtures) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        "WC_TEST_BLOCKED_FILES": json.dumps(shards),
    }
    positive = tmp_path / "positive-control"
    positive.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from safetensors import safe_open\nwith safe_open(sys.argv[1], framework='pt') as f: f.get_tensor(next(iter(f.keys())))",
            shards[0],
        ],
        env={**os.environ, **guard_env, "WC_TEST_GUARD_LOG_DIR": str(positive)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0 and "WC_TEST_CACHED_TENSOR_READ" in result.stderr
    assert any(event["event"] == "blocked_tensor" for event in _events(positive))

    contexts = []
    owner = None
    with tempfile.TemporaryDirectory(prefix="wc-guard-") as runtime:
        socket_path = Path(runtime) / "owner.sock"
        env = {"SGLANG_DIFFUSION_WEIGHT_CACHE_DIR": runtime}
        flags = "--num-gpus 1 --warmup-mode off --performance-mode manual --attention-backend fa"
        with (tmp_path / "owner.log").open("w") as log:
            try:
                ordinary = TimedServerManager(
                    model, get_free_port(), extra_args=flags, env_vars=env
                ).start()
                contexts.append(ordinary)
                checksum = _weights_checksum(ordinary)
                _, reference = _generate(ordinary, model, "guard-ordinary")
                ordinary.cleanup()
                ordinary.process.wait(timeout=20)
                contexts.clear()
                owner = _start_owner(
                    model,
                    socket_path,
                    env,
                    log,
                    extra_args=(
                        "--performance-mode",
                        "manual",
                        "--weight-cache-max-deliveries",
                        "2",
                    ),
                )
                command = [
                    sys.executable,
                    "-m",
                    "sglang.multimodal_gen.runtime.weight_cache.daemon",
                    "--status",
                    "--model-path",
                    model,
                    "--performance-mode",
                    "manual",
                    "--weight-cache-socket",
                    str(socket_path),
                ]

                def status():
                    output = subprocess.check_output(
                        command, env={**os.environ, **env}, text=True, timeout=60
                    )
                    return json.loads(output.strip().splitlines()[-1])

                before = status()
                assert (
                    before["fetches_remaining"] == 2 and before["active_consumers"] == 0
                )
                for index in range(2):
                    events = tmp_path / f"consumer-{index}"
                    events.mkdir()
                    context = TimedServerManager(
                        model,
                        get_free_port(),
                        extra_args=f"{flags} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                        env_vars={
                            **env,
                            **guard_env,
                            "WC_TEST_GUARD_LOG_DIR": str(events),
                        },
                    ).start()
                    contexts.append(context)
                    assert _weights_checksum(context) == checksum
                    workers = [
                        child.pid
                        for child in psutil.Process(context.process.pid).children(
                            recursive=True
                        )
                        if child.name().startswith("sgl_diffusion")
                    ]
                    assert len(workers) == 1
                    observed = _events(events)
                    installed = {
                        event["pid"]
                        for event in observed
                        if event["event"] == "installed"
                    }
                    assert context.process.pid in installed and workers[0] in installed
                    assert not any(
                        event["event"] == "blocked_tensor" for event in observed
                    )
                    assert any(
                        event["event"] == "uncached_tensor"
                        and event["pid"] == workers[0]
                        for event in observed
                    )
                exhausted = status()
                assert exhausted["fetches_remaining"] == 0
                assert exhausted["deliveries_reserved"] == 2
                assert exhausted["active_consumers"] == 2
                assert not exhausted["accepting_fetches"]
                # The helper also patches uploads per request. Keep an outer
                # guard while concurrent inner patches restore in any order.
                with (
                    patch(
                        "sglang.multimodal_gen.test.server.test_server_utils.upload_file_to_slack",
                        return_value=False,
                    ),
                    ThreadPoolExecutor(max_workers=2) as pool,
                ):
                    results = list(
                        pool.map(
                            lambda item: _generate(
                                item[1], model, f"guard-concurrent-{item[0]}"
                            ),
                            enumerate(contexts),
                        )
                    )
                for context, (_, frames) in zip(contexts, results):
                    _assert_parity(frames, reference)
                    assert _weights_checksum(context) == checksum
                # Exhaustion is observable, and a new whole HTTP startup fails
                # before worker launch rather than silently loading from disk.
                with pytest.raises(RuntimeError, match="budget exhausted"):
                    TimedServerManager(
                        model,
                        get_free_port(),
                        wait_deadline=90,
                        extra_args=f"{flags} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                        env_vars=env,
                    ).start()
                for index in range(2):
                    assert not any(
                        event["event"] == "blocked_tensor"
                        for event in _events(tmp_path / f"consumer-{index}")
                    )
                for context in contexts:
                    context.cleanup()
                    context.process.wait(timeout=20)
                contexts.clear()
                drained = status()
                assert drained["active_consumers"] == 0
                assert (
                    drained["deliveries_reserved"] == 2
                    and drained["fetches_remaining"] == 0
                )
                (tmp_path / "status.json").write_text(
                    json.dumps(
                        {
                            "before": before,
                            "exhausted": exhausted,
                            "clients_exited": drained,
                        },
                        indent=2,
                    )
                )
            finally:
                for context in contexts:
                    context.cleanup()
                    context.process.wait(timeout=20)
                _stop_owner(owner)
