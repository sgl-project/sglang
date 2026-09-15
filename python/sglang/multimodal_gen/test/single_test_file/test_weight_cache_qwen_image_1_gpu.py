# SPDX-License-Identifier: Apache-2.0
"""Real Qwen-Image shared-weight parity, immutability and owner fail-stop."""

import hashlib
import io
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
from PIL import Image

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


def generate_qwen_image(context, model, case, *, steps=4):
    params = DiffusionSamplingParams(
        output_size="1024x1024",
        output_format="png",
        prompt="A small boat sails across a calm lake at sunrise.",
        extras={"num_inference_steps": steps, "seed": 42},
    )
    generate = get_generate_fn(model, "image", params)
    with (
        OpenAI(
            base_url=f"http://127.0.0.1:{context.port}/v1",
            api_key="EMPTY",
            timeout=180,
            max_retries=0,
        ) as client,
        patch(
            "sglang.multimodal_gen.test.server.test_server_utils.upload_file_to_slack",
            return_value=False,
        ),
    ):
        _, content = generate(case, client)
    with Image.open(io.BytesIO(content)) as image:
        assert image.size == (1024, 1024)
        assert image.format == "PNG"
    return content


def test_qwen_image_weight_cache_recovery(tmp_path):
    model = maybe_download_model(
        os.environ.get("SGLANG_WEIGHT_CACHE_QWEN_TEST_MODEL", "Qwen/Qwen-Image"),
        force_diffusers_model=True,
        revision="75e0b4be04f60ec59a75f475837eced720f823b6",
    )
    flags = "--num-gpus 1 --attention-backend fa --warmup-mode off"
    references = {}
    ordinary = TimedServerManager(model, get_free_port(), extra_args=flags).start()
    try:
        checksum = _weights_checksum(ordinary, timeout=120)
        for steps in (4, 20):
            content = generate_qwen_image(
                ordinary, model, f"qwen-reference-{steps}", steps=steps
            )
            references[steps] = content
            (tmp_path / f"reference-{steps}.png").write_bytes(content)
    finally:
        (tmp_path / "ordinary.log").write_text(ordinary.stdout_file.read_text())
        ordinary.cleanup()
        ordinary.process.wait(timeout=20)

    with tempfile.TemporaryDirectory(prefix="sgl-qwen-wc-") as runtime:
        socket_path = Path(runtime) / "owner.sock"
        env = {"SGLANG_DIFFUSION_WEIGHT_CACHE_DIR": runtime}
        owner = context = None
        with (tmp_path / "owner.log").open("w") as log:
            try:
                owner = _start_owner(model, socket_path, env, log)
                for index, steps in enumerate((4, 20)):
                    manager = TimedServerManager(
                        model,
                        get_free_port(),
                        env_vars=env,
                        extra_args=f"{flags} --weight-cache-mode client --weight-cache-socket {shlex.quote(str(socket_path))}",
                    )
                    context = manager.start()
                    assert _weights_checksum(context, timeout=120) == checksum
                    _assert_mutations_rejected(context)
                    content = generate_qwen_image(
                        context, model, f"qwen-cached-{steps}", steps=steps
                    )
                    assert content == references[steps]
                    (tmp_path / f"cached-{steps}.png").write_bytes(content)
                    # Full finalized DiT checksum after actual inference, not
                    # merely identical images or zero extra weight VRAM.
                    assert _weights_checksum(context, timeout=120) == checksum
                    text = context.stdout_file.read_text()
                    (tmp_path / f"cached-{index}.log").write_text(text)
                    assert "Using module transformer already provided" in text
                    assert "[ComponentLoader] transformer materialized" not in text
                    imports = re.findall(
                        r"\[WeightCache\] transformer imported in ([\d.]+)s", text
                    )
                    assert len(imports) == 1 and float(imports[0]) < 2
                    workers = [
                        ProcessIdentity.read(p.pid)
                        for p in psutil.Process(context.process.pid).children(
                            recursive=True
                        )
                        if p.name().startswith("sgl_diffusion")
                    ]
                    assert len(workers) == 1
                    # Cover both graceful allocation drain and abrupt death.
                    owner.send_signal(signal.SIGTERM if index == 0 else signal.SIGKILL)
                    deadline = time.monotonic() + 10
                    while any(worker.is_alive() for worker in workers):
                        assert time.monotonic() < deadline, (
                            "Mapped Qwen worker survived owner loss"
                        )
                        time.sleep(0.1)
                    owner.wait(timeout=20)
                    context.cleanup()
                    context.process.wait(timeout=20)
                    context = None
                    owner = _start_owner(model, socket_path, env, log)
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
                "images": {
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
