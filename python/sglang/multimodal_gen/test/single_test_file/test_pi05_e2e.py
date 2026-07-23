# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import pytest

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

pytestmark = pytest.mark.skipif(
    os.getenv("SGLANG_RUN_PI05_E2E") != "1",
    reason="set SGLANG_RUN_PI05_E2E=1 to run Pi0.5 GPU e2e tests",
)

_MODEL_PATH = os.getenv("SGLANG_PI05_E2E_MODEL", "lerobot/pi05_base")
_CAMERA_ORDER = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value is None else float(value)


def _image(camera_index: int) -> np.ndarray:
    height = width = _env_int("SGLANG_PI05_E2E_IMAGE_SIZE", 224)
    y = np.arange(height, dtype=np.uint16)[:, None]
    x = np.arange(width, dtype=np.uint16)[None, :]
    image = np.stack(
        (
            (x + camera_index * 17) % 256 + np.zeros_like(y),
            (y + camera_index * 29) % 256 + np.zeros_like(x),
            (x + y + camera_index * 41) % 256,
        ),
        axis=-1,
    )
    return image.astype(np.uint8)


def _action_request_kwargs(tag: str) -> dict:
    action_horizon = _env_int("SGLANG_PI05_E2E_ACTION_HORIZON", 50)
    action_dim = _env_int("SGLANG_PI05_E2E_ACTION_DIM", 32)
    rng = np.random.default_rng(_env_int("SGLANG_PI05_E2E_NOISE_SEED", 0))
    prompt = os.getenv("SGLANG_PI05_E2E_PROMPT", "pick up the blue block")
    return {
        "prompt": f"{prompt} [{tag}]",
        "images": {name: _image(idx) for idx, name in enumerate(_CAMERA_ORDER)},
        "camera_order": list(_CAMERA_ORDER),
        "state": np.linspace(
            -0.5,
            0.5,
            _env_int("SGLANG_PI05_E2E_STATE_DIM", 32),
            dtype=np.float32,
        ),
        "noise": rng.standard_normal((action_horizon, action_dim)).astype(np.float32),
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "num_inference_steps": _env_int("SGLANG_PI05_E2E_NUM_STEPS", 2),
        "return_timing": True,
        "enable_prefix_cache": True,
        "enable_cuda_graph": os.getenv("SGLANG_PI05_E2E_CUDA_GRAPH", "1") != "0",
    }


@pytest.fixture(scope="module")
def pi05_generator():
    num_gpus = _env_int("SGLANG_PI05_E2E_NUM_GPUS", 1)
    kwargs = {
        "model_path": _MODEL_PATH,
        "num_gpus": num_gpus,
        "warmup": False,
        "trust_remote_code": False,
    }
    if num_gpus > 1:
        kwargs.update(
            {
                "sp_degree": _env_int("SGLANG_PI05_E2E_SP_DEGREE", num_gpus),
                "ulysses_degree": _env_int(
                    "SGLANG_PI05_E2E_ULYSSES_DEGREE",
                    num_gpus,
                ),
                "ring_degree": _env_int("SGLANG_PI05_E2E_RING_DEGREE", 1),
            }
        )
    generator = DiffGenerator.from_pretrained(local_mode=True, **kwargs)
    try:
        yield generator
    finally:
        generator.shutdown()


def _actions(output: dict) -> np.ndarray:
    return np.asarray(output["actions"], dtype=np.float32)


def _assert_action_output(
    output: dict, *, expect_cache_hit: bool | None = None
) -> None:
    actions = _actions(output)
    assert actions.shape[0] == _env_int("SGLANG_PI05_E2E_ACTION_HORIZON", 50)
    expected_output_dim = os.getenv("SGLANG_PI05_E2E_OUTPUT_ACTION_DIM")
    if expected_output_dim is not None:
        assert actions.shape[1] == int(expected_output_dim)
    else:
        assert 0 < actions.shape[1] <= _env_int("SGLANG_PI05_E2E_ACTION_DIM", 32)
    assert np.isfinite(actions).all()
    timings = output.get("timings") or {}
    assert timings.get("preprocess_ms", 0.0) >= 0.0
    assert timings.get("prefix_ms", 0.0) >= 0.0
    assert timings.get("action_denoise_ms", 0.0) > 0.0
    assert timings.get("postprocess_ms", 0.0) >= 0.0

    cache = output.get("cache") or {}
    if expect_cache_hit is not None:
        assert bool(cache.get("hit")) is expect_cache_hit

    parallel = output.get("parallel") or {}
    num_gpus = _env_int("SGLANG_PI05_E2E_NUM_GPUS", 1)
    assert bool(parallel.get("split_group", False)) is (num_gpus > 1)
    if num_gpus > 1:
        assert int(parallel["world_size"]) == num_gpus
        assert parallel["prefix_root"] == 0
        assert parallel["action_root"] == num_gpus - 1
        assert bool(parallel.get("action_sequence_parallel")) is True


def test_pi05_python_action_e2e(pi05_generator):
    output = pi05_generator.generate_action(_action_request_kwargs("e2e"))
    _assert_action_output(output)


def test_pi05_python_action_consistency(pi05_generator):
    first = pi05_generator.generate_action(_action_request_kwargs("consistency"))
    second = pi05_generator.generate_action(_action_request_kwargs("consistency"))
    _assert_action_output(first, expect_cache_hit=False)
    _assert_action_output(second, expect_cache_hit=True)

    first_actions = _actions(first)
    second_actions = _actions(second)
    np.testing.assert_allclose(
        first_actions,
        second_actions,
        rtol=_env_float("SGLANG_PI05_E2E_CONSISTENCY_RTOL") or 1e-3,
        atol=_env_float("SGLANG_PI05_E2E_CONSISTENCY_ATOL") or 1e-3,
    )

    gt_path = os.getenv("SGLANG_PI05_E2E_CONSISTENCY_GT")
    if gt_path is None:
        return
    path = Path(gt_path)
    if os.getenv("SGLANG_PI05_E2E_UPDATE_GT") == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, first_actions)
    else:
        np.testing.assert_allclose(
            first_actions,
            np.load(path),
            rtol=_env_float("SGLANG_PI05_E2E_GT_RTOL") or 1e-2,
            atol=_env_float("SGLANG_PI05_E2E_GT_ATOL") or 1e-2,
        )


def test_pi05_python_action_perf(pi05_generator):
    for _ in range(_env_int("SGLANG_PI05_E2E_PERF_WARMUP", 1)):
        pi05_generator.generate_action(_action_request_kwargs("perf"))

    records = []
    for _ in range(_env_int("SGLANG_PI05_E2E_PERF_REPEAT", 3)):
        start = time.perf_counter()
        output = pi05_generator.generate_action(_action_request_kwargs("perf"))
        wall_ms = (time.perf_counter() - start) * 1000
        _assert_action_output(output, expect_cache_hit=True)
        records.append(
            {
                "wall_ms": wall_ms,
                "timings": output.get("timings") or {},
                "parallel": output.get("parallel") or {},
            }
        )

    wall = [record["wall_ms"] for record in records]
    denoise = [record["timings"].get("action_denoise_ms", 0.0) for record in records]
    summary = {
        "model": _MODEL_PATH,
        "num_gpus": _env_int("SGLANG_PI05_E2E_NUM_GPUS", 1),
        "num_steps": _env_int("SGLANG_PI05_E2E_NUM_STEPS", 2),
        "median_wall_ms": statistics.median(wall),
        "median_action_denoise_ms": statistics.median(denoise),
        "records": records,
    }

    dump_path = os.getenv("SGLANG_PI05_E2E_PERF_DUMP")
    if dump_path:
        path = Path(dump_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    max_wall_ms = _env_float("SGLANG_PI05_E2E_MAX_WALL_MS")
    if max_wall_ms is not None:
        assert summary["median_wall_ms"] <= max_wall_ms
    max_denoise_ms = _env_float("SGLANG_PI05_E2E_MAX_ACTION_DENOISE_MS")
    if max_denoise_ms is not None:
        assert summary["median_action_denoise_ms"] <= max_denoise_ms
