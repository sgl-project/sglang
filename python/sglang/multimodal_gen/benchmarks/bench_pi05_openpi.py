# SPDX-License-Identifier: Apache-2.0

"""Manual Pi0.5 SGLang vs OpenPI benchmark.

This script is intentionally outside the unit-test path. It needs GPU memory,
Pi0.5 checkpoints, and an OpenPI install for the baseline.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)


@dataclass(frozen=True)
class Pi05BenchProfile:
    name: str
    sglang_model: str
    openpi_config: str
    openpi_checkpoint: str
    prompt: str
    sglang_action_horizon: int
    openpi_action_horizon: int
    action_dim: int
    output_action_dim: int


PROFILES = {
    "libero": Pi05BenchProfile(
        name="libero",
        sglang_model="lerobot/pi05_libero_base",
        openpi_config="pi05_libero",
        openpi_checkpoint="gs://openpi-assets/checkpoints/pi05_libero",
        prompt="pick up the object",
        sglang_action_horizon=50,
        openpi_action_horizon=10,
        action_dim=32,
        output_action_dim=7,
    ),
    "aloha": Pi05BenchProfile(
        name="aloha",
        sglang_model="lerobot/pi05_base",
        openpi_config="pi05_aloha",
        openpi_checkpoint="gs://openpi-assets/checkpoints/pi05_base",
        prompt="pick up the block",
        sglang_action_horizon=50,
        openpi_action_horizon=50,
        action_dim=32,
        output_action_dim=14,
    ),
}


def _stats_ms(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {}
    values = np.asarray(samples, dtype=np.float64)
    return {
        "count": float(len(samples)),
        "mean_ms": float(np.mean(values)),
        "std_ms": float(np.std(values)),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def _image(rng: np.random.Generator, *, chw: bool = False) -> np.ndarray:
    image = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    if chw:
        return np.transpose(image, (2, 0, 1))
    return image


def _make_libero_observation(
    rng: np.random.Generator,
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_image = _image(rng)
    wrist_image = _image(rng)
    state = rng.random(8, dtype=np.float32)
    openpi_obs = {
        "observation/state": state,
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "prompt": prompt,
    }
    sglang_observation = {
        "images": {
            "image": base_image,
            "image2": wrist_image,
        },
        "state": state,
    }
    return openpi_obs, sglang_observation


def _make_aloha_observation(
    rng: np.random.Generator,
    prompt: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cam_high = _image(rng, chw=True)
    cam_left = _image(rng, chw=True)
    cam_right = _image(rng, chw=True)
    state = np.ones((14,), dtype=np.float32)
    openpi_obs = {
        "state": state,
        "images": {
            "cam_high": cam_high,
            "cam_low": _image(rng, chw=True),
            "cam_left_wrist": cam_left,
            "cam_right_wrist": cam_right,
        },
        "prompt": prompt,
    }
    sglang_state = np.zeros((32,), dtype=np.float32)
    sglang_state[: state.shape[0]] = state
    sglang_observation = {
        "images": {
            "base_0_rgb": np.transpose(cam_high, (1, 2, 0)),
            "left_wrist_0_rgb": np.transpose(cam_left, (1, 2, 0)),
            "right_wrist_0_rgb": np.transpose(cam_right, (1, 2, 0)),
        },
        "state": sglang_state,
    }
    return openpi_obs, sglang_observation


def build_observations(
    profile: Pi05BenchProfile,
    count: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    openpi_observations = []
    sglang_observations = []
    for _ in range(count):
        if profile.name == "libero":
            openpi_obs, sglang_obs = _make_libero_observation(rng, profile.prompt)
        elif profile.name == "aloha":
            openpi_obs, sglang_obs = _make_aloha_observation(rng, profile.prompt)
        else:
            raise ValueError(f"Unsupported Pi0.5 benchmark profile: {profile.name}")
        openpi_observations.append(openpi_obs)
        sglang_observations.append(sglang_obs)
    return openpi_observations, sglang_observations


def _json_tensor(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "values": array.tolist(),
    }


def pack_numpy_payload(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def unpack_numpy_payload(obj):
    ndarray_marker = obj.get("__ndarray__") or obj.get(b"__ndarray__")
    npgeneric_marker = obj.get("__npgeneric__") or obj.get(b"__npgeneric__")
    data = obj.get("data", obj.get(b"data"))
    dtype = obj.get("dtype", obj.get(b"dtype"))
    shape = obj.get("shape", obj.get(b"shape"))
    if ndarray_marker:
        return np.ndarray(
            buffer=data,
            dtype=np.dtype(dtype),
            shape=shape,
        )
    if npgeneric_marker:
        return np.dtype(dtype).type(data)
    return obj


def pack_msgpack(payload: Any) -> bytes:
    import msgpack

    return msgpack.packb(payload, default=pack_numpy_payload, use_bin_type=True)


def unpack_msgpack(payload: bytes) -> Any:
    import msgpack

    return msgpack.unpackb(payload, object_hook=unpack_numpy_payload, raw=False)


def build_sglang_payload(
    profile: Pi05BenchProfile,
    observation: dict[str, Any],
    *,
    num_inference_steps: int,
    prefix_cache: bool,
    cuda_graph: bool,
    noise: np.ndarray | None,
) -> dict[str, Any]:
    encoded_images = {
        key: _json_tensor(np.asarray(value))
        for key, value in observation["images"].items()
    }
    encoded_observation = {
        "images": encoded_images,
        "state": _json_tensor(np.asarray(observation["state"], dtype=np.float32)),
    }
    if noise is not None:
        encoded_observation["noise"] = _json_tensor(noise.astype(np.float32))
    return {
        "model": profile.sglang_model,
        "input": {
            "task": profile.prompt,
            "observation": encoded_observation,
        },
        "parameters": {
            "num_inference_steps": num_inference_steps,
        },
        "runtime": {
            "return_timing": True,
            "prefix_cache": prefix_cache,
            "cuda_graph": cuda_graph,
        },
    }


def build_sglang_openpi_ws_payload(
    profile: Pi05BenchProfile,
    observation: dict[str, Any],
    *,
    num_inference_steps: int,
    prefix_cache: bool,
    cuda_graph: bool,
    noise: np.ndarray | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": profile.prompt,
        "observation.state": np.asarray(observation["state"], dtype=np.float32),
        "num_inference_steps": num_inference_steps,
        "enable_pi_prefix_cache": prefix_cache,
        "enable_pi_cuda_graph": cuda_graph,
    }
    for key, value in observation["images"].items():
        payload[f"observation.images.{key}"] = np.asarray(value)
    if noise is not None:
        payload["observation.noise"] = noise.astype(np.float32)
    return payload


def _post_action(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def run_sglang_http(
    url: str,
    payloads: list[dict[str, Any]],
    *,
    warmup: int,
    repeats: int,
    batch_size: int,
    timeout_s: float,
) -> dict[str, Any]:
    endpoint = url.rstrip("/") + "/v1/actions/generations"
    for idx in range(min(warmup, len(payloads))):
        _post_action(endpoint, payloads[idx], timeout_s)

    single_latencies = []
    single_outputs = []
    for idx in range(repeats):
        payload = payloads[idx % len(payloads)]
        start = time.perf_counter()
        output = _post_action(endpoint, payload, timeout_s)
        single_latencies.append((time.perf_counter() - start) * 1000)
        single_outputs.append(output)

    batch_latencies = []
    if batch_size > 1:
        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            for warmup_idx in range(warmup):
                batch = [
                    payloads[(warmup_idx * batch_size + offset) % len(payloads)]
                    for offset in range(batch_size)
                ]
                list(
                    pool.map(
                        lambda payload: _post_action(endpoint, payload, timeout_s),
                        batch,
                    )
                )
            for start_idx in range(repeats):
                batch = [
                    payloads[(start_idx * batch_size + offset) % len(payloads)]
                    for offset in range(batch_size)
                ]
                start = time.perf_counter()
                list(
                    pool.map(
                        lambda payload: _post_action(endpoint, payload, timeout_s),
                        batch,
                    )
                )
                batch_latencies.append((time.perf_counter() - start) * 1000)

    stage_timings = {}
    for output in single_outputs:
        for key, value in output.get("timings", {}).items():
            stage_timings.setdefault(key, []).append(float(value))

    return {
        "single": _stats_ms(single_latencies),
        "batch": _stats_ms(batch_latencies),
        "batch_size": batch_size,
        "stage_timings": {
            key: _stats_ms(values) for key, values in stage_timings.items()
        },
        "first_output": single_outputs[0] if single_outputs else None,
        "batch_mode": "concurrent_http_requests",
    }


def _action_ws_url(url: str) -> str:
    if url.startswith("https://"):
        return "wss://" + url[len("https://") :].rstrip("/") + "/openpi/policy"
    if url.startswith("http://"):
        return "ws://" + url[len("http://") :].rstrip("/") + "/openpi/policy"
    return url.rstrip("/") + "/openpi/policy"


async def _ws_send_recv(websocket, payload: dict[str, Any]) -> dict[str, Any]:
    await websocket.send(pack_msgpack(payload))
    response = await websocket.recv()
    if isinstance(response, str):
        raise RuntimeError(response)
    return unpack_msgpack(response)


async def _run_sglang_openpi_ws_async(
    url: str,
    payloads: list[dict[str, Any]],
    *,
    warmup: int,
    repeats: int,
    batch_size: int,
) -> dict[str, Any]:
    import websockets

    endpoint = _action_ws_url(url)
    async with websockets.connect(endpoint, max_size=None) as websocket:
        await websocket.recv()
        for idx in range(min(warmup, len(payloads))):
            await _ws_send_recv(websocket, payloads[idx])

        single_latencies = []
        single_outputs = []
        for idx in range(repeats):
            payload = payloads[idx % len(payloads)]
            start = time.perf_counter()
            output = await _ws_send_recv(websocket, payload)
            single_latencies.append((time.perf_counter() - start) * 1000)
            single_outputs.append(output)

    batch_latencies = []
    if batch_size > 1:
        websockets_list = []
        try:
            for _ in range(batch_size):
                websocket = await websockets.connect(endpoint, max_size=None)
                await websocket.recv()
                websockets_list.append(websocket)

            for warmup_idx in range(warmup):
                batch = [
                    payloads[(warmup_idx * batch_size + offset) % len(payloads)]
                    for offset in range(batch_size)
                ]
                await asyncio.gather(
                    *[
                        _ws_send_recv(websocket, payload)
                        for websocket, payload in zip(websockets_list, batch)
                    ]
                )

            for start_idx in range(repeats):
                batch = [
                    payloads[(start_idx * batch_size + offset) % len(payloads)]
                    for offset in range(batch_size)
                ]
                start = time.perf_counter()
                await asyncio.gather(
                    *[
                        _ws_send_recv(websocket, payload)
                        for websocket, payload in zip(websockets_list, batch)
                    ]
                )
                batch_latencies.append((time.perf_counter() - start) * 1000)
        finally:
            for websocket in websockets_list:
                await websocket.close()

    stage_timings = {}
    server_timings = {}
    for output in single_outputs:
        for key, value in output.get("timings", {}).items():
            stage_timings.setdefault(key, []).append(float(value))
        for key, value in output.get("server_timing", {}).items():
            server_timings.setdefault(key, []).append(float(value))

    return {
        "single": _stats_ms(single_latencies),
        "batch": _stats_ms(batch_latencies),
        "batch_size": batch_size,
        "stage_timings": {
            key: _stats_ms(values) for key, values in stage_timings.items()
        },
        "server_timings": {
            key: _stats_ms(values) for key, values in server_timings.items()
        },
        "first_output": single_outputs[0] if single_outputs else None,
        "batch_mode": "persistent_openpi_websocket",
    }


def run_sglang_openpi_ws(
    url: str,
    payloads: list[dict[str, Any]],
    *,
    warmup: int,
    repeats: int,
    batch_size: int,
) -> dict[str, Any]:
    return asyncio.run(
        _run_sglang_openpi_ws_async(
            url,
            payloads,
            warmup=warmup,
            repeats=repeats,
            batch_size=batch_size,
        )
    )


def create_openpi_policy(
    config_name: str,
    checkpoint_dir: str,
    *,
    pytorch_device: str,
    num_inference_steps: int,
    pytorch_compile_mode: str | None,
):
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(config_name)
    if pytorch_compile_mode != "keep":
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model,
                pytorch_compile_mode=pytorch_compile_mode,
            ),
        )
    return policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        sample_kwargs={"num_steps": num_inference_steps},
        pytorch_device=pytorch_device,
    )


def _openpi_infer(policy, observation: dict[str, Any], noise: np.ndarray | None):
    if noise is None:
        return policy.infer(observation)
    return policy.infer(observation, noise=noise)


def _openpi_direct_batch(
    policy,
    observations: list[dict[str, Any]],
    noises: list[np.ndarray] | None,
):
    import jax
    import numpy as onp
    from openpi.models import model as openpi_model

    inputs_list = [
        policy._input_transform(jax.tree.map(lambda value: value, observation))
        for observation in observations
    ]
    batched_inputs = jax.tree.map(
        lambda *values: onp.stack(values, axis=0),
        *inputs_list,
    )
    sample_kwargs = dict(policy._sample_kwargs)
    if noises is not None:
        noise = onp.stack(noises, axis=0)
        if policy._is_pytorch_model:
            import torch

            sample_kwargs["noise"] = torch.from_numpy(noise).to(policy._pytorch_device)
        else:
            import jax.numpy as jnp

            sample_kwargs["noise"] = jnp.asarray(noise)

    if policy._is_pytorch_model:
        import torch

        inputs = jax.tree.map(
            lambda value: torch.from_numpy(onp.asarray(value)).to(
                policy._pytorch_device
            ),
            batched_inputs,
        )
        sample_key_or_device = policy._pytorch_device
    else:
        import jax.numpy as jnp

        inputs = jax.tree.map(lambda value: jnp.asarray(value), batched_inputs)
        policy._rng, sample_key_or_device = jax.random.split(policy._rng)

    observation = openpi_model.Observation.from_dict(inputs)
    actions = policy._sample_actions(sample_key_or_device, observation, **sample_kwargs)
    if policy._is_pytorch_model:
        actions_np = actions.detach().cpu().numpy()
        states_np = inputs["state"].detach().cpu().numpy()
    else:
        actions_np = onp.asarray(actions)
        states_np = onp.asarray(inputs["state"])

    outputs = []
    for idx in range(actions_np.shape[0]):
        outputs.append(
            policy._output_transform(
                {
                    "state": states_np[idx],
                    "actions": actions_np[idx],
                }
            )
        )
    return outputs


def run_openpi_policy(
    policy,
    observations: list[dict[str, Any]],
    *,
    warmup: int,
    repeats: int,
    batch_size: int,
    noise: np.ndarray | None,
    batch_mode: str,
) -> dict[str, Any]:
    for idx in range(min(warmup, len(observations))):
        _openpi_infer(policy, observations[idx], noise)

    single_latencies = []
    single_outputs = []
    for idx in range(repeats):
        observation = observations[idx % len(observations)]
        start = time.perf_counter()
        output = _openpi_infer(policy, observation, noise)
        single_latencies.append((time.perf_counter() - start) * 1000)
        single_outputs.append(output)

    batch_latencies = []
    if batch_size > 1:
        for warmup_idx in range(warmup):
            batch = [
                observations[(warmup_idx * batch_size + offset) % len(observations)]
                for offset in range(batch_size)
            ]
            noises = [noise] * len(batch) if noise is not None else None
            if batch_mode == "direct_model":
                _openpi_direct_batch(policy, batch, noises)
            elif batch_mode == "policy_loop":
                for obs in batch:
                    _openpi_infer(policy, obs, noise)
            else:
                raise ValueError(f"Unsupported OpenPI batch mode: {batch_mode}")
        for start_idx in range(repeats):
            batch = [
                observations[(start_idx * batch_size + offset) % len(observations)]
                for offset in range(batch_size)
            ]
            noises = [noise] * len(batch) if noise is not None else None
            start = time.perf_counter()
            if batch_mode == "direct_model":
                _openpi_direct_batch(policy, batch, noises)
            elif batch_mode == "policy_loop":
                for obs in batch:
                    _openpi_infer(policy, obs, noise)
            else:
                raise ValueError(f"Unsupported OpenPI batch mode: {batch_mode}")
            batch_latencies.append((time.perf_counter() - start) * 1000)

    policy_timings = {}
    for output in single_outputs:
        for key, value in output.get("policy_timing", {}).items():
            policy_timings.setdefault(key, []).append(float(value))

    return {
        "single": _stats_ms(single_latencies),
        "batch": _stats_ms(batch_latencies),
        "batch_size": batch_size,
        "policy_timings": {
            key: _stats_ms(values) for key, values in policy_timings.items()
        },
        "first_output": single_outputs[0] if single_outputs else None,
        "batch_mode": batch_mode,
    }


def _sglang_actions(output: dict[str, Any]) -> np.ndarray | None:
    if output is None:
        return None
    if "actions" in output:
        return np.asarray(output["actions"], dtype=np.float32)
    return np.asarray(output["data"][0]["action"]["values"], dtype=np.float32)


def _openpi_actions(output: dict[str, Any]) -> np.ndarray | None:
    if output is None:
        return None
    return np.asarray(output["actions"], dtype=np.float32)


def compare_first_actions(
    sglang_output: dict[str, Any] | None,
    openpi_output: dict[str, Any] | None,
) -> dict[str, Any]:
    sglang_actions = _sglang_actions(sglang_output)
    openpi_actions = _openpi_actions(openpi_output)
    if sglang_actions is None or openpi_actions is None:
        return {"available": False}
    horizon = min(sglang_actions.shape[0], openpi_actions.shape[0])
    dim = min(sglang_actions.shape[1], openpi_actions.shape[1])
    if horizon == 0 or dim == 0:
        return {
            "available": False,
            "sglang_shape": list(sglang_actions.shape),
            "openpi_shape": list(openpi_actions.shape),
        }
    diff = np.abs(sglang_actions[:horizon, :dim] - openpi_actions[:horizon, :dim])
    return {
        "available": True,
        "common_shape": [int(horizon), int(dim)],
        "sglang_shape": list(sglang_actions.shape),
        "openpi_shape": list(openpi_actions.shape),
        "max_abs_diff": float(np.max(diff)),
        "mean_abs_diff": float(np.mean(diff)),
    }


def print_summary(result: dict[str, Any]) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
    sglang_result = result.get("sglang") or {}
    openpi_result = result.get("openpi") or {}
    sgl = sglang_result.get("single", {}).get("mean_ms")
    opi = openpi_result.get("single", {}).get("mean_ms")
    if sgl and opi:
        print(
            "\nSingle mean latency: "
            f"SGLang={sgl:.2f} ms, OpenPI={opi:.2f} ms, "
            f"speedup={opi / sgl:.2f}x"
        )
    sgl_batch = sglang_result.get("batch", {}).get("mean_ms")
    opi_batch = openpi_result.get("batch", {}).get("mean_ms")
    batch_size = sglang_result.get("batch_size") or openpi_result.get("batch_size", 0)
    if sgl_batch and opi_batch and batch_size:
        print(
            "Batch mean latency: "
            f"SGLang={sgl_batch:.2f} ms/{batch_size} req, "
            f"OpenPI={opi_batch:.2f} ms/{batch_size} req, "
            f"speedup={opi_batch / sgl_batch:.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="libero")
    parser.add_argument("--sglang-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--sglang-api",
        choices=("http", "openpi_ws"),
        default="http",
    )
    parser.add_argument("--sglang-model", default=None)
    parser.add_argument("--openpi-config", default=None)
    parser.add_argument("--openpi-checkpoint", default=None)
    parser.add_argument("--openpi-device", default="cuda")
    parser.add_argument(
        "--openpi-pytorch-compile-mode",
        choices=(
            "keep",
            "none",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="keep",
    )
    parser.add_argument(
        "--openpi-batch-mode",
        choices=("direct_model", "policy_loop"),
        default="direct_model",
    )
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--deterministic-noise", action="store_true")
    parser.add_argument("--skip-sglang", action="store_true")
    parser.add_argument("--skip-openpi", action="store_true")
    parser.add_argument("--output-file", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    if args.sglang_model:
        profile = Pi05BenchProfile(
            name=profile.name,
            sglang_model=args.sglang_model,
            openpi_config=profile.openpi_config,
            openpi_checkpoint=profile.openpi_checkpoint,
            prompt=profile.prompt,
            sglang_action_horizon=profile.sglang_action_horizon,
            openpi_action_horizon=profile.openpi_action_horizon,
            action_dim=profile.action_dim,
            output_action_dim=profile.output_action_dim,
        )
    openpi_config = args.openpi_config or profile.openpi_config
    openpi_checkpoint = args.openpi_checkpoint or profile.openpi_checkpoint

    openpi_observations, sglang_observations = build_observations(
        profile,
        max(args.num_samples, args.batch_size),
        args.seed,
    )
    noise = None
    if args.skip_sglang and args.skip_openpi:
        raise ValueError("At least one backend must be enabled")
    if args.deterministic_noise:
        if profile.sglang_action_horizon != profile.openpi_action_horizon:
            raise ValueError(
                "--deterministic-noise requires matching SGLang and OpenPI "
                f"action horizons; profile {profile.name!r} has "
                f"{profile.sglang_action_horizon} vs {profile.openpi_action_horizon}"
            )
        rng = np.random.default_rng(args.seed + 1)
        noise = rng.standard_normal(
            (profile.sglang_action_horizon, profile.action_dim),
            dtype=np.float32,
        )

    payloads = []
    if args.skip_sglang:
        pass
    elif args.sglang_api == "openpi_ws":
        payloads = [
            build_sglang_openpi_ws_payload(
                profile,
                observation,
                num_inference_steps=args.num_inference_steps,
                prefix_cache=not args.disable_prefix_cache,
                cuda_graph=not args.disable_cuda_graph,
                noise=noise,
            )
            for observation in sglang_observations
        ]
    else:
        payloads = [
            build_sglang_payload(
                profile,
                observation,
                num_inference_steps=args.num_inference_steps,
                prefix_cache=not args.disable_prefix_cache,
                cuda_graph=not args.disable_cuda_graph,
                noise=noise,
            )
            for observation in sglang_observations
        ]

    openpi_policy = None
    if not args.skip_openpi:
        openpi_policy = create_openpi_policy(
            openpi_config,
            openpi_checkpoint,
            pytorch_device=args.openpi_device,
            num_inference_steps=args.num_inference_steps,
            pytorch_compile_mode=(
                None
                if args.openpi_pytorch_compile_mode == "none"
                else args.openpi_pytorch_compile_mode
            ),
        )

    sglang_result = None
    if args.skip_sglang:
        pass
    elif args.sglang_api == "openpi_ws":
        sglang_result = run_sglang_openpi_ws(
            args.sglang_url,
            payloads,
            warmup=args.warmup,
            repeats=args.repeats,
            batch_size=args.batch_size,
        )
    else:
        sglang_result = run_sglang_http(
            args.sglang_url,
            payloads,
            warmup=args.warmup,
            repeats=args.repeats,
            batch_size=args.batch_size,
            timeout_s=args.timeout_s,
        )
    openpi_result = None
    if openpi_policy is not None:
        openpi_result = run_openpi_policy(
            openpi_policy,
            openpi_observations,
            warmup=args.warmup,
            repeats=args.repeats,
            batch_size=args.batch_size,
            noise=noise,
            batch_mode=args.openpi_batch_mode,
        )

    result = {
        "profile": profile.name,
        "sglang_model": profile.sglang_model,
        "sglang_api": args.sglang_api,
        "openpi_config": openpi_config,
        "openpi_checkpoint": openpi_checkpoint,
        "openpi_pytorch_compile_mode": args.openpi_pytorch_compile_mode,
        "num_inference_steps": args.num_inference_steps,
        "num_samples": args.num_samples,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "deterministic_noise": args.deterministic_noise,
        "action_diff": compare_first_actions(
            None if sglang_result is None else sglang_result.get("first_output"),
            None if openpi_result is None else openpi_result.get("first_output"),
        ),
        "sglang": (
            None
            if sglang_result is None
            else {
                key: value
                for key, value in sglang_result.items()
                if key not in ("first_output",)
            }
        ),
        "openpi": (
            None
            if openpi_result is None
            else {
                key: value
                for key, value in openpi_result.items()
                if key not in ("first_output",)
            }
        ),
    }
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    print_summary(result)


if __name__ == "__main__":
    main()
