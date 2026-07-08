# SPDX-License-Identifier: Apache-2.0

import dataclasses
from types import SimpleNamespace

import numpy as np

from sglang.multimodal_gen.configs.pipeline_configs.pi05 import Pi05PipelineConfig
from sglang.multimodal_gen.configs.sample.pi05 import Pi05SamplingParams
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.configs.sample.vla import VLASamplingParams
from sglang.multimodal_gen.runtime.entrypoints.vla.protocol import (
    action_generation_response,
    action_metadata,
    action_raw_response,
    build_action_sampling_params,
    pack_msgpack,
    unpack_msgpack,
)


def _server_args(config: Pi05PipelineConfig | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        model_id=None,
        model_path="lerobot/pi05_base",
        output_path=None,
        comfyui_mode=False,
        pipeline_config=config or Pi05PipelineConfig(),
    )


def test_pi05_uses_vla_sampling_params_not_visual_sampling_params():
    params = Pi05SamplingParams()
    field_names = {field.name for field in dataclasses.fields(params)}

    assert isinstance(params, VLASamplingParams)
    assert not isinstance(params, SamplingParams)
    assert "action_horizon" in field_names
    assert "action_dim" in field_names
    assert "height" not in field_names
    assert "width" not in field_names
    assert "fps" not in field_names
    assert "negative_prompt" not in field_names
    assert "return_frames" not in field_names
    assert "diffusers_kwargs" not in field_names


def test_action_adjust_skips_visual_image_video_logic():
    params = SamplingParams()
    params.num_frames = 0
    params.adjust_frames = True
    params.return_file_paths_only = True

    params._adjust(_server_args())

    assert params.data_type == DataType.ACTION
    assert params.num_frames == 1
    assert params.adjust_frames is False
    assert params.return_file_paths_only is False


def test_action_request_schema_builds_pi05_sampling_params():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    payload = {
        "model": "lerobot/pi05_base",
        "input": {
            "task": "pick up the block",
            "observation": {
                "images": {
                    "base_0_rgb": {
                        "dtype": "uint8",
                        "shape": [8, 8, 3],
                        "values": image.tolist(),
                    },
                },
                "state": {
                    "dtype": "float32",
                    "shape": [32],
                    "values": np.arange(32, dtype=np.float32).tolist(),
                },
            },
        },
        "parameters": {
            "action_horizon": 25,
            "action_dim": 32,
            "num_inference_steps": 4,
        },
        "runtime": {
            "return_timing": "false",
            "prefix_cache": False,
            "cuda_graph": "0",
        },
    }

    params = build_action_sampling_params(payload, _server_args())

    assert params.prompt == "pick up the block"
    assert params.action_horizon == 25
    assert params.action_dim == 32
    assert params.num_inference_steps == 4
    assert not params.return_timing
    assert not params.enable_prefix_cache
    assert not params.enable_cuda_graph
    assert params.return_file_paths_only is False
    assert params.save_output is False
    assert set(params.images) == {"base_0_rgb"}
    assert params.images["base_0_rgb"].shape == (8, 8, 3)
    assert params.state.shape == (32,)


def test_openpi_raw_observation_compatibility_fields_are_normalized():
    payload = {
        "task": "push the cube",
        "observation.images.base_0_rgb": np.ones((4, 4, 3), dtype=np.uint8),
        "observation.state": {
            "dtype": "float32",
            "shape": [32],
            "data": [0.25] * 32,
        },
        "observation.noise": {
            "dtype": "float32",
            "shape": [50, 32],
            "data": np.zeros((50, 32), dtype=np.float32).tolist(),
        },
        "enable_pi_prefix_cache": False,
        "enable_pi_cuda_graph": False,
    }

    params = build_action_sampling_params(payload, _server_args())

    assert params.prompt == "push the cube"
    assert set(params.images) == {"base_0_rgb"}
    assert params.state.shape == (32,)
    assert params.noise.shape == (50, 32)
    assert not params.enable_prefix_cache
    assert not params.enable_cuda_graph


def test_action_metadata_reports_policy_shape_and_capabilities():
    config = Pi05PipelineConfig(
        image_keys=("front", "wrist"),
        image_size=(256, 256),
        state_dim=8,
        action_horizon=10,
        action_dim=32,
        output_action_dim=7,
        enable_global_prefix_cache=False,
        enable_action_cuda_graph=True,
    )

    metadata = action_metadata(_server_args(config))

    assert metadata["object"] == "action.metadata"
    assert metadata["policy_family"] == "pi05"
    assert metadata["input"]["image_keys"] == ["front", "wrist"]
    assert metadata["input"]["image_size"] == [256, 256]
    assert metadata["input"]["state_dim"] == 8
    assert metadata["output"]["action_horizon"] == 10
    assert metadata["output"]["action_dim"] == 7
    assert metadata["output"]["padded_action_dim"] == 32
    assert metadata["runtime"]["materialize_dtype"] == "bf16"
    assert metadata["runtime"]["enable_autocast"] is True
    assert metadata["defaults"]["prefix_cache"] is False
    assert metadata["capabilities"]["realtime_websocket"]
    assert metadata["capabilities"]["openpi_websocket"]


def test_action_generation_response_uses_actual_output_parameters():
    output = {
        "actions": [[1.0, 2.0], [3.0, 4.0]],
        "parameters": {"num_inference_steps": 3},
        "timings": {"preprocess_ms": 1.5},
        "cache": {"hit": True},
    }

    response = action_generation_response(output, _server_args())

    assert response["object"] == "action.generation"
    assert response["data"][0]["action"]["shape"] == [2, 2]
    assert response["data"][0]["action"]["values"] == output["actions"]
    assert response["usage"]["action_horizon"] == 2
    assert response["usage"]["action_dim"] == 2
    assert response["usage"]["denoise_steps"] == 3
    assert response["usage"]["prefix_cache_hit"] is True
    assert response["timings"] == output["timings"]
    assert response["cache"] == output["cache"]


def test_action_raw_response_preserves_policy_payload_shape():
    actions = np.arange(6, dtype=np.float32).reshape(2, 3)
    output = {
        "actions": actions,
        "timings": {"preprocess_ms": 1.5},
    }

    response = action_raw_response(output)

    assert response["actions"] == actions.tolist()
    assert response["timings"] == output["timings"]


def test_action_raw_response_can_preserve_numpy_for_msgpack():
    actions = np.arange(6, dtype=np.float32).reshape(2, 3)

    response = action_raw_response({"actions": actions}, preserve_numpy=True)

    assert response["actions"] is actions


def test_msgpack_roundtrip_preserves_string_keys_and_numpy_payloads():
    payload = {
        "task": "pick",
        "array": np.arange(6, dtype=np.float32).reshape(2, 3),
        "scalar": np.float32(1.25),
    }

    decoded = unpack_msgpack(pack_msgpack(payload))

    assert decoded["task"] == "pick"
    np.testing.assert_array_equal(decoded["array"], payload["array"])
    assert decoded["scalar"] == payload["scalar"]
