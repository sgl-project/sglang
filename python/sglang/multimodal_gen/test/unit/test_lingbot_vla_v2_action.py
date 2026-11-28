# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sglang.multimodal_gen.configs.pipeline_configs.lingbot_vla_v2 import (
    LingBotVLAV2PipelineConfig,
    is_lingbot_vla_v2_checkpoint,
)
from sglang.multimodal_gen.registry import get_non_diffusers_pipeline_name
from sglang.multimodal_gen.runtime.entrypoints.action.protocol import (
    action_metadata,
    build_action_sampling_params,
)

ROBOTWIN_STATE_DIM = 14
CANONICAL_STATE_DIM = 55
ACTION_HORIZON = 50
CAMERA_NAME = "cam_high"
CAMERA_SHAPE = (8, 8, 3)


def make_server_args(checkpoint_dir: Path, state_dim: int) -> SimpleNamespace:
    checkpoint = {
        "type": "lingbot_vla_v2",
        "input_features": {
            "observation.state": {"shape": [state_dim]},
            f"observation.images.{CAMERA_NAME}": {"type": "VISUAL"},
        },
        "output_features": {"action": {"shape": [state_dim]}},
    }
    if state_dim == ROBOTWIN_STATE_DIM:
        for slot_name, feature in (
            ("state_slots", "observation.state"),
            ("action_slots", "action"),
        ):
            checkpoint[slot_name] = {
                "arm.position": {
                    "origin_keys": [{feature: {"start": 0, "end": state_dim}}]
                }
            }
    (checkpoint_dir / "config.json").write_text(json.dumps(checkpoint))
    config = LingBotVLAV2PipelineConfig(
        inputs_normalized=state_dim == CANONICAL_STATE_DIM
    )
    server_args = SimpleNamespace(
        model_path=str(checkpoint_dir),
        served_model_name="lingbot",
        pipeline_config=config,
        num_gpus=1,
        enable_cfg_parallel=False,
        quantization=None,
        dit_cpu_offload=False,
        text_encoder_cpu_offload=False,
        image_encoder_cpu_offload=False,
        dit_layerwise_offload=False,
        output_path=None,
        comfyui_mode=False,
    )
    config.validate_server_args(server_args)
    return server_args


def make_action_request(state_dim: int) -> dict[str, object]:
    return {
        "input": {
            "observation": {
                "prompt": "stack the blocks",
                "images": {CAMERA_NAME: np.zeros(CAMERA_SHAPE, dtype=np.uint8)},
                "state": np.zeros(state_dim, dtype=np.float32),
                "noise": np.zeros(
                    (ACTION_HORIZON, CANONICAL_STATE_DIM), dtype=np.float32
                ),
            },
        },
        "parameters": {"num_inference_steps": 2},
        "runtime": {"cuda_graph": False},
    }


@pytest.mark.parametrize("config_contents", ["[]", "{"])
def test_lingbot_detector_rejects_invalid_model_config(
    tmp_path: Path, config_contents: str
) -> None:
    (tmp_path / "config.json").write_text(config_contents)

    detected = is_lingbot_vla_v2_checkpoint(str(tmp_path))

    assert not detected


@pytest.mark.parametrize("state_dim", [ROBOTWIN_STATE_DIM, CANONICAL_STATE_DIM])
def test_lingbot_registers_action_metadata_for_both_checkpoints(
    tmp_path: Path, state_dim: int
) -> None:
    server_args = make_server_args(tmp_path, state_dim)

    pipeline_name = get_non_diffusers_pipeline_name(str(tmp_path))
    metadata = action_metadata(server_args)

    assert pipeline_name == "LingBotVLAV2Pipeline"
    assert metadata["input"]["state_dim"] == state_dim
    assert metadata["input"]["state_normalized"] == (state_dim == CANONICAL_STATE_DIM)
    assert metadata["output"]["action_dim"] == state_dim
    assert metadata["output"]["padded_action_dim"] == CANONICAL_STATE_DIM
    assert metadata["output"]["normalized"] == (state_dim == CANONICAL_STATE_DIM)


def test_lingbot_accepts_canonical_noise_with_a_robotwin_state(tmp_path: Path) -> None:
    server_args = make_server_args(tmp_path, ROBOTWIN_STATE_DIM)
    payload = make_action_request(ROBOTWIN_STATE_DIM)

    sampling_params = build_action_sampling_params(payload, server_args)

    assert sampling_params.num_inference_steps == 2
    assert not sampling_params.enable_cuda_graph
    assert tuple(sampling_params.images) == (CAMERA_NAME,)
    assert sampling_params.noise.shape == (ACTION_HORIZON, CANONICAL_STATE_DIM)


@pytest.mark.parametrize(
    ("invalid_fields", "message"),
    (
        ({"images": {}}, "at least one configured camera"),
        ({"images": {CAMERA_NAME: None}}, "numeric RGB pixels"),
        ({"images": {CAMERA_NAME: np.empty((0, 8, 3), dtype=np.uint8)}}, "nonempty"),
        (
            {"images": {CAMERA_NAME: np.zeros((8, 8, 2), dtype=np.uint8)}},
            "three RGB channels",
        ),
        ({"images": {CAMERA_NAME: np.full(CAMERA_SHAPE, 2.0)}}, "finite and in"),
        ({"state": None}, "requires state"),
        ({"state": np.full(ROBOTWIN_STATE_DIM, np.inf)}, "state must be finite"),
        (
            {"noise": np.full((ACTION_HORIZON, CANONICAL_STATE_DIM), np.nan)},
            "noise must be finite",
        ),
        (
            {"noise": np.zeros((ACTION_HORIZON - 1, CANONICAL_STATE_DIM))},
            "LingBot noise must have shape",
        ),
        ({"prompt": "  "}, "nonempty prompt"),
    ),
)
def test_lingbot_rejects_invalid_action_observations(
    tmp_path: Path, invalid_fields: dict[str, object], message: str
) -> None:
    server_args = make_server_args(tmp_path, ROBOTWIN_STATE_DIM)
    payload = make_action_request(ROBOTWIN_STATE_DIM)
    payload["input"]["observation"].update(invalid_fields)

    with pytest.raises(ValueError, match=message):
        build_action_sampling_params(payload, server_args)
