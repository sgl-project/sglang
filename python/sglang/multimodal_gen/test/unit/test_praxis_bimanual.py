# SPDX-License-Identifier: Apache-2.0
"""The bimanual (YAM) rig of Praxis-1: config, canvas, proprio, actions, session and request conversion.

CPU only, no weights. A small encoder stands in for the VAE.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang.multimodal_gen.configs.models.dits.praxis import PraxisArchConfig
from sglang.multimodal_gen.configs.pipeline_configs.praxis import RIGS, praxis_settings
from sglang.multimodal_gen.configs.sample.praxis import PraxisSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.praxis import (
    commands_to_actions,
    reference_observation,
)
from sglang.multimodal_gen.runtime.vla import praxis_bimanual as bm
from sglang.multimodal_gen.runtime.vla.praxis_geometry import (
    NormStats,
    pose8_to_10d,
    rot6d,
)
from sglang.multimodal_gen.runtime.vla.praxis_policy import PolicyLayout

YAM = json.loads((Path(__file__).parent / "data" / "praxis-1-yam.json").read_text())
# config.json of runwayml/praxis-1-yam-lerobot as published.
YAM_LEROBOT = json.loads(
    (Path(__file__).parent / "data" / "praxis-1-yam-lerobot.json").read_text()
)


def _stats() -> NormStats:
    return NormStats.from_config(YAM["normalization"])


def _quat(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def _pose8(xyz: list[float], quat: np.ndarray, grip: float) -> np.ndarray:
    return np.concatenate([xyz, quat, [grip]]).astype(np.float32)


LEFT = _pose8([0.4, 0.2, 0.3], _quat(np.array([0.0, 1.0, 0.2]), 0.7), 0.25)
RIGHT = _pose8([0.5, -0.2, 0.35], _quat(np.array([1.0, 0.0, 0.3]), -0.4), 1.2)


def _views(h: int, w: int, t: int = 1, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {v: rng.integers(0, 256, (t, h, w, 3), dtype=np.uint8) for v in bm.VIEWS}


def _obs(t: int = 1) -> dict:
    obs: dict = {}
    for side, pose in (("left", LEFT), ("right", RIGHT)):
        obs[f"robot_state.{side}.ee_pose"] = np.tile(pose[:7], (t, 1))
        obs[f"grip.{side}"] = np.full(t, pose[7], np.float32)
    for view, frames in _views(260, 416, t=t).items():
        obs[bm.VIEW_KEY_CANDIDATES[view][0]] = frames
    return obs


def test_yam_config_selects_the_bimanual_rig():
    settings = praxis_settings({"type": "praxis", "model_config": YAM})
    assert settings.embodiment == "bimanual"
    arch = PraxisArchConfig.from_model_config(YAM)
    assert arch.action_dim == 80 and arch.expert_dim == 1024
    layout = PolicyLayout.from_model_config(YAM)
    assert layout.embodiment == "bimanual" and layout.tile_size == (256, 416)
    assert bm.canvas_size(YAM) == (256, 1248)
    assert list(bm.q99_dims(YAM)) == list(bm.Q99_DIMS)


def test_published_yam_config_sets_its_own_sampler():
    settings = praxis_settings(YAM_LEROBOT)
    assert settings.embodiment == "bimanual"
    assert (settings.num_steps, settings.t_shift) == (2, 1.0)
    assert (settings.cfg_scale, settings.cfg_video_scale) == (1.0, 1.0)
    assert settings.chunk_size == 24 and settings.max_context_chunks == 1


def test_published_yam_config_matches_the_rig():
    model = YAM_LEROBOT["model_config"]
    assert bm.canvas_size(model) == (256, 1248)
    assert model["action_space"]["dim"] == bm.ACTION_DIM
    assert list(bm.q99_dims(model)) == list(bm.Q99_DIMS)
    cameras = [
        key.removeprefix("observation.images.") for key in YAM_LEROBOT["camera_keys"]
    ]
    assert tuple(cameras) == bm.VIEWS == tuple(model["video"]["views"])
    # The request names the top camera "exterior"; every request view reaches one camera.
    for view in RIGS["bimanual"].image_keys:
        assert any(
            f"image.{view}" in candidates
            for candidates in bm.VIEW_KEY_CANDIDATES.values()
        )


def test_missing_sampler_keys_fall_back_to_the_embodiment():
    config = json.loads(json.dumps(YAM_LEROBOT))
    for key in ("num_steps", "t_shift", "cfg_scale", "cfg_video_scale"):
        del config[key]
    yam = praxis_settings(config)
    assert (yam.t_shift, yam.cfg_video_scale) == (1.0, 1.0)
    config["model_config"]["embodiment"] = "single_arm"
    droid = praxis_settings(config)
    assert (droid.t_shift, droid.cfg_video_scale) == (12.5, 4.0)


def test_a_config_with_another_canvas_is_refused():
    other = json.loads(json.dumps(YAM))
    other["video"]["views"] = ["exterior", "wrist"]
    with pytest.raises(ValueError, match="tile canvas"):
        bm.canvas_size(other)
    other = json.loads(json.dumps(YAM))
    other["video"]["antialias"] = False
    with pytest.raises(ValueError, match="antialiased"):
        PolicyLayout.from_model_config(other)


def test_fit_rule_sizes():
    assert bm.fit_resize_size(1200, 1920) == (260, 416)
    resize, trimmed = bm.fit_row_sizes([(1200, 1920)] * 3)
    assert resize == [(260, 416)] * 3 and trimmed == [(256, 416)] * 3
    # The size the client sends is already the resize size.
    assert bm.fit_row_sizes([(260, 416)] * 3) == (resize, trimmed)


def test_canvas_of_native_and_restricted_views():
    clip = bm.canvas_clip(_views(1200, 1920, t=2))
    assert clip.shape == (1, 3, 2, 256, 1248) and clip.dtype == torch.float32
    small = _views(260, 416)
    clip = bm.canvas_clip(small)
    # A 260x416 view is not resized: the tile is its centre trim, the views side by side.
    for i, view in enumerate(bm.VIEWS):
        want = torch.from_numpy(small[view][0, 2:258]).permute(2, 0, 1).float()
        assert torch.equal(clip[0, :, 0, :, 416 * i : 416 * (i + 1)], want)


def test_canvas_refuses_views_at_another_aspect():
    with pytest.raises(ValueError, match="native aspect"):
        bm.canvas_clip(_views(234, 416))


def test_canvas_needs_equal_frame_counts():
    views = _views(260, 416, t=2)
    views["wrist_right"] = views["wrist_right"][:1]
    with pytest.raises(ValueError, match="frame counts"):
        bm.canvas_clip(views)


def test_views_must_be_uint8():
    obs = _obs()
    obs["image.wrist_left"] = obs["image.wrist_left"].astype(np.float32) / 255
    with pytest.raises(ValueError, match="uint8"):
        bm.view_stacks(obs)


def test_pose16_and_proprio_per_arm():
    pose16 = bm.extract_pose16(_obs(3))
    assert np.array_equal(pose16, np.concatenate([LEFT, RIGHT]))
    p20 = bm.pose16_to_20d(pose16)
    assert np.array_equal(p20[:10], pose8_to_10d(LEFT))
    assert np.array_equal(p20[10:], pose8_to_10d(RIGHT))
    # The proprio keeps the gripper of the rig (1.2): no clip.
    assert p20[19] == pytest.approx(1.2)
    proprio, every = bm.proprio_tensors(pose16[None], _stats(), "cpu")
    assert proprio.shape == (1, 1, 20) and every.shape == (1, 1, 20)


def test_pose16_history_needs_real_rows():
    obs = _obs(25)
    assert bm.pose16_history(obs, 24).shape == (24, 16)
    obs["grip.left"] = obs["grip.left"][-1:]
    assert bm.pose16_history(obs, 24) is None


def test_bad_quaternion_is_refused():
    obs = _obs()
    obs["robot_state.left.ee_pose"] = obs["robot_state.left.ee_pose"] * 2
    with pytest.raises(ValueError, match="quaternion"):
        bm.extract_pose16(obs)


def test_rel20_to_wire_recovers_each_arm_target():
    pose16 = np.concatenate([LEFT, RIGHT])
    target = np.concatenate(
        [
            _pose8([0.45, 0.1, 0.32], _quat(np.array([0.3, 1.0, 0.0]), 0.9), 0.8),
            _pose8([0.55, -0.3, 0.3], _quat(np.array([1.0, 0.2, 0.0]), -0.1), 0.0),
        ]
    )
    rel = bm.pose16_to_20d(target[None], pose16[None])
    wire = bm.rel20_to_wire(rel, pose16, dt=0.1)
    for i, side in enumerate(bm.SIDES):
        pose = np.asarray(wire[0][f"robot_command.{side}"]["pose"])
        arm = target[8 * i : 8 * i + 8]
        assert np.allclose(pose[:3], arm[:3], atol=1e-6)
        rotation = pose[3:].reshape(3, 3).astype(np.float64)
        assert np.allclose(rot6d(rotation), pose8_to_10d(arm)[3:9], atol=1e-5)
        assert wire[0][f"target_grip.{side}"] == pytest.approx(arm[7])
    assert set(wire[0]) == {
        "robot_command.left",
        "robot_command.right",
        "target_grip.left",
        "target_grip.right",
        "timestamp",
    }


def test_decode_actions_clips_and_unpacks():
    stats = _stats()
    tokens = torch.full((1, 6, 80), 3.0)  # far outside [-1, 1]
    rows, rel = bm.decode_actions(tokens, stats, depth=0)
    assert rows.shape == (21, 20) and rows.max() == 3.0
    q99 = stats.action_q99
    assert np.allclose(rel[:, [0, 1, 2, 10, 11, 12]], q99[[0, 1, 2, 10, 11, 12]])
    assert np.all(rel[:, bm.GRIP_DIMS] <= 1.0)
    rows, _ = bm.decode_actions(tokens, stats, depth=1)
    assert rows.shape == (24, 20)


def _encode(clip: torch.Tensor) -> torch.Tensor:
    """A causal stand-in for the VAE: 1 + (T - 1) // 4 latent frames."""
    latents = 1 + (clip.shape[2] - 1) // 4
    return torch.zeros(1, 48, latents, clip.shape[3] // 16, clip.shape[4] // 16)


def _session(**kwargs) -> bm.BimanualSession:
    return bm.BimanualSession(
        encode=_encode,
        stats=_stats(),
        device="cpu",
        max_context_chunks=1,
        dims=bm.q99_dims(YAM),
        **kwargs,
    )


def test_session_builds_context_from_the_executed_frames():
    session = _session()
    first = session.prepare(_obs(25))
    assert first.depth == 0 and first.proprio_all.shape == (1, 1, 20)
    assert first.anchor.shape == (1, 48, 1, 16, 78)
    assert len(session.finish(first, torch.zeros(1, 6, 80))) == 21
    second = session.prepare(_obs(25))
    assert second.depth == 1 and len(second.cleans) == 1
    assert second.proprio_all.shape == (1, 2, 20)
    assert len(session.finish(second, torch.zeros(1, 6, 80))) == 24


def test_session_restarts_on_a_short_stack():
    session = _session()
    session.finish(session.prepare(_obs(25)), torch.zeros(1, 6, 80))
    again = session.prepare(_obs(3))
    assert again.depth == 0 and session.restart_count == 1


def test_session_takes_a_prompt_override():
    session = _session(prompt="old")
    inputs = session.prepare({**_obs(), "prompt": "  Pick up the pen.  "})
    assert inputs.prompt == "Pick up the pen." and session.prompt == "Pick up the pen."


def test_warmup_obs_runs_through_the_session():
    session = _session()
    for step, stack in enumerate((1, 25)):
        inputs = session.prepare(bm.warmup_obs(step=step, stack=stack))
        session.finish(inputs, torch.zeros(1, 6, 80))
    assert inputs.depth == 1


def test_request_state_splits_into_the_two_arms():
    rows = np.stack([np.concatenate([LEFT, RIGHT])] * 3)
    observation = {
        "images": {
            "exterior": _views(260, 416, t=3)["top_left"],
            "wrist_left": _views(260, 416, t=3)["wrist_left"],
            "wrist_right": _views(260, 416, t=3)["wrist_right"],
        },
        "state": rows,
        "prompt": "pick",
    }
    obs = reference_observation(observation=observation, embodiment="bimanual")
    assert np.array_equal(obs["robot_state.left.ee_pose"], rows[:, :7])
    assert np.array_equal(obs["grip.right"], rows[:, 15])
    assert obs["image.wrist_left"].dtype == np.uint8 and obs["task"] == "pick"
    assert bm.extract_pose16(obs).shape == (16,)


def test_commands_become_two_arm_action_rows():
    pose16 = np.concatenate([LEFT, RIGHT])
    wire = bm.rel20_to_wire(bm.pose16_to_20d(pose16[None], pose16[None]), pose16)
    actions = commands_to_actions(commands=wire, embodiment="bimanual")
    assert actions.shape == (1, RIGS["bimanual"].command_dim)
    assert np.allclose(actions[0, :3], LEFT[:3], atol=1e-6)
    assert actions[0, 12] == pytest.approx(LEFT[7])
    assert np.allclose(actions[0, 13:16], RIGHT[:3], atol=1e-6)


def test_a_bimanual_request_needs_all_three_views_and_the_state():
    params = PraxisSamplingParams(observation={"images": {"exterior": 1}})
    params._rig = RIGS["bimanual"]
    with pytest.raises(ValueError, match="wrist_left image, wrist_right image, state"):
        params._validate_observation()
