# SPDX-License-Identifier: Apache-2.0
"""The bimanual (YAM) rig of Praxis-1: config, canvas, proprio, actions, session and request conversion.

CPU only, no weights. A small encoder stands in for the VAE.
"""

import json

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

# fmt: off
_YAM_NORMALIZATION = {
    "domain": "yam",
    "action_q01": [
        -0.1886110007762909, -0.20701800286769867, -0.1853650063276291, 0.46403300762176514,
        -0.6509090065956116, -0.5104100108146667, -0.6505290269851685, 0.45507800579071045,
        -0.5614719986915588, 0.0, -0.1999340057373047, -0.2358579933643341,
        -0.19992700219154358, 0.4093570113182068, -0.6830520033836365, -0.5568550229072571,
        -0.6869149804115295, 0.386911004781723, -0.5972689986228943, 2.6584199730450564e-08,
    ],
    "action_q99": [
        0.18301700055599213, 0.21209199726581573, 0.24737399816513062, 0.9999740123748779,
        0.6522240042686462, 0.6246020197868347, 0.6568049788475037, 0.9999849796295166,
        0.5758579969406128, 1.0, 0.19842800498008728, 0.23249000310897827,
        0.264055997133255, 0.9999549984931946, 0.6877639889717102, 0.6456959843635559,
        0.7018439769744873, 0.9999679923057556, 0.6033779978752136, 1.0,
    ],
    "q99_normed_dims": [
        0, 1, 2, 9,
        10, 11, 12, 19,
    ],
    "proprio_mean": [
        0.6068599820137024, 0.11715299636125565, 0.5181000232696533, -0.5109819769859314,
        -0.09493310004472733, -0.626579999923706, 0.28054600954055786, 0.6581140160560608,
        -0.32249701023101807, 0.5067939758300781, 0.6114770174026489, -0.0875713974237442,
        0.5070589780807495, -0.5247989892959595, -0.0035038399510085583, -0.6334450244903564,
        -0.34640300273895264, 0.6654509902000427, 0.26510700583457947, 0.5514330267906189,
    ],
    "proprio_std": [
        0.1022690013051033, 0.12186499685049057, 0.12526200711727142, 0.34149599075317383,
        0.3882420063018799, 0.2644459903240204, 0.422432005405426, 0.29476499557495117,
        0.34473198652267456, 0.3757230043411255, 0.1051969975233078, 0.12511099874973297,
        0.12934400141239166, 0.34685298800468445, 0.37283799052238464, 0.25299400091171265,
        0.3947789967060089, 0.30974599719047546, 0.33927100896835327, 0.3449360132217407,
    ],
    "action_formula": 'clip(2 * (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1) on q99_normed_dims, identity on the other dims',
    "proprio_formula": '(p - mean) / max(std, 1e-6) on all dims',
}
# fmt: on

# model_config of runwayml/praxis-1-yam-lerobot, reduced to the keys the code reads.
YAM = {
    "embodiment": "bimanual",
    "backbone": {
        "dim": 3072,
        "timestep_dim": 128,
        "num_registers": 4,
        "modulation_dtype": "mixed",
        "dense_ffn_hidden": 11520,
        "layout": "chunk_causal",
        "attention": {
            "head_dim": 128,
            "num_kv_heads": 8,
            "backend": "flash",
            "rope_after_norm": False,
        },
        "rope": {"base": 26000000, "effective_dim": 64, "efficient_cache_max": 1000000},
        "moe": {
            "num_routed_experts": 128,
            "top_k_routed": 8,
            "expert_hidden": 1280,
            "routed_scaling_factor": 2.5,
            "router_activation": "sigmoid",
        },
        "encoder": {
            "depth": 14,
            "dense_blocks": [0, 1],
            "tap_blocks": [0, 2, 4, 6, 7, 9, 11, 13],
            "positive_zero_residual_add": True,
            "input_projections": {
                "TextEncodingTensor": {"c_in": 2048},
                "VideoTensor": {"c_in": 192},
            },
        },
        "decoder": {
            "depth": 6,
            "dense_blocks": [0],
            "unpatch_factor": [1, 2, 2],
            "positive_zero_residual_add": True,
            "input_projections": {"VideoTensor": {"c_in": 48}},
        },
    },
    "action_expert": {
        "dim": 1024,
        "num_layers": 16,
        "num_heads": 16,
        "head_dim": 64,
        "trunk_dim": 3072,
        "action_patch_dim": 80,
        "proprio_patch_dim": 20,
        "num_future_per_chunk": 8,
        "max_chunks": 16,
        "tap_norm": True,
        "clean_only_conditioning": False,
        "head_plus_zero": True,
        "global_order": ["register", "anchor", "text"],
        "rope": {
            "kind": "trunk",
            "order": "before_norm",
            "dim": 64,
            "effective_dim": 64,
            "base": 26000000,
            "table_size": 1000000,
        },
    },
    "chunking": {
        "n_chunks": 4,
        "action_chunk": 24,
        "action_frames_per_token": 4,
        "latents_per_chunk": 6,
        "past_frames": 15,
        "future_frames": 24,
        "window_frames": 93,
        "chunk_geometry": [[0, 21], [21, 24], [45, 24], [69, 24]],
        "grid_hz": 15.0,
        "global_frame_anchor": True,
        "contiguous_chunks": True,
        "context_format": "vxt",
        "proprio_frames": 1,
    },
    "video": {
        "height": 256,
        "width": 1248,
        "views": ["top_left", "wrist_left", "wrist_right"],
        "canvas": "tiles",
        "layout": "hstack",
        "keep_aspect": True,
        "antialias": True,
        "tile_bounds_rows_cols": {
            "top_left": [0, 256, 0, 416],
            "wrist_left": [0, 256, 416, 832],
            "wrist_right": [0, 256, 832, 1248],
        },
        "vae": "Wan2.2",
        "vae_encode_chunk_frames": 4,
        "latent_channels": 48,
        "vae_stride": [4, 16, 16],
        "patch_size": [1, 2, 2],
        "tile_height": 256,
        "tile_width": 416,
        "aspect_tol": 0.03,
        "vae_encode_impl": "megakernel",
        "vae_megakernel_chunk_frames": 16,
    },
    "text_encoder": {
        "model": "Qwen3.5-2B",
        "hidden_layer": -3,
        "pad_to_multiple_of": 32,
        "trim_padding": True,
        "hidden_dim": 2048,
        "uncond": "blank_caption",
        "blank_caption": " ",
    },
    "action_space": {
        "dim": 20,
        "layout": "[L: xyz(3) rot6d(6) grip(1) | R: xyz(3) rot6d(6) grip(1)]",
        "pose_layout": "[L: xyz(3) quat_wxyz(4) grip(1) | R: same]",
    },
    "normalization": _YAM_NORMALIZATION,
    "pose_frame": "rig",
}

# The outer LeRobot config of the same checkpoint, reduced to the keys the code reads.
YAM_LEROBOT = {
    "type": "praxis",
    "camera_keys": [
        "observation.images.top_left",
        "observation.images.wrist_left",
        "observation.images.wrist_right",
    ],
    "chunk_size": 24,
    "n_action_steps": 24,
    "max_context_chunks": 1,
    "num_steps": 2,
    "t_shift": 1.0,
    "cfg_scale": 1.0,
    "cfg_video_scale": 1.0,
    "seed": 0,
    "vae_id": "Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth",
    "text_encoder_id": "Qwen/Qwen3.5-2B",
    "model_config": YAM,
}


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
