# SPDX-License-Identifier: Apache-2.0

import copy
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.dm05 import DM05PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.dm05_preprocess import (
    DM05NormStats,
    DM05Preprocessor,
    _quaternion_to_rotvec,
    _rotvec_to_quaternion,
    _rpy_to_axis_angle,
    collate_dm05_observations,
    denormalize_dm05_vector,
    dm05_observation_signature,
    dm05_state_bins,
    normalize_dm05_vector,
    postprocess_dm05_actions,
    preprocess_dm05_image,
)


def _stats(size, lower=-1.0, upper=1.0):
    return {
        "q01": [lower] * size,
        "q99": [upper] * size,
        "mean": [0.0] * size,
        "std": [1.0] * size,
    }


def _norm(size=7):
    return {"norm_stats": {"state": _stats(size), "action": _stats(size)}}


def _config(**overrides):
    fields = dict(
        processor_name=None,
        norm_stats_path=None,
        image_keys=("front", "wrist"),
        image_prompts=("Head", "Left wrist"),
        image_size=(8, 8),
        max_token_len=2048,
        action_horizon=3,
        action_dim=32,
        output_action_dim=7,
        add_state=True,
        n_state_bins=256,
        is_history=False,
        max_history_images=2,
        clip_to_bounds=True,
        default_robot_type="Franka",
        default_control_mode=None,
        default_speed="0.5",
        state_desc=None,
        use_absolute_action=False,
        compose_eef_rot=False,
        use_transformed_state=False,
        denorm_state=False,
        full_action_mask=False,
        state_layout="identity",
        reject_prompt_truncation=False,
        prompt_is_formatted=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class _RecordingProcessor:
    """Small tokenizer boundary fixture; real processor parity is tested separately."""

    pad_token_id = 0

    def __init__(self):
        self.tokenizer = self
        self.calls = []

    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, tokens, **kwargs):
        return "".join(map(chr, tokens))

    def convert_tokens_to_ids(self, text):
        return {"<unused0>": 6, "<unused1>": 7}[text]

    def image_processor(self, *, images, **kwargs):
        return {
            "pixel_values": torch.stack(
                [
                    torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5
                    - 1
                    for image in images
                ]
            )
        }

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs.copy()))
        tokens = [2]
        token_types = [0]
        images = []
        for item in messages[0]["content"]:
            if item["type"] == "image":
                images.append(item["image"])
                tokens.extend([200, 200])
                token_types.extend([1, 1])
                continue
            parts = re.findall(r"<unused[01]>|.", item["text"], flags=re.DOTALL)
            tokens.extend(
                self.convert_tokens_to_ids(part)
                if part.startswith("<unused")
                else ord(part) + 20
                for part in parts
            )
            token_types.extend([0] * len(parts))
        if kwargs.get("add_generation_prompt"):
            tokens.append(3)
            token_types.append(0)
        return {
            "input_ids": torch.tensor([tokens]),
            "attention_mask": torch.ones(1, len(tokens), dtype=torch.long),
            "token_type_ids": torch.tensor([token_types]),
            **self.image_processor(images=images),
        }


def _request(**overrides):
    value = {
        "prompt": "move_under\nblock",
        "images": {
            "front": np.zeros((5, 8, 3), dtype=np.uint8),
            "wrist": np.full((7, 3, 3), 255, dtype=np.uint8),
        },
        "state": [0.0] * 7,
    }
    value.update(overrides)
    return value


def _preprocessor(config=None, norm=None, processor=None):
    return DM05Preprocessor(
        config or _config(),
        processor=processor or _RecordingProcessor(),
        norm_stats=norm or _norm(),
    )


@pytest.mark.parametrize("robot", ["unknown", {"name": "Franka"}, ["Franka"], False])
def test_legacy_norm_with_configured_state_desc_rejects_unknown_robot(robot):
    preprocessor = _preprocessor(_config(state_desc=("joint",) * 6 + ("gripper",)))
    with pytest.raises(ValueError, match="unsupported robot_type"):
        preprocessor(_request(robot_type=robot))


def test_quantile_scaling_keeps_reference_epsilon_and_zero_dimensions():
    stats = _stats(3)
    stats["q01"] = [-2.0, 0.0, 1.0]
    stats["q99"] = [2.0, 0.0, 1.000001]
    value = np.array([0.0, 1.0, 1.000001], dtype=np.float32)
    expected = (value - np.asarray(stats["q01"], dtype=np.float32)) / (
        np.asarray(stats["q99"], dtype=np.float32)
        - np.asarray(stats["q01"], dtype=np.float32)
        + 1e-6
    ) * 2 - 1
    expected[1] = 0
    np.testing.assert_array_equal(normalize_dm05_vector(value, stats), expected)
    denormalized = denormalize_dm05_vector(expected, stats)
    np.testing.assert_allclose(denormalized[[0, 2]], value[[0, 2]], atol=1e-7)
    assert denormalized[1] == 0


def test_quantile_clipping_is_optional_and_not_applied_to_outputs():
    stats = _stats(1)
    assert normalize_dm05_vector([2.0], stats)[0] < 1
    assert normalize_dm05_vector([2.0], stats, clip_to_bounds=False)[0] > 1
    assert denormalize_dm05_vector([2.0], stats)[0] > 1


def test_state_discretization_matches_floor_rule_at_boundaries():
    assert dm05_state_bins([-2, -1, -0.999, -0.5, 0, 0.5, 1, 2]) == [
        0,
        0,
        0,
        63,
        127,
        191,
        255,
        255,
    ]
    with pytest.raises(ValueError, match="n_state_bins"):
        dm05_state_bins([0], 1)


def test_multi_robot_profiles_and_control_mode_selection_are_exact():
    first = _norm()["norm_stats"]
    second = {"state": _stats(7, -2, 2), "action": _stats(7, -3, 3)}
    norms = DM05NormStats(
        {
            "norm_stats": first,
            "default_robot_type": "robot-a",
            "norm_stats_by_robot": {"robot-a": first, "robot-b": second},
        }
    )
    assert norms.select(None) is first
    assert norms.select("robot-b") is second
    with pytest.raises(ValueError, match="no robot profile"):
        norms.select("missing")
    modes = DM05NormStats({"norm_stats": first, "control_modes": {"eef": second}})
    assert modes.select("legacy-robot", "end effector") is second
    with pytest.raises(ValueError, match="control mode"):
        modes.select(None, "joint")


def test_wrong_default_profile_and_inconsistent_stats_are_rejected():
    first = _norm()["norm_stats"]
    with pytest.raises(ValueError, match="default robot profile"):
        DM05NormStats(
            {
                "norm_stats": first,
                "default_robot_type": "a",
                "norm_stats_by_robot": {
                    "a": {"state": _stats(7, -2, 2), "action": _stats(7)}
                },
            }
        )
    invalid = _norm()
    invalid["norm_stats"]["state"]["q99"] = [0.0]
    with pytest.raises(ValueError, match="equal-length"):
        DM05NormStats(invalid)


def test_image_padding_keeps_original_uint8_pixels_and_odd_extra_pixel_bottom():
    source = np.full((2, 5, 3), [11, 23, 47], dtype=np.uint8)
    actual = np.asarray(preprocess_dm05_image(source, (5, 5)))
    np.testing.assert_array_equal(actual[1:3], source)
    assert not actual[0].any()
    assert not actual[3:].any()


def test_rgb_conversion_accepts_pil_grayscale_and_chw_tensors():
    gray = Image.fromarray(np.full((3, 3), 128, dtype=np.uint8))
    expected = np.full((3, 3, 3), 128, dtype=np.uint8)
    np.testing.assert_array_equal(preprocess_dm05_image(gray, (3, 3)), expected)
    np.testing.assert_array_equal(
        preprocess_dm05_image(torch.from_numpy(expected).permute(2, 0, 1), (3, 3)),
        expected,
    )
    with pytest.raises(ValueError, match="uint8"):
        preprocess_dm05_image(np.full((3, 3, 3), 0.5))


@pytest.mark.parametrize("pixel", ["x", None])
@pytest.mark.parametrize("history", [False, True])
def test_non_numeric_current_and_history_pixels_are_rejected(pixel, history):
    processor = _RecordingProcessor()
    request = _request()
    pixels = [[[pixel] * 3]]
    if history:
        request["history_images"] = [pixels]
    else:
        request["images"]["front"] = pixels
    with pytest.raises(ValueError, match="numeric"):
        _preprocessor(_config(is_history=history), processor=processor)(request)
    assert not processor.calls


def test_prompt_retains_underscores_newlines_and_exact_robot_metadata():
    processor = _RecordingProcessor()
    observation = _preprocessor(processor=processor)(
        _request(control_mode="eef", speed="0.2")
    )
    content = processor.calls[0][0][0]["content"]
    assert content[0]["text"] == (
        "Robot: Franka\nControl mode: end effector\nOverall speed: 0.2\n"
        "Task: move_under\nblock.\nHead image: "
    )
    assert content[2]["text"] == "Left wrist image: "
    assert content[-1]["text"] == "States: " + " ".join(["127"] * 7)
    fields = observation.metadata["dm05"]
    assert fields["pixel_values"].shape == (2, 3, 8, 8)
    assert fields["action_mask"].sum() == 7
    assert fields["action_mask"].shape == (1, 1, 32)
    assert observation.batch_size == 1


def test_formatted_prompt_and_no_state_text_options_are_preserved():
    processor = _RecordingProcessor()
    _preprocessor(
        _config(prompt_is_formatted=True, add_state=False), processor=processor
    )(_request(prompt="Task: first.\nSubtask: next."))
    content = processor.calls[0][0][0]["content"]
    assert "Task: first.\nSubtask: next.\nHead image: " in content[0]["text"]
    assert not any("States:" in item.get("text", "") for item in content)


def test_released_memory_prompt_keeps_single_task_prefix_and_punctuation():
    processor = _RecordingProcessor()
    config = _config(
        is_history=True,
        normalize_task_prompt=True,
        prompt_is_formatted=True,
        default_robot_type="Dual ARX5",
        state_desc=(["joint"] * 6 + ["gripper"]) * 2,
        output_action_dim=14,
    )
    _preprocessor(config, norm=_norm(14), processor=processor)(
        _request(prompt="  Task: pick!  ", state=[0.0] * 14)
    )
    text = processor.calls[0][0][0]["content"][0]["text"]
    assert "Task: pick!\nHistory images:" in text
    assert "Task: Task:" not in text
    assert text.startswith("Robot: Dual ARX5\nOverall speed: 0.5\n")


def test_released_memory_truncation_counts_raw_prompt_before_formatting():
    class PromptRecordingProcessor(_RecordingProcessor):
        def __init__(self):
            super().__init__()
            self.encoded_prompts = []

        def encode(self, text, **kwargs):
            self.encoded_prompts.append(text)
            return super().encode(text, **kwargs)

    processor = PromptRecordingProcessor()
    config = _config(
        is_history=True,
        normalize_task_prompt=True,
        prompt_is_formatted=True,
        max_token_len=220,
        default_robot_type="Dual ARX5",
        state_desc=(["joint"] * 6 + ["gripper"]) * 2,
        output_action_dim=14,
    )
    prompt = "  move the cube " * 100
    observation = _preprocessor(config, norm=_norm(14), processor=processor)(
        _request(prompt=prompt, state=[0.0] * 14)
    )
    assert processor.encoded_prompts == [prompt]
    assert len(processor.calls) == 2
    assert observation.tokens.shape[1] <= config.max_token_len
    for messages, _ in processor.calls:
        text = messages[0]["content"][0]["text"]
        assert "Task: move the cube" in text
        assert ".\nHistory images:" in text
        assert "Task: Task:" not in text


def test_history_keeps_fixed_slots_masks_and_frame_order():
    processor = _RecordingProcessor()
    config = _config(is_history=True)
    request = _request(history_images=[np.full((3, 3, 3), 128, dtype=np.uint8)])
    observation = _preprocessor(config, processor=processor)(request)
    text = processor.calls[0][0][0]["content"][0]["text"]
    assert (
        "History images: " + "<unused1>" * 16 + "<unused0>" * 16 + "\nHead image: "
        in text
    )
    fields = observation.metadata["dm05"]
    assert fields["history_mask"].sum() == 16
    assert (observation.tokens == 7).sum() == 16
    assert fields["token_type_ids"][fields["history_mask"]].eq(1).all()
    assert fields["history_pixel_values"].shape == (1, 3, 8, 8)
    assert fields["history_pixel_values"].mean().item() == pytest.approx(
        128 / 127.5 - 1, abs=1e-7
    )
    empty = _preprocessor(config)(_request())
    assert (empty.tokens == 7).sum() == 32
    assert empty.metadata["dm05"]["history_mask"].sum() == 0


@pytest.mark.parametrize(
    "changes,field",
    [
        ({"control_mode": {}}, "control_mode"),
        ({"control_mode": ["eef"]}, "control_mode"),
        ({"control_mode": 7}, "control_mode"),
        ({"image_masks": [True]}, "image_masks"),
        ({"image_masks": False}, "image_masks"),
        ({"image_masks": {"unknown": True}}, "image_masks"),
        ({"image_masks": {"front": 1}}, "image_masks"),
        ({"image_masks": {"front": "true"}}, "image_masks"),
        ({"image_masks": {"front": [True]}}, "image_masks"),
        ({"camera_order": 7}, "camera_order"),
        ({"camera_order": "front"}, "camera_order"),
        ({"camera_order": ["front", {}]}, "camera_order"),
        ({"camera_order": []}, "camera_order"),
        ({"camera_order": ["front", "front"]}, "camera_order"),
    ],
)
def test_malformed_layout_and_control_fields_raise_value_errors(changes, field):
    processor = _RecordingProcessor()
    with pytest.raises(ValueError, match=field):
        _preprocessor(processor=processor)(_request(**changes))
    assert not processor.calls


def test_explicit_camera_order_and_boolean_visibility_preserve_valid_input():
    preprocessor = _preprocessor()
    implicit = preprocessor(_request())
    explicit = preprocessor(
        _request(
            camera_order=["front", "wrist"],
            image_masks={"front": True, "wrist": np.bool_(True)},
        )
    )
    torch.testing.assert_close(explicit.tokens, implicit.tokens, rtol=0, atol=0)
    torch.testing.assert_close(
        explicit.metadata["dm05"]["pixel_values"],
        implicit.metadata["dm05"]["pixel_values"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"state": None}, "state is required"),
        ({"state": [0.0] * 6}, "7 dimensions"),
        ({"state": [float("nan")] * 7}, "finite"),
        ({"camera_order": ["wrist", "front"]}, "camera_order"),
        ({"images": {}}, "configured views"),
        ({"image_masks": {"front": False}}, "masked cameras"),
        ({"noise": np.zeros((2, 32))}, "noise must have shape"),
        ({"history_images": [np.zeros((3, 3, 3), dtype=np.uint8)]}, "is_history"),
    ],
)
def test_invalid_observations_are_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        _preprocessor()(_request(**changes))


def test_history_limit_and_noise_finiteness_are_checked():
    with pytest.raises(ValueError, match="at most 2"):
        _preprocessor(_config(is_history=True))(
            _request(history_images=[np.zeros((3, 3, 3), dtype=np.uint8)] * 3)
        )
    with pytest.raises(ValueError, match="finite"):
        _preprocessor()(_request(noise=np.full((3, 32), np.nan)))


def test_full_mask_and_explicit_noise_are_preserved_before_denoising():
    noise = np.arange(96, dtype=np.float32).reshape(3, 32)
    obs = _preprocessor(_config(full_action_mask=True))(_request(noise=noise))
    assert obs.metadata["dm05"]["action_mask"].eq(1).all()
    np.testing.assert_array_equal(obs.noise[0], noise)


def test_prompt_truncation_retries_public_chat_template_contract():
    processor = _RecordingProcessor()
    _preprocessor(_config(max_token_len=250), processor=processor)(
        _request(prompt="q" * 300)
    )
    assert len(processor.calls) == 2
    assert processor.calls[0][1]["add_generation_prompt"] is True
    assert "add_generation_prompt" not in processor.calls[1][1]
    assert "q" * 300 not in processor.calls[1][0][0]["content"][0]["text"]
    with pytest.raises(ValueError, match="exceeds configured"):
        _preprocessor(_config(max_token_len=250, reject_prompt_truncation=True))(
            _request(prompt="q" * 300)
        )
    with pytest.raises(ValueError, match="fixed image/history"):
        _preprocessor(_config(max_token_len=3))(_request())


def test_collation_keeps_same_shape_different_content_and_per_robot_postprocessing():
    norm_a = _norm()
    norm_b = _norm()
    norm_b["norm_stats"]["action"] = _stats(7, 10, 20)
    first = _preprocessor(norm=norm_a)(_request())
    request_b = _request(state=[0.5] * 7)
    request_b["images"]["front"] = np.full((5, 8, 3), 127, dtype=np.uint8)
    second = _preprocessor(norm=norm_b)(request_b)
    assert dm05_observation_signature(first) == dm05_observation_signature(second)
    grouped = collate_dm05_observations([first, second])
    assert grouped.batch_size == 2
    assert grouped.metadata["dm05"]["pixel_values"].shape == (4, 3, 8, 8)
    assert grouped.metadata["dm05"]["action_mask"].shape == (2, 1, 32)
    outputs = postprocess_dm05_actions(torch.zeros(2, 3, 32), grouped, _config())
    assert outputs.shape == (2, 3, 7)
    np.testing.assert_allclose(outputs[0], 5e-7, atol=1e-7)
    np.testing.assert_allclose(outputs[1], 15, atol=1e-6)


def test_collation_rejects_incompatible_shapes_and_empty_inputs():
    first = _preprocessor()(_request())
    second = _preprocessor()(_request(prompt="a different prompt length"))
    with pytest.raises(ValueError, match="incompatible"):
        collate_dm05_observations([first, second])
    with pytest.raises(ValueError, match="empty"):
        collate_dm05_observations([])


def test_noise_devices_are_part_of_grouping_compatibility():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.vla import (
        VLAPrefixEncodingStage,
    )

    config = _config(enable_global_prefix_cache=False)
    preprocessor = _preprocessor(config)
    noise = np.zeros((3, 32), dtype=np.float32)
    first = preprocessor(_request(noise=noise))
    second = preprocessor(_request(noise=noise))
    args = SimpleNamespace(
        pipeline_config=config,
        disable_conditioning_cache=True,
        conditioning_cache_max_size_mb=0,
    )
    stage = VLAPrefixEncodingStage(
        None,
        None,
        observation_collator=collate_dm05_observations,
        observation_signature=dm05_observation_signature,
    )

    def request(observation):
        return SimpleNamespace(
            extra={"vla": {"observation_batch": observation}},
            is_warmup=False,
            generator=None,
            action_horizon=3,
            action_dim=32,
            num_inference_steps=10,
        )

    assert dm05_observation_signature(first) == dm05_observation_signature(second)
    assert collate_dm05_observations([first, second]).noise.device.type == "cpu"
    # Meta exercises the device boundary without requiring CUDA.
    second.noise = second.noise.to("meta")
    assert dm05_observation_signature(first) != dm05_observation_signature(second)
    assert stage._observation_fingerprint(request(first), args) != (
        stage._observation_fingerprint(request(second), args)
    )
    with pytest.raises(ValueError, match="incompatible"):
        collate_dm05_observations([first, second])
    first.noise = first.noise.to("meta")
    assert dm05_observation_signature(first) == dm05_observation_signature(second)
    assert collate_dm05_observations([first, second]).noise.device.type == "meta"


def test_history_collation_keeps_empty_and_nonempty_requests_separate_in_mask():
    processor = _preprocessor(_config(is_history=True))
    first = processor(_request())
    second = processor(
        _request(history_images=[np.full((3, 3, 3), 128, dtype=np.uint8)])
    )
    extra = second.tokens.shape[1] - first.tokens.shape[1]
    assert extra > 0
    first.tokens = torch.nn.functional.pad(first.tokens, (0, extra))
    first.token_masks = torch.nn.functional.pad(first.token_masks, (0, extra))
    for key in ("history_mask", "token_type_ids"):
        first.metadata["dm05"][key] = torch.nn.functional.pad(
            first.metadata["dm05"][key], (0, extra)
        )
    assert dm05_observation_signature(first) == dm05_observation_signature(second)
    grouped = collate_dm05_observations([first, second])
    fields = grouped.metadata["dm05"]
    assert grouped.batch_size == 2
    assert fields["history_mask"].sum(dim=1).tolist() == [0, 16]
    assert fields["history_pixel_values"].shape == (1, 3, 8, 8)


def test_relative_postprocessing_retains_gripper_and_original_state():
    config = _config(use_absolute_action=True)
    obs = _preprocessor(config)(_request(state=[0.5] * 7))
    actual = postprocess_dm05_actions(np.zeros((1, 3, 32)), obs, config)
    assert actual.shape == (1, 3, 7)
    np.testing.assert_allclose(actual[0, :, :6], 0.5000005, atol=1e-7)
    np.testing.assert_allclose(actual[0, :, 6], 5e-7, atol=1e-7)


def test_absolute_reconstruction_requires_robot_or_state_description():
    config = _config(default_robot_type=None, use_absolute_action=True)
    with pytest.raises(ValueError, match="robot_type.*state_desc"):
        _preprocessor(config)(_request())


@pytest.mark.parametrize(
    "description_source", ["request", "config", "norm", "state_desc", "relative"]
)
def test_absolute_reconstruction_preserves_resolved_grippers(description_source):
    config = _config(
        default_robot_type=None,
        output_action_dim=14,
        use_absolute_action=description_source != "relative",
    )
    norm = _norm(14)
    request = _request(state=[0.5] * 14)
    if description_source == "request":
        request["robot_type"] = "DOS W1"
    elif description_source == "config":
        config.default_robot_type = "DOS W1"
    elif description_source == "norm":
        norm["default_robot_type"] = "DOS W1"
        norm["norm_stats_by_robot"] = {"DOS W1": copy.deepcopy(norm["norm_stats"])}
    elif description_source == "state_desc":
        config.state_desc = (["joint"] * 6 + ["gripper"]) * 2
    observation = _preprocessor(config, norm=norm)(request)
    normalized = np.full((1, 3, 32), -5e-7, dtype=np.float32)
    actions = postprocess_dm05_actions(normalized, observation, config)
    predicted = denormalize_dm05_vector(
        normalized[..., :14], norm["norm_stats"]["action"]
    )
    np.testing.assert_array_equal(actions[..., [6, 13]], predicted[..., [6, 13]])


def test_eef_preprocessing_and_absolute_rotation_keep_identity_round_trip():
    config = _config(
        compose_eef_rot=True,
        use_transformed_state=True,
        denorm_state=True,
        use_absolute_action=True,
    )
    state = np.asarray([0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.5], dtype=np.float32)
    obs = _preprocessor(config)(_request(state=state, control_mode="eef"))
    actual = postprocess_dm05_actions(np.full((1, 3, 32), -5e-7), obs, config)
    np.testing.assert_allclose(
        actual[0, :, :6], np.broadcast_to(state[:6], (3, 6)), atol=2e-7
    )
    np.testing.assert_allclose(actual[0, :, 6], 0, atol=1e-7)


def test_eef_rotation_preserves_reference_float32_rounding():
    # Captured from the public OpenDM transforms at fe1ed79a, NumPy 1.26.4.
    # In particular, casting before quaternion normalization and retaining the
    # division parentheses affects the returned actions by several FP32 ULPs.
    fixtures = [
        (
            _rpy_to_axis_angle,
            np.array([0.0, 0.0, np.pi / 2], dtype=np.float32),
            [0, 0, 1070141402],
        ),
        (
            _quaternion_to_rotvec,
            np.array([0.71, 0.12, -0.31, 0.62], dtype=np.float64),
            [1049125355, 3207604517, 1068509477],
        ),
        (
            _rotvec_to_quaternion,
            np.array([0.23, -0.84, 1.72], dtype=np.float64),
            [1058143524, 1036563415, 3199680519, 1060871419],
        ),
    ]
    for convert, value, expected_bits in fixtures:
        actual = convert(value)
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual.view(np.uint32), expected_bits)


def test_eef_delta_rotation_composes_delta_before_current_rotation():
    config = _config(
        compose_eef_rot=True,
        use_transformed_state=True,
        denorm_state=True,
        use_absolute_action=True,
        clip_to_bounds=False,
    )
    norms = {
        "norm_stats": {
            "state": _stats(7, -np.pi, np.pi),
            "action": _stats(7, -np.pi, np.pi),
        }
    }
    state = np.array([0, 0, 0, 0, 0, np.pi / 2, 0.4], dtype=np.float32)
    desired_delta = np.array([0, 0, 0, np.pi / 2, 0, 0, 0.3], dtype=np.float32)
    obs = _preprocessor(config, norm=norms)(_request(state=state, control_mode="eef"))
    noise = np.zeros((1, 3, 32), dtype=np.float32)
    noise[..., :7] = normalize_dm05_vector(
        desired_delta, norms["norm_stats"]["action"], clip_to_bounds=False
    )
    actual = postprocess_dm05_actions(noise, obs, config)
    expected_rotation = np.array([1, -1, 1]) / np.sqrt(3) * (2 * np.pi / 3)
    np.testing.assert_allclose(
        actual[0, :, 3:6], np.broadcast_to(expected_rotation, (3, 3)), atol=1e-6
    )
    np.testing.assert_allclose(actual[0, :, 6], 0.3, atol=1e-6)


@pytest.mark.parametrize(
    "layout,size", [("robocoliseum_g2", 16), ("robocoliseum_manipulation", 21)]
)
def test_robocoliseum_shared_layout_and_torso_reconstruction(layout, size):
    config = _config(
        state_layout=layout,
        output_action_dim=22,
        default_robot_type="Agibot G2",
        use_absolute_action=True,
    )
    state = np.arange(size, dtype=np.float32) / size
    obs = _preprocessor(config, norm=_norm(22))(_request(state=state, prompt="lift"))
    normalized = np.full((1, 3, 32), -5e-7, dtype=np.float32)
    actions = postprocess_dm05_actions(normalized, obs, config)
    assert actions.shape == (1, 3, size)
    metadata = DM05PipelineConfig(
        state_layout=layout, state_dim=size, output_action_dim=22
    ).action_metadata(SimpleNamespace(served_model_name="dm05", model_path="local"))
    assert metadata["output"]["action_dim"] == actions.shape[-1]
    joint_indices = [i for i in range(size) if i not in (7, 15)]
    np.testing.assert_allclose(
        actions[0, :, joint_indices],
        np.broadcast_to(state[joint_indices, None], (len(joint_indices), 3)),
        atol=2e-7,
    )
    np.testing.assert_allclose(actions[0, :, [7, 15]], 0, atol=1e-7)


def test_postprocessing_preserves_batch_and_rejects_missing_contexts():
    obs = _preprocessor()(_request())
    with pytest.raises(ValueError, match="one action chunk"):
        postprocess_dm05_actions(np.zeros((2, 3, 32)), obs, _config())
    obs.metadata["dm05_postprocess"] = []
    with pytest.raises(ValueError, match="contexts"):
        postprocess_dm05_actions(np.zeros((1, 3, 32)), obs, _config())
