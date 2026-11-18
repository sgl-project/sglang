# SPDX-License-Identifier: Apache-2.0
"""Image request validation, plan semantics and release admission."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
    MiniMaxH3ReleaseMetadata,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    MINIMAX_H3_MAX_SIGNED_SEED,
    MINIMAX_H3_REQUEST_SCHEMA,
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY,
    MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY,
    MiniMaxH3ResolvedPlan,
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.task_profiles import (
    MINIMAX_H3_TASK_PARTITIONS,
    minimax_h3_task_profile,
    partition_for_task,
)

REFERENCE = {"role": "reference", "type": "image", "uri": "file:///reference.png"}
TARGET = {"width": 1024, "height": 768}


def _request(task="t2i", **overrides):
    request = {
        "task": task,
        "prompt": "p",
        "conditions": [dict(REFERENCE)] if task == "edit" else [],
        "target": dict(TARGET),
        "seed": 0,
    }
    request.update(overrides)
    return minimax_h3_validate_canonical_request(**request)


def _image_index(tasks=("t2i", "edit")):
    return {
        "_class_name": "MiniMaxH3ImagePipeline",
        "transformer": ["diffusers", "MiniMaxH3DiTModel"],
        "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
        "tokenizer": ["transformers", "Qwen2TokenizerFast"],
        "processor": ["transformers", "Qwen3VLProcessor"],
        "vae": ["diffusers", "MiniMaxH3ImageVAE"],
        "scheduler": None,
        "_minimax_h3": {
            "schema_version": 1,
            "partition": "image",
            "tasks": list(tasks),
            "task_aliases": {},
        },
    }


def _video_index(partition="fl2va"):
    return {
        "_class_name": "MiniMaxH3Pipeline",
        "_minimax_h3": {
            "schema_version": 1,
            "partition": partition,
            "tasks": ["t2va", "fl2va"] if partition == "fl2va" else ["ref2va"],
            "task_aliases": {},
            "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
        },
    }


@pytest.mark.parametrize("task", ["t2i", "edit"])
@pytest.mark.parametrize("prompt", ["p", "  verbatim\n<input>\u6587\u5b57</input>  "])
def test_image_prompts_and_seed_are_preserved(task, prompt):
    canonical = _request(task, prompt=prompt)
    assert canonical["schema"] == MINIMAX_H3_REQUEST_SCHEMA
    assert canonical["prompt"] == prompt
    assert canonical["seed"] == 0
    plan = minimax_h3_resolve_plan(canonical)
    assert plan.prompt == prompt
    assert plan.branches == ({"name": "positive"},)
    assert (
        msgspec.json.decode(msgspec.json.encode(plan), type=MiniMaxH3ResolvedPlan)
        == plan
    )


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_optional_seed(task):
    canonical = _request(task, seed=None)
    assert "seed" not in canonical
    assert minimax_h3_resolve_plan(canonical).seed is None
    seed = MINIMAX_H3_MAX_SIGNED_SEED
    assert _request(task, seed=seed)["seed"] == seed


# One case per validation rule; the HTTP layer maps these ValueErrors to 400.
@pytest.mark.parametrize(
    "task,overrides,match",
    [
        ("t2i", {"prompt": None}, "prompt"),
        ("t2i", {"prompt": ""}, "prompt must be a non-empty string"),
        ("edit", {"prompt": " \n"}, "prompt must be a non-empty string"),
        ("t2i", {"negative_prompt": ""}, "negative_prompt"),
        ("t2i", {"seed": True}, "seed"),
        ("t2i", {"seed": -1}, "seed"),
        ("t2i", {"target": {"width": 1000, "height": 768}}, "target.width"),
        ("t2i", {"target": {"width": 1024}}, "target.height"),
        ("t2i", {"target": {**TARGET, "fps": 24}}, "fps"),
        (
            "t2i",
            {"target": {"width": 0, "height": 768, "min_generation_area": 1024}},
            "target.width must be positive",
        ),
        (
            "t2i",
            {"target": {**TARGET, "min_generation_area": 1023}},
            "min_generation_area",
        ),
        (
            "t2i",
            {"target": {**TARGET, "min_generation_area": 4096 * 4096 + 1}},
            "min_generation_area",
        ),
        (
            "t2i",
            {"target": {**TARGET, "min_generation_area": True}},
            "min_generation_area",
        ),
        (
            "t2i",
            {"target": {**TARGET, "min_generation_area": 4194304.0}},
            "min_generation_area",
        ),
        ("t2i", {"num_frames": None}, "num_frames"),
        ("t2i", {"flow_shift": 12.0}, "flow_shift"),
        ("t2i", {"schema": "minimax_h3.request/v2"}, "schema"),
        ("t2i", {"conditions": [REFERENCE]}, "conditions must be empty"),
        ("edit", {"conditions": []}, "at least one"),
        ("edit", {"conditions": [{**REFERENCE, "uri": " "}]}, "uri"),
        ("edit", {"conditions": [{**REFERENCE, "type": "video"}]}, r"conditions\[0\]"),
        ("edit", {"conditions": [{**REFERENCE, "frame_index": 0}]}, "frame_index"),
    ],
)
def test_invalid_image_requests_are_rejected(task, overrides, match):
    with pytest.raises(ValueError, match=match):
        _request(task, **overrides)


@pytest.mark.parametrize("count", [1, 3])
def test_edit_preserves_reference_order_in_the_plan(count):
    references = [
        {**REFERENCE, "uri": f"file:///{index}.png"} for index in reversed(range(count))
    ]
    original = deepcopy(references)
    canonical = _request("edit", conditions=references)
    plan = minimax_h3_resolve_plan(canonical)
    assert references == original
    assert canonical["conditions"] == references
    assert [item.uri for item in plan.materials] == [ref["uri"] for ref in references]
    assert [item.condition_index for item in plan.materials] == list(range(count))
    assert {item.material_chain for item in plan.materials} == {
        "image.reference_preserve"
    }
    assert all(item.start_time_seconds is None for item in plan.materials)
    assert plan.encoders["qwen"]["ordered_condition_indices"] == list(range(count))
    assert plan.encoders["visual"] == list(range(count))
    assert plan.encoders["audio"] == []
    assert plan.condition_mask == {}


@pytest.mark.parametrize("task", ["t2i", "edit"])
@pytest.mark.parametrize("width,height", [(32, 32), (64, 320), (1024, 768)])
def test_image_plan_is_exact_single_frame_and_no_audio(task, width, height):
    canonical = _request(task, target={"width": width, "height": height})
    original = deepcopy(canonical)
    plan = minimax_h3_resolve_plan(canonical)
    assert canonical == original
    assert plan.shape == {
        "geometry": "resolved_v2",
        "width": width,
        "height": height,
        "output_width": None,
        "output_height": None,
        "fps": None,
        "frame_count": 1,
        "video_latent_t": 1,
        "video_latent_h": height // 16,
        "video_latent_w": width // 16,
        "audio_latent_t": 0,
    }
    assert plan.default_flow_shift is plan.default_audio_flow_shift is None
    assert plan.flow_shift is plan.audio_flow_shift is None
    assert "audio_vae" not in minimax_h3_task_profile(task).required_components
    if task == "t2i":
        assert plan.materials == () and plan.encoders["visual"] == []


M = 2048 * 2048


@pytest.mark.parametrize(
    "size,min_area,canvas",
    [
        # Enlarged by sqrt(M / area), each side on the nearest 32px multiple.
        ((1024, 1024), M, (2048, 2048)),
        ((810, 810), M, (2048, 2048)),
        ((1280, 720), M, (2720, 1536)),
        ((1000, 1500), M, (1664, 2496)),
        ((1024, 768), M, (2368, 1760)),
        ((1, 1), M, (2048, 2048)),
        # At or above the area: only aligned to the grid.
        ((2720, 1536), M, (2720, 1536)),
        ((3000, 2000), M, (3008, 1984)),
        ((5472, 3072), M, (5472, 3072)),
        ((33, 47), 1024, (32, 32)),
        # The smallest area never enlarges an aligned request.
        ((64, 320), 1024, (64, 320)),
        ((1024, 768), 4096 * 4096, (4736, 3552)),
    ],
)
def test_min_generation_area_canvas(size, min_area, canvas):
    width, height = size
    target = {"width": width, "height": height, "min_generation_area": min_area}
    canonical = _request("t2i", target=target)
    assert canonical["target"] == target
    shape = minimax_h3_resolve_plan(canonical).shape
    assert (shape["width"], shape["height"]) == canvas
    assert (shape["video_latent_w"], shape["video_latent_h"]) == (
        canvas[0] // 16,
        canvas[1] // 16,
    )
    resized = canvas != size
    assert (shape["output_width"], shape["output_height"]) == (
        size if resized else (None, None)
    )


def test_min_generation_area_plan_matches_requesting_the_canvas():
    """Noise, latents and positions read plan.shape, so a small request is the
    canvas request plus the resize."""
    small = minimax_h3_resolve_plan(
        _request(
            "edit", target={"width": 1024, "height": 1024, "min_generation_area": M}
        )
    )
    canvas = minimax_h3_resolve_plan(
        _request(
            "edit", target={"width": 2048, "height": 2048, "min_generation_area": M}
        )
    )
    assert (small.shape["output_width"], small.shape["output_height"]) == (1024, 1024)
    unresized = {**small.shape, "output_width": None, "output_height": None}
    assert msgspec.structs.replace(small, shape=unresized) == canvas


def _admit(metadata, task):
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(task=task),
        extra={MINIMAX_H3_CANONICAL_REQUEST_EXTRA_KEY: _request(task)},
    )
    return stage.forward(batch, SimpleNamespace()) is batch


def test_one_image_release_serves_t2i_and_edit():
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(_image_index())
    assert metadata.is_image and metadata.tasks == ("t2i", "edit")
    assert metadata.sigma_shift_scales is None
    assert _admit(metadata, "t2i") and _admit(metadata, "edit")
    for wrong in ("t2va", "fl2va", "ref2va"):
        with pytest.raises(ValueError, match="not served"):
            metadata.canonical_task(wrong)


@pytest.mark.parametrize("task", ["t2i", "edit"])
def test_single_task_image_release_rejects_the_other_task(task):
    other = "edit" if task == "t2i" else "t2i"
    metadata = MiniMaxH3ReleaseMetadata.from_model_index(_image_index([task]))
    assert _admit(metadata, task)
    with pytest.raises(ValueError, match=f"task '{other}' is not served"):
        _admit(metadata, other)
    stage = MiniMaxH3PartitionAdmissionStage(metadata)
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(task=task),
        extra={
            MINIMAX_H3_RESOLVED_PLAN_EXTRA_KEY: minimax_h3_resolve_plan(_request(other))
        },
    )
    with pytest.raises(ValueError, match="plan task mismatch"):
        stage.forward(batch, SimpleNamespace())


@pytest.mark.parametrize(
    "block",
    [
        {"tasks": ["edit", "t2i"]},
        {"tasks": ["t2i", "t2va"]},
        {"tasks": ["t2i"], "task_aliases": {"text": "t2i"}},
        {"partition": "t2i", "tasks": ["t2i"]},
        {"partition": "edit", "tasks": ["edit"]},
    ],
)
def test_image_release_block_is_strict(block):
    index = _image_index()
    index["_minimax_h3"].update(block)
    with pytest.raises(ValueError, match="partition"):
        MiniMaxH3ReleaseMetadata.from_model_index(index)


@pytest.mark.parametrize("vae_keys", [("video_vae",), ("vae", "video_vae"), ()])
def test_image_release_declares_its_vae_as_vae(vae_keys):
    index = _image_index()
    spec = index.pop("vae")
    index.update({key: spec for key in vae_keys})
    with pytest.raises(ValueError, match="component `vae`"):
        MiniMaxH3ReleaseMetadata.from_model_index(index)


def test_video_partitions_never_serve_image_tasks():
    assert partition_for_task(" T2I ") == partition_for_task("edit") == "image"
    for partition in ("fl2va", "ref2va"):
        metadata = MiniMaxH3ReleaseMetadata.from_model_index(_video_index(partition))
        hybrid = replace(
            metadata, partition="hybrid", tasks=tuple(MINIMAX_H3_TASK_PARTITIONS)
        )
        for release, task in ((metadata, "t2i"), (hybrid, "edit")):
            with pytest.raises(ValueError, match="not served"):
                release.canonical_task(task)


def test_video_pipeline_rejects_image_release():
    from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
        MiniMaxH3Pipeline,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
        ComposedPipelineBase,
    )

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.server_args = SimpleNamespace(model_variant=None)
    with patch.object(
        ComposedPipelineBase, "_load_config", return_value=_image_index()
    ):
        with pytest.raises(ValueError, match="cannot serve the 'image'"):
            pipeline._load_config()
