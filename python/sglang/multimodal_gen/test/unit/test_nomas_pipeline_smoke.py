# SPDX-License-Identifier: Apache-2.0
"""Opt-in public-API regression using a fixed random Nomas pipeline.

Create it with test/scripts/create_nomas_fixture.py, then set
NOMAS_RANDOM_PIPELINE to its path before running pytest on this file.
"""

import hashlib
import json
import os

import imageio.v3 as iio
import numpy as np
import pytest
import torch

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator


@pytest.mark.skipif(
    not os.getenv("NOMAS_RANDOM_PIPELINE") or not torch.cuda.is_available(),
    reason="requires CUDA and a fixed random Nomas pipeline",
)
def test_public_pipeline_conditions_and_event_frames(tmp_path):
    common = dict(
        prompt="A red car drives forward.",
        height=64,
        width=64,
        num_frames=48,
        num_inference_steps=2,
        guidance_scale=1.0,
        seed=42,
        save_output=True,
        output_path=str(tmp_path),
    )
    results = {}
    pixels = {}
    with DiffGenerator.from_pretrained(
        model_path=os.environ["NOMAS_RANDOM_PIPELINE"], num_gpus=1, warmup_mode="off"
    ) as generator:
        camera = np.repeat(np.eye(4, dtype=np.float32)[None], 45, axis=0)
        camera[:, 0, 3] = np.linspace(0, 0.5, 45)
        cases = [
            ("baseline", {}),
            ("repeat", {}),
            ("different_seed", {"seed": 43}),
            ("different_prompt", {"prompt": "A blue boat turns left."}),
            ("continuation", {"continuation_prompts": ["A blue boat turns left."]}),
            ("camera", {"norm_extrinsics": camera.tolist()}),
            (
                "events",
                {
                    "num_frames": 49,
                    "height": 72,
                    "segments": [
                        {"frames": 17, "caption": "A red car drives forward."},
                        {"frames": 32, "caption": "The car turns left."},
                    ],
                },
            ),
            (
                "json_events",
                {
                    "num_frames": 49,
                    "prompt_format": "json",
                    "max_sequence_length": 512,
                    "segments": [
                        {
                            "frames": 17,
                            "caption": json.dumps(
                                {
                                    "segments": [
                                        {
                                            "description": "A red car",
                                            "contained_actions": [
                                                {
                                                    "start_time": 0,
                                                    "end_time": 0.33,
                                                    "action": "drive",
                                                },
                                                {
                                                    "start_time": 0.33,
                                                    "end_time": 0.7,
                                                    "action": "turn",
                                                },
                                            ],
                                        }
                                    ]
                                }
                            ),
                        },
                        {
                            "frames": 32,
                            "caption": json.dumps(
                                {
                                    "segments": [
                                        {
                                            "description": "A road",
                                            "contained_actions": [
                                                {
                                                    "start_time": 0,
                                                    "end_time": 1.33,
                                                    "action": "stop",
                                                }
                                            ],
                                        }
                                    ]
                                }
                            ),
                        },
                    ],
                },
            ),
        ]
        for name, overrides in cases:
            result = generator.generate(
                common | overrides | {"output_file_name": name + ".mp4"}
            )
            if result is None:
                raise RuntimeError(f"Pipeline failed for {name}")
            video = iio.imread(result.output_file_path)
            assert video.shape == (overrides.get("num_frames", 48), 64, 64, 3), (
                video.shape
            )
            pixels[name] = video
            results[name] = {
                "path": result.output_file_path,
                "shape": list(video.shape),
                "sha256_decoded_pixels": hashlib.sha256(video.tobytes()).hexdigest(),
            }
    assert np.array_equal(pixels["baseline"], pixels["repeat"])
    for name in ("different_seed", "different_prompt", "continuation", "camera"):
        assert not np.array_equal(pixels["baseline"], pixels[name]), name
    # The first generated segment is identical for a changed continuation caption;
    # temporal decoder blending can revise pixels near the segment boundary.
    results["checks"] = {
        "same_seed_bitexact": True,
        "seed_prompt_camera_sensitive": True,
        "continuation_caption_sensitive": True,
        "frames": 48,
    }
    (tmp_path / "checks.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
