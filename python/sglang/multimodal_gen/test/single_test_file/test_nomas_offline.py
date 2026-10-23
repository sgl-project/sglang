# SPDX-License-Identifier: Apache-2.0
"""Real-weight, single-GPU regression through the public offline API.

Generate goldens with SGLANG_GEN_GT=1 and SGLANG_GT_OUTPUT_DIR=/path/to/gt.
Unset SGLANG_GEN_GT to compare against the pinned ci-data-diffusion goldens.
SGLANG_CONSISTENCY_GT_DIR can select a local golden directory for development.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.test.test_utils import (
    _consistency_gt_filenames,
    compare_with_gt,
    extract_key_frames_from_video,
    get_consistency_thresholds,
    get_video_frame_count,
    load_consistency_gt,
)

CASE_ID = "nomas_offline_two_events"
MODEL_ID = "tencent/Hy-Nomas"
FIRST_EVENT = json.dumps(
    {
        "Style": "Cinematic fantasy",
        "perspective_view": "Third-person view",
        "subject_list": [
            {
                "subject_id": "fox",
                "display_name": "a small orange fox",
                "subject_attribute": "Orange fur, white chest, and a fluffy tail.",
            }
        ],
        "segments": [
            {
                "visual_description_long": "A small orange fox stands on a stone path"
                " in a luminous forest. Blue fireflies drift between the trees.",
                "contained_actions": [
                    {
                        "start_time": "00:00.000",
                        "end_time": "00:05.000",
                        "action_description": "The fox walks slowly along the path.",
                    }
                ],
            }
        ],
    }
)
SECOND_EVENT = json.dumps(
    {
        "segments": [
            {
                "description_change": "Glowing blue flowers open beside the path"
                " as the fox continues forward.",
                "contained_actions": [
                    {
                        "start_time": "00:00.000",
                        "end_time": "00:05.000",
                        "action_description": "The fox passes the flowers and looks around.",
                    }
                ],
            }
        ]
    }
)
# Two 120-frame events at 352x640; changing any field invalidates the goldens.
REQUEST = {
    "prompt": FIRST_EVENT,
    "prompt_format": "json",
    "max_sequence_length": 1280,
    "height": 352,
    "width": 640,
    "num_frames": 240,
    "fps": 24,
    "num_inference_steps": 8,
    "guidance_scale": 1.0,
    "seed": 42,
    "segments": [
        {"frames": 120, "caption": FIRST_EVENT},
        {"frames": 120, "caption": SECOND_EVENT},
    ],
}


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="Nomas pretrained regression requires NVIDIA CUDA",
)
def test_nomas_offline_two_events(tmp_path):
    with DiffGenerator.from_pretrained(
        model_path=MODEL_ID,
        num_gpus=1,
        warmup_mode="off",
        text_encoder_cpu_offload=True,
    ) as generator:
        result = generator.generate(
            {
                **REQUEST,
                "save_output": True,
                "output_path": str(tmp_path),
                "output_file_name": CASE_ID + ".mp4",
            }
        )

    assert result is not None, "Nomas offline generation failed; inspect worker logs"
    video_path = Path(result.output_file_path)
    assert get_video_frame_count(str(video_path)) == 240
    frames = extract_key_frames_from_video(video_path.read_bytes())
    assert len(frames) == 3
    assert all(frame.shape == (352, 640, 3) for frame in frames)

    if os.environ.get("SGLANG_GEN_GT") == "1":
        out_dir = Path(os.environ["SGLANG_GT_OUTPUT_DIR"])
        out_dir.mkdir(parents=True, exist_ok=True)
        names = _consistency_gt_filenames(CASE_ID, 1, is_video=True)
        for frame, name in zip(frames, names):
            Image.fromarray(frame).save(out_dir / name)
        return

    comparison = compare_with_gt(
        output_frames=frames,
        gt_data=load_consistency_gt(CASE_ID, 1, is_video=True),
        thresholds=get_consistency_thresholds(CASE_ID, is_video=True),
        case_id=CASE_ID,
    )
    assert comparison.passed, comparison
