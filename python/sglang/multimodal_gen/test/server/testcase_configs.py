"""
Configuration and data structures for diffusion performance tests.

Usage:

pytest python/sglang/multimodal_gen/test/server/test_server_a.py
# for a single testcase, look for the name of the testcases in DIFFUSION_CASES
pytest python/sglang/multimodal_gen/test/server/test_server_a.py -k qwen_image_t2i


To add a new testcase:
1. add your testcase with case-id: `my_new_test_case_id` to DIFFUSION_CASES
2. run `SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/test_server_a.py -k my_new_test_case_id`
3. insert or override the corresponding scenario in `scenarios` section of perf_baselines.json with the output baseline of step-2


"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sglang.multimodal_gen.configs.sample.base import SamplingParams
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord


@dataclass
class ToleranceConfig:
    """Tolerance ratios for performance validation."""

    e2e: float
    denoise_stage: float
    non_denoise_stage: float
    denoise_step: float
    denoise_agg: float


@dataclass
class ScenarioConfig:
    """Expected performance metrics for a test scenario."""

    stages_ms: dict[str, float]
    denoise_step_ms: dict[int, float]
    expected_e2e_ms: float
    expected_avg_denoise_ms: float
    expected_median_denoise_ms: float


@dataclass
class BaselineConfig:
    """Full baseline configuration."""

    scenarios: dict[str, ScenarioConfig]
    step_fractions: Sequence[float]
    warmup_defaults: dict[str, int]
    tolerances: ToleranceConfig
    improvement_threshold: float

    @classmethod
    def load(cls, path: Path) -> BaselineConfig:
        """Load baseline configuration from JSON file."""
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        tol_data = data["tolerances"]
        tolerances = ToleranceConfig(
            e2e=float(os.getenv("SGLANG_E2E_TOLERANCE", tol_data["e2e"])),
            denoise_stage=float(
                os.getenv("SGLANG_STAGE_TIME_TOLERANCE", tol_data["denoise_stage"])
            ),
            non_denoise_stage=float(
                os.getenv(
                    "SGLANG_NON_DENOISE_STAGE_TIME_TOLERANCE",
                    tol_data["non_denoise_stage"],
                )
            ),
            denoise_step=float(
                os.getenv("SGLANG_DENOISE_STEP_TOLERANCE", tol_data["denoise_step"])
            ),
            denoise_agg=float(
                os.getenv("SGLANG_DENOISE_AGG_TOLERANCE", tol_data["denoise_agg"])
            ),
        )

        scenarios = {}
        for name, cfg in data["scenarios"].items():
            scenarios[name] = ScenarioConfig(
                stages_ms=cfg["stages_ms"],
                denoise_step_ms={int(k): v for k, v in cfg["denoise_step_ms"].items()},
                expected_e2e_ms=float(cfg["expected_e2e_ms"]),
                expected_avg_denoise_ms=float(cfg["expected_avg_denoise_ms"]),
                expected_median_denoise_ms=float(cfg["expected_median_denoise_ms"]),
            )

        return cls(
            scenarios=scenarios,
            step_fractions=tuple(data["sampling"]["step_fractions"]),
            warmup_defaults=data["sampling"].get("warmup_requests", {}),
            tolerances=tolerances,
            improvement_threshold=data.get("improvement_reporting", {}).get(
                "threshold", 0.2
            ),
        )


@dataclass(frozen=True)
class DiffusionTestCase:
    """Configuration for a single model/scenario test case."""

    id: str  # pytest test id and scenario name
    server_args: ServerArgs
    sampling_params: SamplingParams
    warmup_text: int = 1  # number of text-to-image/video warmups
    warmup_edit: int = 0  # number of image/video-edit warmups
    custom_validator: str | None = None  # optional custom validator name
    # Optional edit prompt, as SamplingParams only has prompt
    edit_prompt: str | None = None


def sample_step_indices(
    step_map: dict[int, float], fractions: Sequence[float]
) -> list[int]:
    if not step_map:
        return []
    max_idx = max(step_map.keys())
    indices = set()
    for fraction in fractions:
        idx = min(max_idx, max(0, int(round(fraction * max_idx))))
        if idx in step_map:
            indices.add(idx)
    return sorted(indices)


@dataclass
class PerformanceSummary:
    """Summary of performance of a request, built from RequestPerfRecord"""

    e2e_ms: float
    avg_denoise_ms: float
    median_denoise_ms: float
    # { "stage_1": time_1, "stage_2": time_2 }
    stage_metrics: dict[str, float]
    step_metrics: list[float]
    sampled_steps: dict[int, float]
    all_denoise_steps: dict[int, float]
    frames_per_second: float | None = None
    total_frames: int | None = None
    avg_frame_time_ms: float | None = None

    @staticmethod
    def from_req_perf_record(
        record: RequestPerfRecord, step_fractions: Sequence[float]
    ):
        """Collect all performance metrics into a summary without validation."""
        e2e_ms = record.total_duration_ms

        step_durations = record.steps
        avg_denoise = 0.0
        median_denoise = 0.0
        if step_durations:
            avg_denoise = sum(step_durations) / len(step_durations)
            median_denoise = statistics.median(step_durations)

        per_step = {index: s for index, s in enumerate(step_durations)}
        sample_indices = sample_step_indices(per_step, step_fractions)
        sampled_steps = {idx: per_step[idx] for idx in sample_indices}

        # convert from list to dict
        stage_metrics = {}
        for item in record.stages:
            if isinstance(item, dict) and "name" in item:
                val = item.get("execution_time_ms", 0.0)
                stage_metrics[item["name"]] = val

        return PerformanceSummary(
            e2e_ms=e2e_ms,
            avg_denoise_ms=avg_denoise,
            median_denoise_ms=median_denoise,
            stage_metrics=stage_metrics,
            step_metrics=step_durations,
            sampled_steps=sampled_steps,
            all_denoise_steps=per_step,
        )


# All test cases with clean default values
# To test different models, simply add more DiffusionCase entries
ONE_GPU_CASES_A: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "qwen_image_t2i",
        ServerArgs.from_kwargs(model_path="Qwen/Qwen-Image", num_gpus=1),
        SamplingParams(
            prompt="A futuristic cityscape at sunset with flying cars",
            width=1024,
            height=1024,
            num_frames=1,
        ),
        warmup_text=1,
    ),
    DiffusionTestCase(
        "flux_image_t2i",
        ServerArgs.from_kwargs(model_path="black-forest-labs/FLUX.1-dev", num_gpus=1),
        SamplingParams(
            prompt="A futuristic cityscape at sunset with flying cars",
            width=1024,
            height=1024,
            num_frames=1,
        ),
        warmup_text=1,
    ),
    # === Text and Image to Image (TI2I) ===
    DiffusionTestCase(
        "qwen_image_edit_ti2i",
        ServerArgs.from_kwargs(model_path="Qwen/Qwen-Image-Edit", num_gpus=1),
        SamplingParams(
            prompt=None,
            width=1024,
            height=1536,
            image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
            num_frames=1,
        ),
        edit_prompt="Convert 2D style to 3D style",
        warmup_text=0,
        warmup_edit=1,
    ),
]

ONE_GPU_CASES_B: list[DiffusionTestCase] = [
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_1_t2v_1.3b",
        ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers", num_gpus=1),
        SamplingParams(
            prompt="A curious raccoon",
            width=848,
            height=480,
            num_frames=24,  # 1s * 24fps
        ),
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    # NOTE(mick): flaky
    # DiffusionTestCase(
    #     id="hunyuan_video",
    #     model_path="hunyuanvideo-community/HunyuanVideo",
    #     modality="video",
    #     prompt="A curious raccoon",
    #     output_size="720x480",
    #     warmup_text=0,
    #     warmup_edit=0,
    #     custom_validator="video",
    # ),
    DiffusionTestCase(
        "fast_hunyuan_video",
        ServerArgs.from_kwargs(model_path="FastVideo/FastHunyuan-diffusers", num_gpus=1),
        SamplingParams(
            prompt="A curious raccoon",
            width=720,
            height=480,
            num_frames=24,
        ),
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    # === Text and Image to Video (TI2V) ===
    DiffusionTestCase(
        "wan2_2_ti2v_5b",
        ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers", num_gpus=1),
        SamplingParams(
            prompt="Animate this image",
            width=832,
            height=1104,
            image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
            num_frames=24,
        ),
        edit_prompt="Add dynamic motion to the scene",
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    DiffusionTestCase(
        "fastwan2_2_ti2v_5b",
        ServerArgs(
            model_path="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers", num_gpus=1
        ),
        SamplingParams(
            prompt="Animate this image",
            width=832,
            height=1104,
            image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
            num_frames=24,
        ),
        edit_prompt="Add dynamic motion to the scene",
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
]

TWO_GPU_CASES_A = [
    DiffusionTestCase(
        "wan2_2_i2v_a14b_2gpu",
        ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers", num_gpus=2),
        SamplingParams(
            prompt="generate",
            width=832,
            height=1104,
            image_path="https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true",
            num_frames=1,  # Explicitly 1 frame? Or maybe it was generating video? Original config had num_frames=1
        ),
        edit_prompt="generate",
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    DiffusionTestCase(
        "wan2_2_t2v_a14b_2gpu",
        ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers", num_gpus=2),
        SamplingParams(
            prompt="A curious raccoon",
            width=720,
            height=480,
            num_frames=24,
        ),
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    DiffusionTestCase(
        "wan2_1_t2v_14b_2gpu",
        ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers", num_gpus=2),
        SamplingParams(
            prompt="A curious raccoon",
            width=720,
            height=480,
            num_frames=24,
        ),
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
]

TWO_GPU_CASES_B = [
    DiffusionTestCase(
        "wan2_1_i2v_14b_480P_2gpu",
        ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers", num_gpus=2),
        SamplingParams(
            prompt="Animate this image",
            width=832,
            height=1104,
            image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
            num_frames=24,
        ),
        edit_prompt="Add dynamic motion to the scene",
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    DiffusionTestCase(
        "wan2_1_i2v_14b_720P_2gpu",
        ServerArgs.from_kwargs(model_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", num_gpus=2),
        SamplingParams(
            prompt="Animate this image",
            width=832,
            height=1104,
            image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
            num_frames=24,
        ),
        edit_prompt="Add dynamic motion to the scene",
        warmup_text=0,
        warmup_edit=0,
        custom_validator="video",
    ),
    DiffusionTestCase(
        "qwen_image_t2i_2_gpus",
        ServerArgs.from_kwargs(model_path="Qwen/Qwen-Image", num_gpus=2),
        SamplingParams(
            prompt="A futuristic cityscape at sunset with flying cars",
            width=1024,
            height=1024,
            num_frames=1,
        ),
        warmup_text=1,
    ),
    DiffusionTestCase(
        "flux_image_t2i_2_gpus",
        ServerArgs.from_kwargs(model_path="black-forest-labs/FLUX.1-dev", num_gpus=2),
        SamplingParams(
            prompt="A futuristic cityscape at sunset with flying cars",
            width=1024,
            height=1024,
            num_frames=1,
        ),
        warmup_text=1,
    ),
]

# Load global configuration
BASELINE_CONFIG = BaselineConfig.load(Path(__file__).with_name("perf_baselines.json"))
