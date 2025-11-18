"""
Configuration and data structures for diffusion performance tests.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class ToleranceConfig:
    """Tolerance ratios for performance validation."""

    e2e: float
    stage: float
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

    @classmethod
    def load(cls, path: Path) -> BaselineConfig:
        """Load baseline configuration from JSON file."""
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        tol_data = data["tolerances"]
        tolerances = ToleranceConfig(
            e2e=float(os.getenv("SGLANG_E2E_TOLERANCE", tol_data["e2e"])),
            stage=float(os.getenv("SGLANG_STAGE_TIME_TOLERANCE", tol_data["stage"])),
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
        )


@dataclass(frozen=True)
class DiffusionCase:
    """Configuration for a single model/scenario test case."""

    id: str  # pytest test id
    model_path: str  # HF repo or local path
    scenario_name: str  # key into BASELINE_CONFIG.scenarios
    modality: str = "image"  # "image" or "video" or "3d"
    prompt: str | None = None  # text prompt for generation
    output_size: str = "1024x1024"  # output image dimensions (or video resolution)
    num_frames: int | None = None  # for video: number of frames
    fps: int | None = None  # for video: frames per second
    warmup_text: int = 1  # number of text-to-image/video warmups
    warmup_edit: int = 0  # number of image/video-edit warmups
    image_edit_prompt: str | None = None  # prompt for editing
    image_edit_path: Path | str | None = (
        None  # input image/video for editing (Path or URL)
    )
    startup_grace_seconds: float = 0.0  # wait time after server starts
    custom_validator: str | None = None  # optional custom validator name
    seconds: int = 4  # for video: duration in seconds

    def is_image_url(self) -> bool:
        """Check if image_edit_path is a URL."""
        if self.image_edit_path is None:
            return False
        return isinstance(self.image_edit_path, str) and (
            self.image_edit_path.startswith("http://")
            or self.image_edit_path.startswith("https://")
        )


@dataclass
class PerformanceSummary:
    """Summary of performance metrics."""

    e2e_ms: float
    avg_denoise_ms: float
    median_denoise_ms: float
    stage_metrics: dict[str, float]
    sampled_steps: dict[int, float]
    frames_per_second: float | None = None
    total_frames: int | None = None
    avg_frame_time_ms: float | None = None


# Common paths
IMAGE_INPUT_FILE = Path(__file__).resolve().parents[1] / "test_files" / "girl.jpg"

# All test cases with clean default values
# To test different models, simply add more DiffusionCase entries
DIFFUSION_CASES: list[DiffusionCase] = [
    # === Text to Image (T2I) ===
    DiffusionCase(
        id="qwen_image_t2i",
        model_path="Qwen/Qwen-Image",
        scenario_name="text_to_image",
        modality="image",
        prompt="A futuristic cityscape at sunset with flying cars",
        output_size="1024x1024",
        warmup_text=1,
        warmup_edit=0,
        startup_grace_seconds=30.0,
    ),
    DiffusionCase(
        id="flux_image_t2i",
        model_path="black-forest-labs/FLUX.1-dev",
        scenario_name="text_to_image",
        modality="image",
        prompt="A futuristic cityscape at sunset with flying cars",
        output_size="1024x1024",
        warmup_text=1,
        warmup_edit=0,
        startup_grace_seconds=30.0,
    ),
    # === Text and Image to Image (TI2I) ===
    DiffusionCase(
        id="qwen_image_edit_ti2i",
        model_path="Qwen/Qwen-Image-Edit",
        scenario_name="image_edit",
        modality="image",
        prompt=None,  # not used for editing
        output_size="1024x1536",
        warmup_text=0,
        warmup_edit=1,
        image_edit_prompt="Convert 2D style to 3D style",
        image_edit_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        startup_grace_seconds=30.0,
    ),
    # === Text to Video (T2V) ===
    DiffusionCase(
        id="fastwan2_1_t2v",
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        scenario_name="text_to_video",
        modality="video",
        prompt="A curious raccoon",
        output_size="848x480",
        seconds=4,
        warmup_text=0,  # warmups only for image gen models
        warmup_edit=0,
        startup_grace_seconds=30.0,
        custom_validator="video",
    ),
    # === Image to Video (I2V) ===
    DiffusionCase(
        id="wan2_2_i2v",
        model_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        scenario_name="image_to_video",
        modality="video",
        prompt="generate",  # passing in something since failing if no prompt is passed
        warmup_text=0,  # warmups only for image gen models
        warmup_edit=0,
        output_size="832x1104",
        image_edit_prompt="generate",
        image_edit_path="https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true",
        startup_grace_seconds=30.0,
        custom_validator="video",
        seconds=1,
    ),
    # === Text and Image to Video (TI2V) ===
    DiffusionCase(
        id="wan2_2_ti2v_5b",
        model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        scenario_name="text_image_to_video",
        modality="video",
        prompt="Animate this image",
        output_size="832x1104",
        warmup_text=0,  # warmups only for image gen models
        warmup_edit=0,
        image_edit_prompt="Add dynamic motion to the scene",
        image_edit_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
        startup_grace_seconds=30.0,
        custom_validator="video",
        seconds=4,
    ),
]


# Load global configuration
BASELINE_CONFIG = BaselineConfig.load(Path(__file__).with_name("perf_baselines.json"))
