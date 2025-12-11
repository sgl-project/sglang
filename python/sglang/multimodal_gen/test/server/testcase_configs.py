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
class DiffusionServerArgs:
    """Configuration for a single model/scenario test case."""

    model_path: str  # HF repo or local path
    modality: str = "image"  # "image" or "video" or "3d"

    warmup_text: int = 1  # number of text-to-image/video warmups
    warmup_edit: int = 0  # number of image/video-edit warmups
    custom_validator: str | None = None  # optional custom validator name
    # resources
    num_gpus: int = 1
    # LoRA
    lora_path: str | None = None  # LoRA adapter path (HF repo or local path)


@dataclass(frozen=True)
class DiffusionSamplingParams:
    """Configuration for a single model/scenario test case."""

    output_size: str = ""

    # inputs and conditioning
    prompt: str | None = None  # text prompt for generation
    image_path: Path | str | None = None  # input image/video for editing (Path or URL)

    # duration
    seconds: int = 1  # for video: duration in seconds
    num_frames: int | None = None  # for video: number of frames
    fps: int | None = None  # for video: frames per second


@dataclass(frozen=True)
class DiffusionTestCase:
    """Configuration for a single model/scenario test case."""

    id: str  # pytest test id and scenario name
    server_args: DiffusionServerArgs
    sampling_params: DiffusionSamplingParams


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


T2I_sampling_params = DiffusionSamplingParams(
    prompt="Doraemon is eating dorayaki",
    output_size="1024x1024",
)

TI2I_sampling_params = DiffusionSamplingParams(
    prompt="Convert 2D style to 3D style",
    image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
)

T2V_PROMPT = "A curious raccoon"

TI2V_sampling_params = DiffusionSamplingParams(
    output_size="832x1104",
    prompt="The man in the picture slowly turns his head, his expression enigmatic and otherworldly. The camera performs a slow, cinematic dolly out, focusing on his face. Moody lighting, neon signs glowing in the background, shallow depth of field.",
    image_path="https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/5f/fa/56/5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/825646124206.jpg/600x600bb.jpg",
)

# All test cases with clean default values
# To test different models, simply add more DiffusionCase entries
ONE_GPU_CASES_A: list[DiffusionTestCase] = [
    # === Text to Image (T2I) ===
    DiffusionTestCase(
        "qwen_image_t2i",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "flux_image_t2i",
        DiffusionServerArgs(
            model_path="black-forest-labs/FLUX.1-dev",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "flux_2_image_t2i",
        DiffusionServerArgs(
            model_path="black-forest-labs/FLUX.2-dev",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "zimage_image_t2i",
        DiffusionServerArgs(
            model_path="Tongyi-MAI/Z-Image-Turbo",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        T2I_sampling_params,
    ),
    # === Text and Image to Image (TI2I) ===
    DiffusionTestCase(
        "qwen_image_edit_ti2i",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image-Edit",
            modality="image",
            warmup_text=0,
            warmup_edit=1,
        ),
        TI2I_sampling_params,
    ),
]

ONE_GPU_CASES_B: list[DiffusionTestCase] = [
    # === Text to Video (T2V) ===
    DiffusionTestCase(
        "wan2_1_t2v_1.3b",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            output_size="848x480",
        ),
    ),
    # LoRA test case for single transformer + merge/unmerge API test
    DiffusionTestCase(
        "wan2_1_t2v_1_3b_lora_1gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
            num_gpus=1,
            lora_path="Cseti/Wan-LoRA-Arcane-Jinx-v1",
        ),
        DiffusionSamplingParams(
            prompt="csetiarcane Nfj1nx with blue hair, a woman walking in a cyberpunk city at night",
            output_size="480x320",
            num_frames=8,
        ),
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
        "flux_2_ti2i",
        DiffusionServerArgs(
            model_path="black-forest-labs/FLUX.2-dev",
            modality="image",
            warmup_text=0,
            warmup_edit=1,
        ),
        TI2I_sampling_params,
    ),
    DiffusionTestCase(
        "fast_hunyuan_video",
        DiffusionServerArgs(
            model_path="FastVideo/FastHunyuan-diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            output_size="720x480",
        ),
    ),
    # === Text and Image to Video (TI2V) ===
    DiffusionTestCase(
        "wan2_2_ti2v_5b",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "fastwan2_2_ti2v_5b",
        DiffusionServerArgs(
            model_path="FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
        ),
        TI2V_sampling_params,
    ),
]

TWO_GPU_CASES_A = [
    # TODO: Timeout with Torch2.9. Add back when it can pass CI
    DiffusionTestCase(
        "wan2_2_i2v_a14b_2gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "wan2_2_t2v_a14b_2gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
            num_gpus=2,
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            output_size="720x480",
        ),
    ),
    # LoRA test case for transformer_2 support
    DiffusionTestCase(
        "wan2_2_t2v_a14b_lora_2gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
            num_gpus=2,
            lora_path="Cseti/wan2.2-14B-Arcane_Jinx-lora-v1",
        ),
        DiffusionSamplingParams(
            prompt="Nfj1nx with blue hair, a woman walking in a cyberpunk city at night",
            output_size="720x480",
        ),
    ),
    DiffusionTestCase(
        "wan2_1_t2v_14b_2gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-T2V-14B-Diffusers",
            warmup_text=0,
            warmup_edit=0,
            modality="video",
            num_gpus=2,
            custom_validator="video",
        ),
        DiffusionSamplingParams(
            prompt=T2V_PROMPT,
            output_size="720x480",
        ),
    ),
]

TWO_GPU_CASES_B = [
    DiffusionTestCase(
        "wan2_1_i2v_14b_480P_2gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
            warmup_text=0,
            warmup_edit=0,
            modality="video",
            custom_validator="video",
            num_gpus=2,
        ),
        TI2V_sampling_params,
    ),
    # I2V LoRA test case
    DiffusionTestCase(
        "wan2_1_i2v_14b_lora_2gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
            num_gpus=2,
            lora_path="starsfriday/Wan2.1-Divine-Power-LoRA",
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "wan2_1_i2v_14b_720P_2gpu",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
            modality="video",
            warmup_text=0,
            warmup_edit=0,
            custom_validator="video",
            num_gpus=2,
        ),
        TI2V_sampling_params,
    ),
    DiffusionTestCase(
        "qwen_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
            num_gpus=2,
        ),
        T2I_sampling_params,
    ),
    DiffusionTestCase(
        "flux_image_t2i_2_gpus",
        DiffusionServerArgs(
            model_path="black-forest-labs/FLUX.1-dev",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        T2I_sampling_params,
    ),
]

# Load global configuration
BASELINE_CONFIG = BaselineConfig.load(Path(__file__).with_name("perf_baselines.json"))
