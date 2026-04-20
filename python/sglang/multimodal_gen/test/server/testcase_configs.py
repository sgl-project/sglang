"""
Configuration and data structures for diffusion performance tests.

Usage:

pytest python/sglang/multimodal_gen/test/server/test_server_1_gpu.py
# for a single testcase, look for the name of the testcase in ONE_GPU_CASES,
# ONE_GPU_CASES_C, or TWO_GPU_CASES
pytest python/sglang/multimodal_gen/test/server/test_server_1_gpu.py -k qwen_image_t2i


To add a new testcase:
1. add your testcase with case-id: `my_new_test_case_id` to the appropriate `*_CASES_*` list
2. run `SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/ -k my_new_test_case_id`
3. insert or override the corresponding scenario in `scenarios` section of perf_baselines.json with the output baseline of step-2


"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.registry import get_model_info
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord


@dataclass
class ToleranceConfig:
    """Tolerance ratios for performance validation."""

    e2e: float
    denoise_stage: float
    non_denoise_stage: float
    denoise_step: float
    denoise_agg: float

    @classmethod
    def load_profile(cls, all_tolerances: dict, profile_name: str) -> ToleranceConfig:
        """Load a specific tolerance profile from a dictionary of profiles."""
        # Support both flat structure (backward compatibility) and profiled structure
        if "e2e" in all_tolerances and not isinstance(all_tolerances["e2e"], dict):
            tol_data = all_tolerances
            actual_profile = "legacy/flat"
        else:
            tol_data = all_tolerances.get(
                profile_name, all_tolerances.get("pr_test", {})
            )
            actual_profile = (
                profile_name if profile_name in all_tolerances else "pr_test"
            )

        if not tol_data:
            raise ValueError(
                f"No tolerance profile found for '{profile_name}' and no default 'pr_test' profile exists."
            )

        print(f"--- Performance Tolerance Profile: {actual_profile} ---")

        return cls(
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


@dataclass
class ScenarioConfig:
    """Expected performance metrics for a test scenario."""

    stages_ms: dict[str, float]
    denoise_step_ms: dict[int, float]
    expected_e2e_ms: float
    expected_avg_denoise_ms: float
    expected_median_denoise_ms: float
    estimated_full_test_time_s: float | None = None


@dataclass
class BaselineConfig:
    """Full baseline configuration."""

    scenarios: dict[str, ScenarioConfig]
    step_fractions: Sequence[float]
    tolerances: ToleranceConfig
    improvement_threshold: float

    @classmethod
    def load(cls, path: Path) -> BaselineConfig:
        """Load baseline configuration from JSON file."""
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Get tolerance profile, defaulting to 'pr_test'
        profile_name = "pr_test"
        tolerances = ToleranceConfig.load_profile(
            data.get("tolerances", {}), profile_name
        )

        scenarios = {}
        for name, cfg in data["scenarios"].items():
            scenarios[name] = ScenarioConfig(
                stages_ms=cfg["stages_ms"],
                denoise_step_ms={int(k): v for k, v in cfg["denoise_step_ms"].items()},
                expected_e2e_ms=float(cfg["expected_e2e_ms"]),
                expected_avg_denoise_ms=float(cfg["expected_avg_denoise_ms"]),
                expected_median_denoise_ms=float(cfg["expected_median_denoise_ms"]),
                estimated_full_test_time_s=cfg.get("estimated_full_test_time_s"),
            )

        return cls(
            scenarios=scenarios,
            step_fractions=tuple(data["sampling"]["step_fractions"]),
            tolerances=tolerances,
            improvement_threshold=data.get("improvement_reporting", {}).get(
                "threshold", 0.2
            ),
        )

    def update(self, path: Path):
        """Load baseline configuration from JSON file."""
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        scenarios_new = {}
        for name, cfg in data["scenarios"].items():
            scenarios_new[name] = ScenarioConfig(
                stages_ms=cfg["stages_ms"],
                denoise_step_ms={int(k): v for k, v in cfg["denoise_step_ms"].items()},
                expected_e2e_ms=float(cfg["expected_e2e_ms"]),
                expected_avg_denoise_ms=float(cfg["expected_avg_denoise_ms"]),
                expected_median_denoise_ms=float(cfg["expected_median_denoise_ms"]),
                estimated_full_test_time_s=cfg.get("estimated_full_test_time_s"),
            )

        self.scenarios.update(scenarios_new)
        return self


@dataclass
class DiffusionServerArgs:
    """Configuration for a single model/scenario test case."""

    model_path: str  # HF repo or local path
    modality: str | None = None  # auto-inferred: "image" or "video" or "3d"

    custom_validator: str | None = None  # auto-derived unless explicitly overridden
    # resources
    num_gpus: int = 1
    tp_size: int | None = None
    ulysses_degree: int | None = None
    ring_degree: int | None = None
    cfg_parallel: bool | None = None
    # LoRA
    lora_path: str | None = (
        None  # LoRA adapter path (HF repo or local path, loaded at startup)
    )
    dynamic_lora_path: str | None = (
        None  # LoRA path for dynamic loading test (loaded via set_lora after startup)
    )
    second_lora_path: str | None = (
        None  # Second LoRA adapter path for multi-LoRA testing
    )

    dit_layerwise_offload: bool = False
    dit_offload_prefetch_size: int | float | None = None
    enable_cache_dit: bool = False
    text_encoder_cpu_offload: bool = False
    enable_warmup: bool = True

    extras: list[str] = field(default_factory=lambda: [])
    env_vars: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        if self.modality is None:
            self.modality = _infer_modality_from_model_path(self.model_path)

        if self.custom_validator is not None:
            return

        if self.modality == "image":
            self.custom_validator = "image"
        elif self.modality == "video":
            self.custom_validator = "video"
        elif self.modality == "3d":
            self.custom_validator = "mesh"


@lru_cache(maxsize=None)
def _infer_modality_from_model_path(model_path: str) -> str:
    model_info = get_model_info(model_path)
    if model_info is None:
        raise ValueError(f"Could not resolve model info for {model_path!r}")

    task_type = model_info.pipeline_config_cls.task_type
    if task_type == ModelTaskType.I2M:
        return "3d"
    if task_type.is_image_gen():
        return "image"
    return "video"


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

    # URL direct test flag - if True, don't pre-download URL images
    direct_url_test: bool = False

    # output format
    output_format: str | None = None  # "png", "jpeg", "mp4", etc.

    num_outputs_per_prompt: int = 1

    # Additional request-level parameters (e.g. enable_teacache, enable_upscaling, …)
    # merged directly into the OpenAI extra_body dict.
    extras: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DiffusionTestCase:
    """Configuration for a single model/scenario test case."""

    id: str  # pytest test id and scenario name
    server_args: DiffusionServerArgs
    sampling_params: DiffusionSamplingParams
    run_perf_check: bool = True
    run_consistency_check: bool = True
    run_models_api_check: bool = True
    run_t2v_input_reference_check: bool = True
    run_lora_basic_api_check: bool = False
    run_lora_dynamic_load_check: bool = False
    run_lora_dynamic_switch_check: bool = False
    run_multi_lora_api_check: bool = False

    def __post_init__(self) -> None:
        has_startup_lora = self.server_args.lora_path is not None
        has_dynamic_lora = self.server_args.dynamic_lora_path is not None
        has_second_lora = self.server_args.second_lora_path is not None

        if self.run_lora_basic_api_check and not (has_startup_lora or has_dynamic_lora):
            raise ValueError(
                f"{self.id}: run_lora_basic_api_check requires lora_path or dynamic_lora_path"
            )

        if self.run_lora_dynamic_load_check and not has_dynamic_lora:
            raise ValueError(
                f"{self.id}: run_lora_dynamic_load_check requires dynamic_lora_path"
            )

        if self.run_lora_dynamic_switch_check and not has_second_lora:
            raise ValueError(
                f"{self.id}: run_lora_dynamic_switch_check requires second_lora_path"
            )

        if self.run_multi_lora_api_check and not (has_startup_lora and has_second_lora):
            raise ValueError(
                f"{self.id}: run_multi_lora_api_check requires lora_path and second_lora_path"
            )


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

MODELOPT_T2I_CI_sampling_params = DiffusionSamplingParams(
    prompt="Doraemon is eating dorayaki",
    output_size="768x768",
    extras={"num_inference_steps": 12, "seed": 0},
)

TI2I_sampling_params = DiffusionSamplingParams(
    prompt="Convert 2D style to 3D style",
    image_path="https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg",
)

MULTI_IMAGE_TI2I_sampling_params = DiffusionSamplingParams(
    prompt="The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square.",
    image_path=[
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg",
    ],
    direct_url_test=True,
)
MULTI_IMAGE_TI2I_UPLOAD_sampling_params = DiffusionSamplingParams(
    prompt="The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square.",
    image_path=[
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg",
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg",
    ],
)
MULTI_FRAME_I2I_sampling_params = DiffusionSamplingParams(
    prompt="a high quality, cute halloween themed illustration, consistent style and lighting",
    image_path=[
        "https://raw.githubusercontent.com/QwenLM/Qwen-Image-Layered/main/assets/test_images/4.png"
    ],
    num_frames=4,
    direct_url_test=True,
    output_format="png",
)

T2V_PROMPT = "A curious raccoon"

T2V_sampling_params = DiffusionSamplingParams(
    prompt=T2V_PROMPT,
)

MODELOPT_T2V_CI_sampling_params = DiffusionSamplingParams(
    prompt=T2V_PROMPT,
    output_size="640x384",
    seconds=5,
    num_frames=17,
    extras={"num_inference_steps": 12, "seed": 0},
)

TI2V_sampling_params = DiffusionSamplingParams(
    prompt="The man in the picture slowly turns his head, his expression enigmatic and otherworldly. The camera performs a slow, cinematic dolly out, focusing on his face. Moody lighting, neon signs glowing in the background, shallow depth of field.",
    image_path="https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/5f/fa/56/5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/825646124206.jpg/600x600bb.jpg",
    direct_url_test=True,
)

TURBOWAN_I2V_sampling_params = DiffusionSamplingParams(
    prompt="The man in the picture slowly turns his head, his expression enigmatic and otherworldly. The camera performs a slow, cinematic dolly out, focusing on his face. Moody lighting, neon signs glowing in the background, shallow depth of field.",
    image_path="https://is1-ssl.mzstatic.com/image/thumb/Music114/v4/5f/fa/56/5ffa56c2-ea1f-7a17-6bad-192ff9b6476d/825646124206.jpg/600x600bb.jpg",
    direct_url_test=True,
    output_size="960x960",
    num_frames=4,
    fps=4,
)

HUNYUAN3D_SHAPE_sampling_params = DiffusionSamplingParams(
    prompt="",
    image_path="https://raw.githubusercontent.com/sgl-project/sgl-test-files/main/diffusion-ci/consistency_gt/1-gpu/hunyuan3d_2_0/hunyuan3d.png",
)

MODELOPT_FLUX1_FP8_TRANSFORMER = "BBuf/flux1-dev-modelopt-fp8-sglang-transformer"
MODELOPT_FLUX2_FP8_TRANSFORMER = "BBuf/flux2-dev-modelopt-fp8-sglang-transformer"
MODELOPT_WAN22_FP8_TRANSFORMER = "BBuf/wan22-t2v-a14b-modelopt-fp8-sglang-transformer"
MODELOPT_FLUX1_NVFP4_TRANSFORMER = "BBuf/flux1-dev-modelopt-nvfp4-sglang-transformer"
MODELOPT_FLUX2_NVFP4_WEIGHTS = "black-forest-labs/FLUX.2-dev-NVFP4"
MODELOPT_WAN22_NVFP4_TRANSFORMER = (
    "BBuf/wan22-t2v-a14b-modelopt-nvfp4-sglang-transformer"
)
MODELOPT_NVFP4_B200_ENV_VARS = {"SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND": "cudnn"}


def _make_modelopt_ci_case(
    case_id: str,
    *,
    model_path: str,
    modality: str,
    sampling_params: DiffusionSamplingParams,
    extras: list[str],
    env_vars: dict[str, str] | None = None,
) -> DiffusionTestCase:
    return DiffusionTestCase(
        case_id,
        DiffusionServerArgs(
            model_path=model_path,
            modality=modality,
            enable_warmup=False,
            extras=extras,
            env_vars=env_vars or {},
        ),
        sampling_params,
        run_perf_check=False,
        run_consistency_check=False,
    )


def _with_default_num_gpus(
    cases: list[DiffusionTestCase], num_gpus: int
) -> list[DiffusionTestCase]:
    return [
        replace(case, server_args=replace(case.server_args, num_gpus=num_gpus))
        for case in cases
    ]


# Load global configuration
BASELINE_CONFIG = BaselineConfig.load(
    Path(__file__).with_name("perf_baselines.json")
).update(Path(__file__).parent / "ascend" / "perf_baselines_npu.json")
