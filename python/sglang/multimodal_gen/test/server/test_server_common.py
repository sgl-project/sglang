"""
Config-driven diffusion generation test with pytest parametrization.


If the actual run is significantly better than the baseline, the improved cases with their updated baseline will be printed
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable

import openai
import pytest
import requests
from openai import OpenAI

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.server import conftest
from sglang.multimodal_gen.test.server.test_server_utils import (
    VALIDATOR_REGISTRY,
    PerformanceValidator,
    ServerContext,
    ServerManager,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    DiffusionTestCase,
    PerformanceSummary,
    ScenarioConfig,
)
from sglang.multimodal_gen.test.test_utils import (
    _consistency_gt_filenames,
    _get_consistency_gt_dir,
    compare_with_gt,
    extract_key_frames_from_video,
    get_consistency_gt_candidates,
    get_consistency_gt_remote_files,
    get_consistency_thresholds,
    get_dynamic_server_port,
    gt_exists,
    image_bytes_to_numpy,
    load_consistency_gt,
    wait_for_req_perf_record,
)

logger = init_logger(__name__)

# Track test cases missing estimated_full_test_time_s for time measurement output
_MISSING_ESTIMATED_TIME_CASES: set[str] = set()
_PENDING_BASELINE_DUMPS: dict[str, tuple["PerformanceSummary", bool]] = {}


@pytest.fixture
def diffusion_server(case: DiffusionTestCase) -> ServerContext:
    """Start a diffusion server for a single case and tear it down afterwards."""
    _fixture_start_time = time.perf_counter()
    server_args = case.server_args

    # Skip ring attention tests on AMD/ROCm - Ring Attention requires Flash Attention
    # which is not available on AMD. Use Ulysses parallelism instead.
    if (
        current_platform.is_hip()
        and server_args.ring_degree is not None
        and server_args.ring_degree > 1
    ):
        pytest.skip(
            f"Skipping {case.id}: Ring Attention (ring_degree={server_args.ring_degree}) "
            "requires Flash Attention which is not available on AMD/ROCm"
        )

    default_port = get_dynamic_server_port()
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", default_port))
    sampling_params = case.sampling_params
    extra_args = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")

    extra_args += f" --num-gpus {server_args.num_gpus}"

    if server_args.tp_size is not None:
        extra_args += f" --tp-size {server_args.tp_size}"

    if server_args.ulysses_degree is not None:
        extra_args += f" --ulysses-degree {server_args.ulysses_degree}"

    if server_args.dit_layerwise_offload:
        extra_args += f" --dit-layerwise-offload true"

    if server_args.dit_offload_prefetch_size:
        extra_args += (
            f" --dit-offload-prefetch-size {server_args.dit_offload_prefetch_size}"
        )

    if server_args.text_encoder_cpu_offload:
        extra_args += f" --text-encoder-cpu-offload"

    if server_args.ring_degree is not None:
        extra_args += f" --ring-degree {server_args.ring_degree}"

    if server_args.cfg_parallel:
        extra_args += " --enable-cfg-parallel"

    # LoRA support
    if server_args.lora_path:
        extra_args += f" --lora-path {server_args.lora_path}"

    if server_args.enable_warmup:
        extra_args += " --warmup"

    # Strict ports: fail immediately if port is occupied instead of silently
    # picking another one (which causes the test client to connect to the wrong server).
    extra_args += " --strict-ports"

    for arg in server_args.extras:
        extra_args += f" {arg}"

    # Build custom environment variables
    env_vars = {}
    if server_args.enable_cache_dit:
        env_vars["SGLANG_CACHE_DIT_ENABLED"] = "true"

    # start server
    wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200"))
    logger.info(
        "[server-test] Starting server for test case: %s\n"
        "  Model: %s\n"
        "  Port: %s\n"
        "  Wait deadline: %ss\n"
        "  Extra args: %s\n"
        "  Num GPUs: %s",
        case.id,
        server_args.model_path,
        port,
        wait_deadline,
        extra_args,
        server_args.num_gpus,
    )

    manager = ServerManager(
        model=server_args.model_path,
        port=port,
        wait_deadline=wait_deadline,
        extra_args=extra_args,
        env_vars=env_vars,
    )
    try:
        ctx = manager.start()
    except (RuntimeError, TimeoutError) as exc:
        # Auto-skip when the installed diffusers version lacks the required
        # pipeline class.  This avoids hard failures when a model needs a
        # newer diffusers release than what is currently installed in CI.
        msg = str(exc)
        if "not found in diffusers" in msg or (
            "has no attribute" in msg and "diffusers" in msg.lower()
        ):
            pytest.skip(
                f"Skipping {case.id}: required diffusers pipeline class "
                f"is not available in the installed version. "
                f"Upgrade diffusers to enable this test."
            )
        raise

    try:
        # Reconstruct output size for OpenAI API
        # Allow override via environment variable (useful for AMD where large resolutions can cause GPU hang)
        output_size = os.environ.get(
            "SGLANG_TEST_OUTPUT_SIZE", sampling_params.output_size
        )
    except Exception as exc:
        logger.error("Warm-up failed for %s: %s", case.id, exc)
        ctx.cleanup()
        raise

    try:
        yield ctx
    finally:
        ctx.cleanup()

        _fixture_end_time = time.perf_counter()
        _measured_full_time = _fixture_end_time - _fixture_start_time
        is_baseline_generation_mode = os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"

        pending_dump = _PENDING_BASELINE_DUMPS.pop(case.id, None)
        if pending_dump is not None:
            summary, missing_scenario = pending_dump
            DiffusionServerBase()._dump_baseline_for_testcase(
                case,
                summary,
                missing_scenario=missing_scenario,
                measured_full_time=_measured_full_time,
            )

        scenario = BASELINE_CONFIG.scenarios.get(case.id)
        needs_estimated_time = (
            scenario is None or scenario.estimated_full_test_time_s is None
        )

        if needs_estimated_time and not is_baseline_generation_mode:
            _MISSING_ESTIMATED_TIME_CASES.add(case.id)
            logger.error(
                f'\n{"=" * 60}\n'
                f'Add "estimated_full_test_time_s" to scenario "{case.id}":\n\n'
                f"File: python/sglang/multimodal_gen/test/server/perf_baselines.json\n\n"
                f'    "{case.id}": {{\n'
                f"        ...\n"
                f'        "estimated_full_test_time_s": {_measured_full_time:.1f}\n'
                f"    }}\n"
                f'{"=" * 60}\n'
            )


class DiffusionServerBase:
    """Performance tests for all diffusion models/scenarios.

    This single test class runs against all cases defined in ONE_GPU_CASES.
    Each case gets its own server instance via the parametrized fixture.
    """

    _perf_results: list[dict[str, Any]] = []
    _improved_baselines: list[dict[str, Any]] = []
    _pytest_config = None  # Store pytest config for stash access

    @classmethod
    def setup_class(cls):
        cls._perf_results = []
        cls._improved_baselines = []

    @classmethod
    def teardown_class(cls):
        print(
            f"\n[DEBUG teardown_class] Called for {cls.__name__}, _perf_results has {len(cls._perf_results)} entries"
        )
        if cls._pytest_config:
            # Add results to pytest stash (shared across all import contexts)
            for result in cls._perf_results:
                result["class_name"] = cls.__name__
            conftest.add_perf_results(cls._pytest_config, cls._perf_results)
            print(
                f"[DEBUG teardown_class] Added {len(cls._perf_results)} results to stash"
            )
        else:
            print(
                "[DEBUG teardown_class] No pytest_config available, skipping stash update"
            )

        if cls._improved_baselines:
            import json

            output = """
--- POTENTIAL BASELINE IMPROVEMENTS DETECTED ---
The following test cases performed significantly better than their baselines.
Consider updating perf_baselines.json with the snippets below:
"""
            for item in cls._improved_baselines:
                output += (
                    f'\n"{item["id"]}": {json.dumps(item["baseline"], indent=4)},\n'
                )
            print(output)

    @pytest.fixture(autouse=True)
    def _capture_pytest_config(self, request):
        """Capture pytest config for use in teardown_class."""
        self.__class__._pytest_config = request.config

    def _client(self, ctx: ServerContext) -> OpenAI:
        """Get OpenAI client for the server."""
        return OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{ctx.port}/v1",
        )

    def run_and_collect(
        self,
        ctx: ServerContext,
        case_id: str,
        generate_fn: Callable[[str, openai.Client], tuple[str, bytes]],
        collect_perf: bool = True,
    ) -> tuple[RequestPerfRecord | None, bytes]:
        """Run generation and optionally collect performance records.

        Returns:
            Tuple of (performance_record, content_bytes)
        """
        client = self._client(ctx)
        rid, content = generate_fn(case_id, client)

        if not collect_perf:
            return None, content

        log_path = ctx.perf_log_path
        log_wait_timeout = 30
        req_perf_record = wait_for_req_perf_record(
            rid,
            log_path,
            timeout=log_wait_timeout,
        )

        return (req_perf_record, content)

    def _validate_and_record(
        self,
        case: DiffusionTestCase,
        perf_record: RequestPerfRecord,
    ) -> None:
        """Validate metrics and record results."""
        is_baseline_generation_mode = os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"

        scenario = BASELINE_CONFIG.scenarios.get(case.id)
        missing_scenario = False
        if scenario is None:
            # Create dummy scenario to allow metric collection
            scenario = type(
                "DummyScenario",
                (),
                {
                    "expected_e2e_ms": 0,
                    "expected_avg_denoise_ms": 0,
                    "expected_median_denoise_ms": 0,
                    "stages_ms": {},
                    "denoise_step_ms": {},
                },
            )()
            if not is_baseline_generation_mode:
                missing_scenario = True

        # Check for missing estimated_full_test_time_s
        missing_estimated_time = False
        if (
            not missing_scenario
            and not is_baseline_generation_mode
            and scenario.estimated_full_test_time_s is None
        ):
            missing_estimated_time = True
            _MISSING_ESTIMATED_TIME_CASES.add(case.id)

        validator_name = case.server_args.custom_validator or "default"
        validator_class = VALIDATOR_REGISTRY.get(validator_name, PerformanceValidator)

        validator = validator_class(
            scenario=scenario,
            tolerances=BASELINE_CONFIG.tolerances,
            step_fractions=BASELINE_CONFIG.step_fractions,
        )

        summary = validator.collect_metrics(perf_record)

        if case.run_perf_check:
            if is_baseline_generation_mode:
                _PENDING_BASELINE_DUMPS[case.id] = (summary, missing_scenario)
                return

            if missing_scenario:
                self._dump_baseline_for_testcase(case, summary, missing_scenario)
                if missing_scenario:
                    pytest.fail(
                        f"Testcase '{case.id}' not found in perf_baselines.json"
                    )
                return

            self._check_for_improvement(case, summary, scenario)

            # only run performance validation if run_perf_check is True
            try:
                validator.validate(perf_record, case.sampling_params.num_frames)
            except AssertionError as e:
                logger.error(f"Performance validation failed for {case.id}:\n{e}")
                self._dump_baseline_for_testcase(case, summary, missing_scenario)
                raise

        result = {
            "test_name": case.id,
            "modality": case.server_args.modality,
            "e2e_ms": summary.e2e_ms,
            "avg_denoise_ms": summary.avg_denoise_ms,
            "median_denoise_ms": summary.median_denoise_ms,
            "stage_metrics": summary.stage_metrics,
            "sampled_steps": summary.sampled_steps,
        }

        # video-specific metrics
        if summary.frames_per_second:
            result.update(
                {
                    "frames_per_second": summary.frames_per_second,
                    "total_frames": summary.total_frames,
                    "avg_frame_time_ms": summary.avg_frame_time_ms,
                }
            )

        self.__class__._perf_results.append(result)
        print(
            f"[DEBUG _validate_and_record] Appended result for {case.id}, class {self.__class__.__name__} now has {len(self.__class__._perf_results)} results"
        )

    def _check_for_improvement(
        self,
        case: DiffusionTestCase,
        summary: PerformanceSummary,
        scenario: "ScenarioConfig",
    ) -> None:
        """Check for potential significant performance improvements and record them."""
        is_improved = False
        threshold = BASELINE_CONFIG.improvement_threshold

        def is_sig_faster(actual, expected):
            if expected == 0 or expected is None:
                return False
            return actual < expected * (1 - threshold)

        def safe_get_metric(metric_dict, key):
            val = metric_dict.get(key)
            return val if val is not None else float("inf")

        # Check for any significant improvement
        if (
            is_sig_faster(summary.e2e_ms, scenario.expected_e2e_ms)
            or is_sig_faster(summary.avg_denoise_ms, scenario.expected_avg_denoise_ms)
            or is_sig_faster(
                summary.median_denoise_ms, scenario.expected_median_denoise_ms
            )
        ):
            is_improved = True
        # Combine metrics, always taking the better (lower) value
        new_stages = {
            stage: min(
                safe_get_metric(summary.stage_metrics, stage),
                safe_get_metric(scenario.stages_ms, stage),
            )
            for stage in set(summary.stage_metrics) | set(scenario.stages_ms)
        }
        new_denoise_steps = {
            step: min(
                safe_get_metric(summary.all_denoise_steps, step),
                safe_get_metric(scenario.denoise_step_ms, step),
            )
            for step in set(summary.all_denoise_steps.keys())
            | set(scenario.denoise_step_ms)
        }

        # Check for stage-level improvements
        if not is_improved:
            for stage, new_val in new_stages.items():
                if is_sig_faster(new_val, scenario.stages_ms.get(stage, float("inf"))):
                    is_improved = True
                    break
        if not is_improved:
            for step, new_val in new_denoise_steps.items():
                if is_sig_faster(
                    new_val, scenario.denoise_step_ms.get(step, float("inf"))
                ):
                    is_improved = True
                    break

        if is_improved:
            new_baseline = {
                "stages_ms": {k: round(v, 2) for k, v in new_stages.items()},
                "denoise_step_ms": {
                    str(k): round(v, 2) for k, v in new_denoise_steps.items()
                },
                "expected_e2e_ms": round(
                    min(summary.e2e_ms, scenario.expected_e2e_ms), 2
                ),
                "expected_avg_denoise_ms": round(
                    min(summary.avg_denoise_ms, scenario.expected_avg_denoise_ms), 2
                ),
                "expected_median_denoise_ms": round(
                    min(summary.median_denoise_ms, scenario.expected_median_denoise_ms),
                    2,
                ),
            }
            self._improved_baselines.append({"id": case.id, "baseline": new_baseline})

    def _dump_baseline_for_testcase(
        self,
        case: DiffusionTestCase,
        summary: "PerformanceSummary",
        missing_scenario: bool = False,
        measured_full_time: float | None = None,
    ) -> None:
        """Dump performance metrics as a JSON scenario for baselines."""
        import json

        denoise_steps_formatted = {
            str(k): round(v, 2) for k, v in summary.all_denoise_steps.items()
        }
        stages_formatted = {k: round(v, 2) for k, v in summary.stage_metrics.items()}

        baseline = {
            "stages_ms": stages_formatted,
            "denoise_step_ms": denoise_steps_formatted,
            "expected_e2e_ms": round(summary.e2e_ms, 2),
            "expected_avg_denoise_ms": round(summary.avg_denoise_ms, 2),
            "expected_median_denoise_ms": round(summary.median_denoise_ms, 2),
        }

        if measured_full_time is not None:
            baseline["estimated_full_test_time_s"] = round(measured_full_time, 1)

        # Video-specific metrics
        if case.server_args.modality == "video":
            if "per_frame_generation" not in baseline["stages_ms"]:
                baseline["stages_ms"]["per_frame_generation"] = (
                    round(summary.avg_frame_time_ms, 2)
                    if summary.avg_frame_time_ms
                    else None
                )
        action = "add" if missing_scenario else "update"
        output = f"""
{action} this baseline in the "scenarios" section of perf_baselines.json:

"{case.id}": {json.dumps(baseline, indent=4)}

"""
        logger.error(output)

    def _validate_consistency(
        self,
        case: DiffusionTestCase,
        content: bytes,
    ) -> None:
        """Validate output consistency against ground truth using CLIP similarity."""
        if os.environ.get("SGLANG_SKIP_CONSISTENCY", "0") == "1":
            logger.info(
                f"[Consistency] Skipping consistency check for {case.id} (SGLANG_SKIP_CONSISTENCY=1)"
            )
            return

        if not content:
            logger.warning(
                f"[Consistency] Skipping consistency check for {case.id}: "
                "content is empty (generation may have timed out)"
            )
            return

        num_gpus = case.server_args.num_gpus
        is_video = case.server_args.modality == "video"
        output_format = case.sampling_params.output_format

        if not gt_exists(
            case.id, num_gpus, is_video=is_video, output_format=output_format
        ):
            if _get_consistency_gt_dir() is not None:
                names = ", ".join(
                    get_consistency_gt_candidates(
                        case.id, num_gpus, is_video, output_format
                    )
                )
            else:
                names = ", ".join(
                    _consistency_gt_filenames(
                        case.id, num_gpus, is_video, output_format
                    )
                )
            logger.error(f"""
--- MISSING GROUND TRUTH DETECTED ---
GT image(s) not found for '{case.id}'.

Add the expected file(s) to sglang-ci-data in diffusion-ci/consistency_gt/ with naming (n=num_gpus).
  Image: {case.id}_{{n}}gpu.<ext> (ext from output_format: png, jpg, webp)
  Video: {case.id}_{{n}}gpu_frame_0.png, {case.id}_{{n}}gpu_frame_mid.png, {case.id}_{{n}}gpu_frame_last.png

For this case, expected file(s): {names}

Repository: https://github.com/sglang-bot/sglang-ci-data (path: diffusion-ci/consistency_gt/)

(Optional) Per-case override in consistency_threshold.json:
  "cases": {{
    "{case.id}": {{
      "clip_threshold": 0.92,
      "ssim_threshold": 0.95,
      "psnr_threshold": 28.0,
      "mean_abs_diff_threshold": 8.0
    }}
  }}
""")
            pytest.fail(
                f"GT not found for {case.id}. See logs for instructions to add GT."
            )

        gt_data = load_consistency_gt(
            case.id, num_gpus, is_video=is_video, output_format=output_format
        )
        thresholds = get_consistency_thresholds(case.id, is_video=is_video)

        if is_video:
            output_frames = extract_key_frames_from_video(content)
        else:
            output_frames = [image_bytes_to_numpy(content)]

        result = compare_with_gt(
            output_frames=output_frames,
            gt_data=gt_data,
            thresholds=thresholds,
            case_id=case.id,
        )

        if not result.passed:
            failed_frames = []
            gt_remote_files = get_consistency_gt_remote_files(
                case.id,
                num_gpus,
                is_video=is_video,
                output_format=output_format,
            )
            gt_remote_info = "\n".join(
                f"    - {filename}: {url}" for filename, url in gt_remote_files
            )
            for metric in result.frame_metrics:
                failed_metrics = []
                if not metric.clip_passed:
                    failed_metrics.append("clip")
                if not metric.ssim_passed:
                    failed_metrics.append("ssim")
                if not metric.psnr_passed:
                    failed_metrics.append("psnr")
                if not metric.mean_abs_diff_passed:
                    failed_metrics.append("mean_abs_diff")
                if failed_metrics:
                    failed_frames.append(
                        f"    - f{metric.frame_index} "
                        f"[{', '.join(failed_metrics)}] "
                        f"clip={metric.clip_similarity:.4f} "
                        f"ssim={metric.ssim:.4f} "
                        f"psnr={metric.psnr:.4f} "
                        f"mean_abs_diff={metric.mean_abs_diff:.4f}"
                    )
            pytest.fail(
                f"Consistency check failed for {case.id}:\n"
                f"  Metrics: sim={result.min_similarity:.4f}, "
                f"ssim={result.min_ssim:.4f}, "
                f"psnr={result.min_psnr:.4f}, "
                f"mean_abs_diff={result.max_mean_abs_diff:.4f}\n"
                f"  Thresholds: clip>={result.thresholds.clip_threshold}, "
                f"ssim>={result.thresholds.ssim_threshold}, "
                f"psnr>={result.thresholds.psnr_threshold}, "
                f"mean_abs_diff<={result.thresholds.mean_abs_diff_threshold}\n"
                f"  Failed frames:\n"
                + "\n".join(failed_frames)
                + f"\n  Compared GT files and links:\n{gt_remote_info}"
            )

        logger.info(
            f"[Consistency] {case.id}: PASSED "
            f"(min_similarity={result.min_similarity:.4f}, "
            f"min_ssim={result.min_ssim:.4f}, "
            f"min_psnr={result.min_psnr:.4f}, "
            f"max_mean_abs_diff={result.max_mean_abs_diff:.4f})"
        )

    def _save_gt_output(
        self,
        case: DiffusionTestCase,
        content: bytes,
    ) -> None:
        """Save generated content as ground truth files.

        Args:
            case: Test case configuration
            content: Generated content bytes (image or video)
        """
        gt_output_dir = os.environ.get("SGLANG_GT_OUTPUT_DIR")
        if not gt_output_dir:
            logger.error("SGLANG_GT_OUTPUT_DIR not set, cannot save GT output")
            return

        out_dir = Path(gt_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        num_gpus = case.server_args.num_gpus
        is_video = case.server_args.modality == "video"

        if is_video:
            # Extract key frames from video
            frames = extract_key_frames_from_video(
                content, num_frames=case.sampling_params.num_frames
            )

            if len(frames) != 3:
                logger.warning(
                    f"{case.id}: expected 3 frames, got {len(frames)}, skipping frame save"
                )
                return

            # Save frames (reuse naming from _consistency_gt_filenames)
            filenames = _consistency_gt_filenames(case.id, num_gpus, is_video=True)
            from PIL import Image

            for frame, fn in zip(frames, filenames):
                frame_path = out_dir / fn
                Image.fromarray(frame).save(frame_path)
                logger.info(f"Saved GT frame: {frame_path}")
        else:
            # Save image
            from sglang.multimodal_gen.test.test_utils import detect_image_format

            detected_format = detect_image_format(content)
            filenames = _consistency_gt_filenames(
                case.id, num_gpus, is_video=False, output_format=detected_format
            )
            output_path = out_dir / filenames[0]
            output_path.write_bytes(content)
            logger.info(f"Saved GT image: {output_path} (format: {detected_format})")

    def _test_lora_api_functionality(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
        generate_fn: Callable[[str, openai.Client], tuple[str, bytes]],
    ) -> None:
        """
        Test LoRA API functionality with end-to-end validation: merge, unmerge, and set_lora.
        This test verifies that each API call succeeds AND that generation works after each operation.
        """
        base_url = f"http://localhost:{ctx.port}/v1"
        client = OpenAI(base_url=base_url, api_key="dummy")

        # Test 1: unmerge_lora_weights - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing unmerge_lora_weights for %s", case.id)
        resp = requests.post(f"{base_url}/unmerge_lora_weights")
        assert resp.status_code == 200, f"unmerge_lora_weights failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after unmerge for %s", case.id)
        rid_after_unmerge, _ = generate_fn(case.id, client)
        assert rid_after_unmerge is not None, "Generation after unmerge failed"
        logger.info("[LoRA E2E] Generation after unmerge succeeded")

        # Test 2: merge_lora_weights - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing merge_lora_weights for %s", case.id)
        resp = requests.post(f"{base_url}/merge_lora_weights")
        assert resp.status_code == 200, f"merge_lora_weights failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after re-merge for %s", case.id)
        rid_after_merge, _ = generate_fn(case.id, client)
        assert rid_after_merge is not None, "Generation after merge failed"
        logger.info("[LoRA E2E] Generation after merge succeeded")

        # Test 3: set_lora (re-set the same adapter) - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing set_lora for %s", case.id)
        resp = requests.post(f"{base_url}/set_lora", json={"lora_nickname": "default"})
        assert resp.status_code == 200, f"set_lora failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after set_lora for %s", case.id)
        rid_after_set, _ = generate_fn(case.id, client)
        assert rid_after_set is not None, "Generation after set_lora failed"
        logger.info("[LoRA E2E] Generation after set_lora succeeded")

        # Test 4: list_loras - API should return the expected list of LoRA adapters
        logger.info("[LoRA E2E] Testing list_loras for %s", case.id)
        resp = requests.get(f"{base_url}/list_loras")
        assert resp.status_code == 200, f"list_loras failed: {resp.text}"
        lora_info = resp.json()
        logger.info("[LoRA E2E] list_loras returned %s", lora_info)
        assert (
            isinstance(lora_info["loaded_adapters"], list)
            and len(lora_info["loaded_adapters"]) > 0
        ), "loaded_adapters should be a non-empty list"
        assert any(
            a.get("nickname") == "default" for a in lora_info["loaded_adapters"]
        ), f"nickname 'default' not found in loaded_adapters: {lora_info['loaded_adapters']}"
        logger.info("[LoRA E2E] list_loras returned expected LoRA adapters")

        logger.info("[LoRA E2E] All LoRA API E2E tests passed for %s", case.id)

    def _test_lora_dynamic_switch_e2e(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
        generate_fn: Callable[[str, openai.Client], tuple[str, bytes]],
        second_lora_path: str,
    ) -> None:
        """
        Test dynamic LoRA switching with end-to-end validation.
        This test verifies that switching between LoRA adapters works correctly
        and generation succeeds after each switch.
        """
        base_url = f"http://localhost:{ctx.port}/v1"
        client = OpenAI(base_url=base_url, api_key="dummy")

        # Test 1: Generate with initial LoRA
        logger.info(
            "[LoRA Switch E2E] Testing generation with initial LoRA for %s", case.id
        )
        rid_initial, _ = generate_fn(case.id, client)
        assert rid_initial is not None, "Generation with initial LoRA failed"
        logger.info("[LoRA Switch E2E] Generation with initial LoRA succeeded")

        # Test 2: Switch to second LoRA and generate
        logger.info(
            "[LoRA Switch E2E] Switching to second LoRA adapter for %s", case.id
        )
        resp = requests.post(
            f"{base_url}/set_lora",
            json={"lora_nickname": "lora2", "lora_path": second_lora_path},
        )
        assert (
            resp.status_code == 200
        ), f"set_lora to second adapter failed: {resp.text}"

        logger.info(
            "[LoRA Switch E2E] Verifying generation with second LoRA for %s", case.id
        )
        rid_second, _ = generate_fn(case.id, client)
        assert rid_second is not None, "Generation with second LoRA failed"
        logger.info("[LoRA Switch E2E] Generation with second LoRA succeeded")

        # Test 3: Switch back to original LoRA and generate
        logger.info("[LoRA Switch E2E] Switching back to original LoRA for %s", case.id)
        resp = requests.post(f"{base_url}/set_lora", json={"lora_nickname": "default"})
        assert resp.status_code == 200, f"set_lora back to default failed: {resp.text}"

        logger.info(
            "[LoRA Switch E2E] Verifying generation after switching back for %s",
            case.id,
        )
        rid_switched_back, _ = generate_fn(case.id, client)
        assert rid_switched_back is not None, "Generation after switching back failed"
        logger.info("[LoRA Switch E2E] Generation after switching back succeeded")

        logger.info(
            "[LoRA Switch E2E] All dynamic switch E2E tests passed for %s", case.id
        )

    def _test_dynamic_lora_loading(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
    ) -> None:
        """
        Test dynamic LoRA loading after server startup.

        This test reproduces the LayerwiseOffload + set_lora issue:
        - Server starts WITHOUT lora_path (LayerwiseOffloadManager initializes first)
        - Then set_lora is called via API to load LoRA dynamically
        - This tests the interaction between layerwise offload and dynamic LoRA loading
        """
        base_url = f"http://localhost:{ctx.port}/v1"
        dynamic_lora_path = case.server_args.dynamic_lora_path

        # Call set_lora to load LoRA dynamically after server startup
        logger.info(
            "[Dynamic LoRA] Loading LoRA dynamically via set_lora API for %s", case.id
        )
        logger.info("[Dynamic LoRA] LoRA path: %s", dynamic_lora_path)
        resp = requests.post(
            f"{base_url}/set_lora",
            json={"lora_nickname": "default", "lora_path": dynamic_lora_path},
        )
        assert resp.status_code == 200, f"Dynamic set_lora failed: {resp.text}"
        logger.info("[Dynamic LoRA] set_lora succeeded for %s", case.id)

    def _test_multi_lora_e2e(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
        generate_fn: Callable[[str, openai.Client], tuple[str, bytes]],
        first_lora_path: str,
        second_lora_path: str,
    ) -> None:
        """
        Test multiple LoRA adapters with different set_lora input scenarios.
        Tests: basic multi-LoRA, different strengths, cached adapters, switch back to single.
        """
        base_url = f"http://localhost:{ctx.port}/v1"
        client = OpenAI(base_url=base_url, api_key="dummy")

        # Test 1: Basic multi-LoRA with list format
        resp = requests.post(
            f"{base_url}/set_lora",
            json={
                "lora_nickname": ["default", "lora2"],
                "lora_path": [first_lora_path, second_lora_path],
                "target": "all",
                "strength": [1.0, 1.0],
            },
        )
        assert (
            resp.status_code == 200
        ), f"set_lora with multiple adapters failed: {resp.text}"
        rid, _ = generate_fn(case.id, client)
        assert rid is not None

        # Test 2: Different strengths
        resp = requests.post(
            f"{base_url}/set_lora",
            json={
                "lora_nickname": ["default", "lora2"],
                "lora_path": [first_lora_path, second_lora_path],
                "target": "all",
                "strength": [0.8, 0.5],
            },
        )
        assert (
            resp.status_code == 200
        ), f"set_lora with different strengths failed: {resp.text}"
        rid, _ = generate_fn(case.id, client)
        assert rid is not None

        # Test 3: Different targets
        requests.post(f"{base_url}/set_lora", json={"lora_nickname": "default"})
        resp = requests.post(
            f"{base_url}/set_lora",
            json={
                "lora_nickname": ["default", "lora2"],
                "lora_path": [first_lora_path, second_lora_path],
                "target": ["transformer", "transformer_2"],
                "strength": [0.8, 0.5],
            },
        )
        assert (
            resp.status_code == 200
        ), f"set_lora with cached adapters failed: {resp.text}"
        rid, _ = generate_fn(case.id, client)
        assert rid is not None

        # Test 4: Switch back to single LoRA
        resp = requests.post(f"{base_url}/set_lora", json={"lora_nickname": "default"})
        assert (
            resp.status_code == 200
        ), f"set_lora back to single adapter failed: {resp.text}"
        rid, _ = generate_fn(case.id, client)
        assert rid is not None

        logger.info("[Multi-LoRA] All multi-LoRA tests passed for %s", case.id)

    def _test_v1_models_endpoint(
        self, ctx: ServerContext, case: DiffusionTestCase
    ) -> None:
        """
        Test /v1/models endpoint returns OpenAI-compatible response.
        This endpoint is required for sgl-model-gateway router compatibility.
        """
        base_url = f"http://localhost:{ctx.port}"

        # Test GET /v1/models
        logger.info("[Models API] Testing GET /v1/models for %s", case.id)
        resp = requests.get(f"{base_url}/v1/models")
        assert resp.status_code == 200, f"/v1/models failed: {resp.text}"

        data = resp.json()
        assert (
            data["object"] == "list"
        ), f"Expected object='list', got {data.get('object')}"
        assert len(data["data"]) >= 1, "Expected at least one model in response"

        model = data["data"][0]
        assert "id" in model, "Model missing 'id' field"
        assert (
            model["object"] == "model"
        ), f"Expected object='model', got {model.get('object')}"
        assert (
            model["id"] == case.server_args.model_path
        ), f"Model ID mismatch: expected {case.server_args.model_path}, got {model['id']}"

        # Verify extended diffusion-specific fields
        assert "num_gpus" in model, "Model missing 'num_gpus' field"
        assert "task_type" in model, "Model missing 'task_type' field"
        assert "dit_precision" in model, "Model missing 'dit_precision' field"
        assert "vae_precision" in model, "Model missing 'vae_precision' field"
        assert (
            model["num_gpus"] == case.server_args.num_gpus
        ), f"num_gpus mismatch: expected {case.server_args.num_gpus}, got {model['num_gpus']}"
        # Verify task_type is consistent with the modality specified in the test config.
        # We can't access pipeline_config from test config, but we can validate against modality.
        modality_to_valid_task_types = {
            "image": {"T2I", "I2I", "TI2I"},
            "video": {"T2V", "I2V", "TI2V"},
            "3d": {"I2M"},
        }
        valid_task_types = modality_to_valid_task_types.get(
            case.server_args.modality, set()
        )
        assert model["task_type"] in valid_task_types, (
            f"task_type '{model['task_type']}' not valid for modality "
            f"'{case.server_args.modality}'. Expected one of: {valid_task_types}"
        )
        logger.info(
            "[Models API] GET /v1/models returned valid response with extended fields"
        )

        # Test GET /v1/models/{model_path}
        model_path = model["id"]
        logger.info("[Models API] Testing GET /v1/models/%s", model_path)
        resp = requests.get(f"{base_url}/v1/models/{model_path}")
        assert resp.status_code == 200, f"/v1/models/{model_path} failed: {resp.text}"

        single_model = resp.json()
        assert single_model["id"] == model_path, "Single model ID mismatch"
        assert single_model["object"] == "model", "Single model object type mismatch"

        # Verify extended fields on single model endpoint too
        assert "num_gpus" in single_model, "Single model missing 'num_gpus' field"
        assert "task_type" in single_model, "Single model missing 'task_type' field"
        assert single_model["task_type"] in valid_task_types, (
            f"Single model task_type '{single_model['task_type']}' not valid for modality "
            f"'{case.server_args.modality}'. Expected one of: {valid_task_types}"
        )
        logger.info(
            "[Models API] GET /v1/models/{model_path} returned valid response with extended fields"
        )

        # Test GET /v1/models/{non_existent_model} returns 404
        logger.info("[Models API] Testing GET /v1/models/non_existent_model")
        resp = requests.get(f"{base_url}/v1/models/non_existent_model")
        assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"
        error_data = resp.json()
        assert "error" in error_data, "404 response missing 'error' field"
        assert (
            error_data["error"]["code"] == "model_not_found"
        ), f"Incorrect error code: {error_data['error'].get('code')}"
        logger.info("[Models API] GET /v1/models/non_existent returns 404 as expected")

        logger.info("[Models API] All /v1/models tests passed for %s", case.id)

    def _test_t2v_rejects_input_reference(
        self, ctx: ServerContext, case: DiffusionTestCase
    ) -> None:
        if case.server_args.modality != "video":
            return

        base_url = f"http://localhost:{ctx.port}"
        resp = requests.get(f"{base_url}/v1/models")
        assert resp.status_code == 200, f"/v1/models failed: {resp.text}"
        data = resp.json().get("data", [])
        if not data:
            pytest.fail("/v1/models returned empty model list")

        task_type = data[0].get("task_type")
        if task_type != "T2V":
            return

        prompt = case.sampling_params.prompt or "test"
        payload = {"prompt": prompt, "input_reference": "dummy"}
        if case.sampling_params.output_size:
            payload["size"] = case.sampling_params.output_size

        resp = requests.post(f"{base_url}/v1/videos", json=payload)
        assert (
            resp.status_code == 400
        ), f"Expected 400 for T2V input_reference, got {resp.status_code}: {resp.text}"
        detail = resp.json().get("detail", "")
        assert (
            "input_reference is not supported" in detail
        ), f"Unexpected error detail for T2V input_reference: {detail}"

    def test_diffusion_generation(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        """Single parametrized test that runs for all cases.

        This test performs:
        1. Generation
        2. Performance validation against baselines
        3. Consistency validation against ground truth

        Pytest will execute this test once per case in ONE_GPU_CASES,
        with test IDs like:
        - test_diffusion_generation[qwen_image_text]
        - test_diffusion_generation[qwen_image_edit]
        - etc.
        """
        # Check if we're in GT generation mode
        is_gt_gen_mode = os.environ.get("SGLANG_GEN_GT", "0") == "1"

        # GT generation also needs the dynamic set_lora step before generation.
        if case.run_lora_dynamic_load_check:
            self._test_dynamic_lora_loading(diffusion_server, case)

        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )

        # Single generation - output is reused for both validations
        perf_record, content = self.run_and_collect(
            diffusion_server,
            case.id,
            generate_fn,
            collect_perf=not is_gt_gen_mode,
        )

        if is_gt_gen_mode:
            # GT generation mode: save output and skip all validations/tests
            self._save_gt_output(case, content)
            return

        failures: list[tuple[str, str]] = []

        def run_case_check(name: str, fn: Callable[[], None]) -> None:
            try:
                fn()
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                failures.append((name, str(exc)))

        run_case_check(
            "performance",
            lambda: self._validate_and_record(case, perf_record),
        )

        if case.server_args.custom_validator == "mesh":
            from sglang.multimodal_gen.test.server.test_server_utils import (
                MESH_OUTPUT_PATHS,
                validate_mesh_correctness,
            )

            def validate_mesh_output() -> None:
                mesh_path = MESH_OUTPUT_PATHS.pop(case.id, None)
                if mesh_path:
                    validate_mesh_correctness(mesh_path)

            run_case_check("mesh correctness", validate_mesh_output)

        if case.run_models_api_check:
            run_case_check(
                "/v1/models endpoint",
                lambda: self._test_v1_models_endpoint(diffusion_server, case),
            )
        if case.run_t2v_input_reference_check:
            run_case_check(
                "t2v input_reference rejection",
                lambda: self._test_t2v_rejects_input_reference(diffusion_server, case),
            )

        if case.run_consistency_check:
            run_case_check(
                "consistency",
                lambda: self._validate_consistency(case, content),
            )

        if case.run_lora_basic_api_check:
            run_case_check(
                "LoRA basic API",
                lambda: self._test_lora_api_functionality(
                    diffusion_server, case, generate_fn
                ),
            )

        if case.run_lora_dynamic_switch_check:
            run_case_check(
                "LoRA dynamic switch",
                lambda: self._test_lora_dynamic_switch_e2e(
                    diffusion_server,
                    case,
                    generate_fn,
                    case.server_args.second_lora_path,
                ),
            )

        if case.run_multi_lora_api_check:
            run_case_check(
                "multi-LoRA API",
                lambda: self._test_multi_lora_e2e(
                    diffusion_server,
                    case,
                    generate_fn,
                    case.server_args.lora_path,
                    case.server_args.second_lora_path,
                ),
            )

        if failures:
            formatted_failures = []
            for name, message in failures:
                if "\n" in message:
                    formatted_failures.append(f"[{name}]\n{message}")
                else:
                    formatted_failures.append(f"[{name}] {message}")
            pytest.fail(
                f"Diffusion testcase '{case.id}' failed {len(failures)} check(s):\n\n"
                + "\n\n".join(formatted_failures),
                pytrace=False,
            )
