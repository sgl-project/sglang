"""
Config-driven diffusion performance test with pytest parametrization.


If the actual run is significantly better than the baseline, the improved cases with their updated baseline will be printed
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import openai
import pytest
import requests
from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.server.conftest import _GLOBAL_PERF_RESULTS
from sglang.multimodal_gen.test.server.test_server_utils import (
    VALIDATOR_REGISTRY,
    PerformanceValidator,
    ServerContext,
    ServerManager,
    WarmupRunner,
    download_image_from_url,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    DiffusionTestCase,
    PerformanceSummary,
    ScenarioConfig,
)
from sglang.multimodal_gen.test.test_utils import (
    get_dynamic_server_port,
    is_image_url,
    read_perf_logs,
    wait_for_req_perf_record,
)

logger = init_logger(__name__)


@pytest.fixture
def diffusion_server(case: DiffusionTestCase) -> ServerContext:
    """Start a diffusion server for a single case and tear it down afterwards."""
    default_port = get_dynamic_server_port()
    port = int(os.environ.get("SGLANG_TEST_SERVER_PORT", default_port))
    server_args = case.server_args
    sampling_params = case.sampling_params
    extra_args = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")
    extra_args += (
        f" --num-gpus {server_args.num_gpus} --ulysses-degree {server_args.num_gpus}"
    )
    # LoRA support
    if server_args.lora_path:
        extra_args += f" --lora-path {server_args.lora_path}"

    # start server
    manager = ServerManager(
        model=server_args.model_path,
        port=port,
        wait_deadline=float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200")),
        extra_args=extra_args,
    )
    ctx = manager.start()

    try:
        # Reconstruct output size for OpenAI API
        output_size = sampling_params.output_size
        warmup = WarmupRunner(
            port=ctx.port,
            model=server_args.model_path,
            prompt=sampling_params.prompt or "A colorful raccoon icon",
            output_size=output_size,
        )
        warmup.run_text_warmups(server_args.warmup_text)

        if (
            case.server_args.warmup_edit > 0
            and case.sampling_params.prompt
            and sampling_params.image_path
        ):
            # Handle URL or local path
            if is_image_url(sampling_params.image_path):
                image_path = download_image_from_url(str(sampling_params.image_path))
            else:
                image_path = Path(sampling_params.image_path)

            warmup.run_edit_warmups(
                count=server_args.warmup_edit,
                edit_prompt=sampling_params.prompt,
                image_path=image_path,
            )
    except Exception as exc:
        logger.error("Warm-up failed for %s: %s", case.id, exc)
        ctx.cleanup()
        raise

    try:
        yield ctx
    finally:
        ctx.cleanup()


class DiffusionServerBase:
    """Performance tests for all diffusion models/scenarios.

    This single test class runs against all cases defined in ONE_GPU_CASES.
    Each case gets its own server instance via the parametrized fixture.
    """

    _perf_results: list[dict[str, Any]] = []
    _improved_baselines: list[dict[str, Any]] = []

    @classmethod
    def setup_class(cls):
        cls._perf_results = []
        cls._improved_baselines = []

    @classmethod
    def teardown_class(cls):
        for result in cls._perf_results:
            result["class_name"] = cls.__name__
            _GLOBAL_PERF_RESULTS.append(result)

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
        generate_fn: Callable[[str, openai.Client], str],
    ) -> RequestPerfRecord:
        """Run generation and collect performance records."""
        log_path = ctx.perf_log_path
        prev_len = len(read_perf_logs(log_path))
        log_wait_timeout = 30

        client = self._client(ctx)
        rid = generate_fn(case_id, client)

        req_perf_record, _ = wait_for_req_perf_record(
            rid,
            prev_len,
            log_path,
            timeout=log_wait_timeout,
        )

        return req_perf_record

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

        validator_name = case.server_args.custom_validator or "default"
        validator_class = VALIDATOR_REGISTRY.get(validator_name, PerformanceValidator)

        validator = validator_class(
            scenario=scenario,
            tolerances=BASELINE_CONFIG.tolerances,
            step_fractions=BASELINE_CONFIG.step_fractions,
        )

        summary = validator.collect_metrics(perf_record)

        if is_baseline_generation_mode or missing_scenario:
            self._dump_baseline_for_testcase(case, summary, missing_scenario)
            if missing_scenario:
                pytest.fail(f"Testcase '{case.id}' not found in perf_baselines.json")
            return

        self._check_for_improvement(case, summary, scenario)

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

    def _test_lora_api_functionality(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
        generate_fn: Callable[[str, openai.Client], str],
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
        output_after_unmerge = generate_fn(case.id, client)
        assert output_after_unmerge is not None, "Generation after unmerge failed"
        logger.info("[LoRA E2E] Generation after unmerge succeeded")

        # Test 2: merge_lora_weights - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing merge_lora_weights for %s", case.id)
        resp = requests.post(f"{base_url}/merge_lora_weights")
        assert resp.status_code == 200, f"merge_lora_weights failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after re-merge for %s", case.id)
        output_after_merge = generate_fn(case.id, client)
        assert output_after_merge is not None, "Generation after merge failed"
        logger.info("[LoRA E2E] Generation after merge succeeded")

        # Test 3: set_lora (re-set the same adapter) - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing set_lora for %s", case.id)
        resp = requests.post(f"{base_url}/set_lora", json={"lora_nickname": "default"})
        assert resp.status_code == 200, f"set_lora failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after set_lora for %s", case.id)
        output_after_set = generate_fn(case.id, client)
        assert output_after_set is not None, "Generation after set_lora failed"
        logger.info("[LoRA E2E] Generation after set_lora succeeded")

        logger.info("[LoRA E2E] All LoRA API E2E tests passed for %s", case.id)

    def _test_lora_dynamic_switch_e2e(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
        generate_fn: Callable[[str, openai.Client], str],
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
        output_initial = generate_fn(case.id, client)
        assert output_initial is not None, "Generation with initial LoRA failed"
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
        output_second = generate_fn(case.id, client)
        assert output_second is not None, "Generation with second LoRA failed"
        logger.info("[LoRA Switch E2E] Generation with second LoRA succeeded")

        # Test 3: Switch back to original LoRA and generate
        logger.info("[LoRA Switch E2E] Switching back to original LoRA for %s", case.id)
        resp = requests.post(f"{base_url}/set_lora", json={"lora_nickname": "default"})
        assert resp.status_code == 200, f"set_lora back to default failed: {resp.text}"

        logger.info(
            "[LoRA Switch E2E] Verifying generation after switching back for %s",
            case.id,
        )
        output_switched_back = generate_fn(case.id, client)
        assert (
            output_switched_back is not None
        ), "Generation after switching back failed"
        logger.info("[LoRA Switch E2E] Generation after switching back succeeded")

        logger.info(
            "[LoRA Switch E2E] All dynamic switch E2E tests passed for %s", case.id
        )

    def test_diffusion_perf(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        """Single parametrized test that runs for all cases.

        Pytest will execute this test once per case in ONE_GPU_CASES,
        with test IDs like:
        - test_diffusion_perf[qwen_image_text]
        - test_diffusion_perf[qwen_image_edit]
        - etc.
        """
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )
        perf_record = self.run_and_collect(
            diffusion_server,
            case.id,
            generate_fn,
        )
        self._validate_and_record(case, perf_record)

        # LoRA API functionality test with E2E validation (only for LoRA-enabled cases)
        if case.server_args.lora_path:
            self._test_lora_api_functionality(diffusion_server, case, generate_fn)
