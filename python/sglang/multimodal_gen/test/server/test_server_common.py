"""
Config-driven diffusion generation test with pytest parametrization.


Each collected request prints a performance log before validation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import openai
import pytest
import requests
from openai import OpenAI

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.server import conftest
from sglang.multimodal_gen.test.server.realtime_consistency import (
    pop_realtime_key_frames,
    pop_realtime_perf_stats,
    validate_realtime_perf_stats,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    VALIDATOR_REGISTRY,
    PerformanceValidator,
    ServerContext,
    ServerManager,
    get_generate_fn,
    parse_dimensions,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    BASELINE_CONFIG,
    DiffusionSamplingParams,
    DiffusionTestCase,
    PerformanceSummary,
    ScenarioConfig,
    get_model_task_type_for_server_args,
    get_perf_baseline_path,
)
from sglang.multimodal_gen.test.test_utils import (
    SGL_TEST_FILES_CI_DATA_REVISION,
    _consistency_gt_filenames,
    _get_consistency_gt_dir,
    action_gt_exists,
    compare_with_gt,
    detect_image_format,
    extract_key_frames_from_video,
    get_action_consistency_gt_candidates,
    get_action_consistency_gt_remote_files,
    get_consistency_gt_candidates,
    get_consistency_gt_remote_files,
    get_consistency_threshold_path,
    get_consistency_thresholds,
    get_dynamic_server_port,
    gt_exists,
    image_bytes_to_numpy,
    load_action_consistency_gt,
    load_consistency_gt,
    save_consistency_failure_artifact,
    validate_image,
    wait_for_req_perf_record,
)

logger = init_logger(__name__)

# Track test cases missing estimated_full_test_time_s for time measurement output
_MISSING_ESTIMATED_TIME_CASES: set[str] = set()
_PENDING_BASELINE_DUMPS: dict[str, tuple[PerformanceSummary, bool]] = {}
_OPENAI_REQUEST_TIMEOUT_SECS = float(
    os.environ.get("SGLANG_TEST_OPENAI_REQUEST_TIMEOUT_SECS", "600")
)
_SERVER_EXIT_POLL_INTERVAL_SECS = float(
    os.environ.get("SGLANG_TEST_SERVER_EXIT_POLL_INTERVAL_SECS", "1")
)
_CONTROL_API_TIMEOUT_SECS = float(
    os.environ.get("SGLANG_TEST_CONTROL_API_TIMEOUT_SECS", "300")
)
_DYNAMIC_BATCH_LOG_TIMEOUT_SECS = float(
    os.environ.get("SGLANG_TEST_DYNAMIC_BATCH_LOG_TIMEOUT_SECS", "10")
)
_DYNAMIC_BATCH_REQUEST_TIMEOUT_SECS = float(
    os.environ.get("SGLANG_TEST_DYNAMIC_BATCH_REQUEST_TIMEOUT_SECS", "120")
)
_DYNAMIC_BATCH_THREAD_JOIN_TIMEOUT_SECS = float(
    os.environ.get("SGLANG_TEST_DYNAMIC_BATCH_THREAD_JOIN_TIMEOUT_SECS", "5")
)
_SERVER_FATAL_LOG_PATTERNS = (
    "terminate called after throwing an instance of",
    "Fatal Python error:",
    "Segmentation fault",
    "Aborted (core dumped)",
)
_CASE_LOG_SEPARATOR = "=" * 88


def _print_case_log_separator(case_id: str, state: str) -> None:
    print(
        f"\n{_CASE_LOG_SEPARATOR}\n"
        f"[server-test] {state}: {case_id}\n"
        f"{_CASE_LOG_SEPARATOR}",
        flush=True,
    )


@pytest.fixture
def diffusion_server(case: DiffusionTestCase) -> ServerContext:
    """Start a diffusion server for a single case and tear it down afterwards."""
    _fixture_start_time = time.perf_counter()
    server_args = case.server_args
    _print_case_log_separator(case.id, "BEGIN diffusion testcase")

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
    extra_args = os.environ.get("SGLANG_TEST_SERVE_ARGS", "")
    extra_args = f"--model-type diffusion {extra_args}".strip()

    extra_args += f" --num-gpus {server_args.num_gpus}"

    if server_args.tp_size is not None:
        extra_args += f" --tp-size {server_args.tp_size}"

    if server_args.ulysses_degree is not None:
        extra_args += f" --ulysses-degree {server_args.ulysses_degree}"

    if server_args.dit_layerwise_offload:
        extra_args += " --dit-layerwise-offload true"

    if server_args.dit_offload_prefetch_size:
        extra_args += (
            f" --dit-offload-prefetch-size {server_args.dit_offload_prefetch_size}"
        )

    if server_args.text_encoder_cpu_offload:
        extra_args += " --text-encoder-cpu-offload"

    if server_args.ring_degree is not None:
        extra_args += f" --ring-degree {server_args.ring_degree}"

    if server_args.cfg_parallel:
        extra_args += " --enable-cfg-parallel"

    # LoRA support
    if server_args.lora_path:
        extra_args += f" --lora-path {server_args.lora_path}"

    # Strict ports: fail immediately if port is occupied instead of silently
    # picking another one (which causes the test client to connect to the wrong server).
    extra_args += " --strict-ports"

    # Shape-only mesh cases (e.g. hunyuan3d_shape_gen) validate geometry via
    # mesh-correctness and must NOT run the paint/texture stages, whose
    # verification checks texture artifacts (paint_mesh/normal_maps/renderer)
    # that the shape-only path never produces. Inject a pipeline-config override
    # disabling paint for these cases.
    if server_args.custom_validator == "mesh":
        import json as _json
        import tempfile as _tempfile

        _paint_off_cfg = os.path.join(
            _tempfile.gettempdir(), f"{case.id}_paint_off.json"
        )
        with open(_paint_off_cfg, "w") as _f:
            _json.dump({"paint_enable": False}, _f)
        extra_args += f" --config {_paint_off_cfg}"

    for arg in server_args.extras:
        extra_args += f" {arg}"

    # Build custom environment variables
    env_vars = {}
    if server_args.enable_cache_dit:
        env_vars["SGLANG_CACHE_DIT_ENABLED"] = "true"
    env_vars.update(server_args.env_vars)

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
        _print_case_log_separator(case.id, "FAILED during server startup")
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
                f"File: {get_perf_baseline_path()}\n\n"
                f'    "{case.id}": {{\n'
                f"        ...\n"
                f'        "estimated_full_test_time_s": {_measured_full_time:.1f}\n'
                f"    }}\n"
                f'{"=" * 60}\n'
            )
        _print_case_log_separator(case.id, "END diffusion testcase")


class DiffusionServerBase:
    """Performance tests for all diffusion models/scenarios.

    This single test class runs against all cases defined in ONE_GPU_CASES.
    Each case gets its own server instance via the parametrized fixture.
    """

    _perf_results: list[dict[str, Any]] = []
    _pytest_config = None  # Store pytest config for stash access

    @classmethod
    def setup_class(cls):
        cls._perf_results = []

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

    @pytest.fixture(autouse=True)
    def _capture_pytest_config(self, request):
        """Capture pytest config for use in teardown_class."""
        self.__class__._pytest_config = request.config

    def _client(
        self,
        ctx: ServerContext,
        *,
        timeout: float | None = None,
    ) -> OpenAI:
        """Get an OpenAI client for the server.

        Args:
            ctx: Running server context whose API should receive requests.
            timeout: Optional per-request timeout in seconds.

        Returns:
            Configured OpenAI client with retries disabled.
        """
        return OpenAI(
            api_key="sglang-anything",
            base_url=f"http://localhost:{ctx.port}/v1",
            timeout=(_OPENAI_REQUEST_TIMEOUT_SECS if timeout is None else timeout),
            max_retries=0,
        )

    def _test_chat_completion_smoke(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
    ) -> None:
        """Validate a text-generating diffusion server through the Chat API.

        Args:
            ctx: Running server context for the test case.
            case: Image-to-text test case with one prompt and one image URL.

        Raises:
            AssertionError: If the request inputs or response contract are invalid.
        """
        sampling_params = case.sampling_params
        assert sampling_params is not None
        prompt = sampling_params.prompt
        image_url = sampling_params.image_path
        assert (
            isinstance(prompt, str) and prompt.strip()
        ), f"{case.id}: chat smoke requires a non-empty prompt"
        assert (
            isinstance(image_url, str) and image_url.strip()
        ), f"{case.id}: chat smoke requires one image URL"

        def generate_fn(
            case_id: str,
            client: openai.Client,
        ) -> tuple[str, bytes]:
            response = client.chat.completions.create(
                model=case.server_args.model_path,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                max_completion_tokens=64,
                temperature=0,
            )

            assert (
                isinstance(response.id, str) and response.id
            ), f"{case_id}: chat completion must return a request id"
            assert (
                len(response.choices) == 1
            ), f"{case_id}: expected exactly one chat completion choice"
            choice = response.choices[0]
            content = choice.message.content
            assert (
                isinstance(content, str) and content.strip()
            ), f"{case_id}: assistant content must be non-empty text"
            assert choice.finish_reason in (
                "stop",
                "length",
            ), f"{case_id}: unsupported finish reason {choice.finish_reason!r}"

            usage = response.usage
            assert usage is not None, f"{case_id}: token usage must be present"
            assert (
                usage.prompt_tokens > 0
            ), f"{case_id}: prompt token count must be positive"
            assert (
                usage.completion_tokens > 0
            ), f"{case_id}: completion token count must be positive"
            assert usage.total_tokens == (
                usage.prompt_tokens + usage.completion_tokens
            ), f"{case_id}: total token count must equal prompt plus completion"
            return response.id, content.encode("utf-8")

        client = self._client(ctx)
        self._run_generation_with_server_watchdog(
            ctx,
            case.id,
            generate_fn,
            client,
        )

    def _test_dynamic_batching_smoke(
        self,
        ctx: ServerContext,
        case: DiffusionTestCase,
    ) -> None:
        """Prove that two independent image requests run as one dynamic batch.

        Args:
            ctx: Running server context for the test case.
            case: Image generation case with two compatible batching requests.

        Raises:
            AssertionError: If responses, output images, or batching evidence are invalid.
        """
        requests_to_batch = case.dynamic_batching_requests
        assert (
            len(requests_to_batch) == 2
        ), f"{case.id}: dynamic batching smoke requires exactly two requests"

        def generate_image_response(
            index: int,
            sampling_params: DiffusionSamplingParams,
            client: openai.Client,
        ) -> tuple[str, bytes]:
            """Issue one Images API request and validate its response contract."""
            prompt = sampling_params.prompt
            assert (
                isinstance(prompt, str) and prompt.strip()
            ), f"{case.id}: dynamic batching request {index} needs a prompt"
            response = client.images.with_raw_response.generate(
                model=case.server_args.model_path,
                prompt=prompt,
                n=1,
                size=sampling_params.output_size,
                response_format="b64_json",
                output_format=sampling_params.output_format,
                extra_body=(
                    dict(sampling_params.extras) if sampling_params.extras else None
                ),
            )
            parsed = response.parse()
            assert (
                isinstance(parsed.id, str) and parsed.id
            ), f"{case.id}: dynamic batching response {index} needs an id"
            assert (
                len(parsed.data) == 1
            ), f"{case.id}: dynamic batching response {index} must contain one image"
            image_b64 = parsed.data[0].b64_json
            assert isinstance(
                image_b64, str
            ), f"{case.id}: dynamic batching response {index} needs base64 image data"
            validate_image(image_b64)
            image_bytes = base64.b64decode(image_b64)
            requested_format = sampling_params.output_format
            if requested_format is not None:
                normalized_format = (
                    "jpeg"
                    if requested_format.lower() == "jpg"
                    else requested_format.lower()
                )
                assert detect_image_format(image_bytes) == normalized_format, (
                    f"{case.id}: dynamic batching response {index} ignored requested "
                    f"output format {requested_format!r}"
                )
            image_array = image_bytes_to_numpy(image_bytes)
            expected_width, expected_height = parse_dimensions(
                sampling_params.output_size
            )
            assert expected_width is not None and expected_height is not None
            assert image_array.shape[:2] == (expected_height, expected_width), (
                f"{case.id}: dynamic batching response {index} has shape "
                f"{image_array.shape[:2]}, expected "
                f"{(expected_height, expected_width)}"
            )
            return parsed.id, image_bytes

        def image_distance(first: bytes, second: bytes) -> float:
            """Return mean absolute RGB pixel distance for request identity checks."""
            first_array = image_bytes_to_numpy(first).astype("int16")
            second_array = image_bytes_to_numpy(second).astype("int16")
            assert first_array.shape == second_array.shape, (
                f"{case.id}: reference and batched outputs have different shapes: "
                f"{first_array.shape} vs {second_array.shape}"
            )
            return float(abs(first_array - second_array).mean())

        def read_new_server_log(log_offset: int) -> str:
            """Read server log bytes written after the singleton references."""
            with ctx.stdout_file.open("rb") as log_file:
                log_file.seek(log_offset)
                return log_file.read().decode("utf-8", errors="ignore")

        client_count = len(requests_to_batch) * 2
        clients = [
            self._client(ctx, timeout=_DYNAMIC_BATCH_REQUEST_TIMEOUT_SECS)
            for _ in range(client_count)
        ]
        reference_clients = clients[: len(requests_to_batch)]
        batch_clients = clients[len(requests_to_batch) :]
        threads: list[threading.Thread] = []
        start_barrier: threading.Barrier | None = None

        try:
            # Singleton controls make a swapped request/output mapping observable
            # without requiring numerically identical batch and singleton kernels.
            references = [
                generate_image_response(index, params, reference_clients[index])[1]
                for index, params in enumerate(requests_to_batch)
            ]

            # Ignore singleton dispatches and only inspect the concurrent requests.
            log_offset = ctx.stdout_file.stat().st_size
            start_barrier = threading.Barrier(len(requests_to_batch))
            result_queue: queue.Queue[tuple[int, tuple[str, bytes] | BaseException]] = (
                queue.Queue(maxsize=len(requests_to_batch))
            )

            def send_request(
                index: int,
                sampling_params: DiffusionSamplingParams,
                client: openai.Client,
            ) -> None:
                try:
                    start_barrier.wait(timeout=10)
                    result_queue.put(
                        (
                            index,
                            generate_image_response(index, sampling_params, client),
                        )
                    )
                except BaseException as exc:
                    result_queue.put((index, exc))

            threads = [
                threading.Thread(
                    target=send_request,
                    args=(index, sampling_params, batch_clients[index]),
                    name=f"dynamic-batching-{case.id}-{index}",
                    daemon=True,
                )
                for index, sampling_params in enumerate(requests_to_batch)
            ]

            # Release both HTTP requests together so they share one admission window.
            for thread in threads:
                thread.start()

            # Watch the server while collecting both responses in request order.
            results: dict[int, tuple[str, bytes]] = {}
            request_deadline = time.monotonic() + _DYNAMIC_BATCH_REQUEST_TIMEOUT_SECS
            while len(results) < len(requests_to_batch):
                remaining = request_deadline - time.monotonic()
                if remaining <= 0:
                    pytest.fail(f"{case.id}: dynamic batching requests timed out")
                try:
                    index, payload = result_queue.get(
                        timeout=min(_SERVER_EXIT_POLL_INTERVAL_SECS, remaining)
                    )
                except queue.Empty:
                    self._fail_if_server_stopped_or_crashed(ctx, case.id)
                    continue
                if isinstance(payload, BaseException):
                    self._fail_if_server_stopped_or_crashed(ctx, case.id)
                    raise payload
                results[index] = payload

            ordered_results = [results[index] for index in range(len(results))]
            request_ids = [request_id for request_id, _ in ordered_results]
            assert len(set(request_ids)) == len(
                request_ids
            ), f"{case.id}: dynamic batching response ids must be distinct"
            image_hashes = {
                hashlib.sha256(image_bytes).hexdigest()
                for _, image_bytes in ordered_results
            }
            assert len(image_hashes) == len(
                ordered_results
            ), f"{case.id}: dynamic batching outputs must be distinct"

            # Each batched output must match its own prompt/seed control more closely
            # than the other request's control, catching swapped split/routing results.
            identity_distances: list[tuple[float, float]] = []
            for index, (_, image_bytes) in enumerate(ordered_results):
                own_distance = image_distance(image_bytes, references[index])
                cross_distance = image_distance(image_bytes, references[1 - index])
                identity_distances.append((own_distance, cross_distance))
                assert own_distance < cross_distance, (
                    f"{case.id}: dynamic batching response {index} maps to the wrong "
                    f"request (own MAE={own_distance:.4f}, "
                    f"cross MAE={cross_distance:.4f})"
                )

            dispatch_marker = "Dynamic batch dispatch: size=2/2, user_max=2"
            processed_marker = "Processed dynamic batch of 2/2 request(s)"
            log_deadline = time.monotonic() + _DYNAMIC_BATCH_LOG_TIMEOUT_SECS

            # Require both scheduler admission and post-forward split evidence.
            while True:
                log_slice = read_new_server_log(log_offset)
                if "Dynamic batching failed" in log_slice:
                    pytest.fail(
                        f"{case.id}: server reported a dynamic batching failure:\n"
                        f"{log_slice}"
                    )
                if dispatch_marker in log_slice and processed_marker in log_slice:
                    break
                self._fail_if_server_stopped_or_crashed(ctx, case.id)
                remaining = log_deadline - time.monotonic()
                if remaining <= 0:
                    # One final read closes the race where markers arrive after the
                    # previous poll but before the timeout boundary.
                    log_slice = read_new_server_log(log_offset)
                    if dispatch_marker in log_slice and processed_marker in log_slice:
                        break
                    missing_markers = [
                        marker
                        for marker in (dispatch_marker, processed_marker)
                        if marker not in log_slice
                    ]
                    pytest.fail(
                        f"{case.id}: missing dynamic batching server evidence: "
                        + ", ".join(missing_markers)
                    )
                time.sleep(min(0.1, remaining))

            logger.info(
                "[Dynamic batching] %s merged two requests with output hashes %s "
                "and identity distances %s",
                case.id,
                sorted(image_hashes),
                identity_distances,
            )
        finally:
            # Closing the clients interrupts in-flight HTTP reads on failure. Join
            # every worker before fixture teardown so no request can outlive the test.
            if start_barrier is not None:
                start_barrier.abort()
            close_errors: list[str] = []
            for client in clients:
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        close_errors.append(str(exc))
            for thread in threads:
                thread.join(timeout=_DYNAMIC_BATCH_THREAD_JOIN_TIMEOUT_SECS)
            live_threads = [thread.name for thread in threads if thread.is_alive()]
            assert not live_threads, (
                f"{case.id}: dynamic batching request threads did not exit after "
                f"client close: {live_threads}"
            )
            assert (
                not close_errors
            ), f"{case.id}: failed to close dynamic batching clients: {close_errors}"

    def _fail_if_server_stopped_or_crashed(
        self, ctx: ServerContext, case_id: str
    ) -> None:
        returncode = ctx.process.poll()
        if returncode is None:
            tail = ctx.log_tail()
            for pattern in _SERVER_FATAL_LOG_PATTERNS:
                if pattern in tail:
                    pytest.fail(
                        f"{case_id}: server reported a fatal backend error during "
                        f"generation: {pattern}\n\nServer log tail:\n{tail}",
                        pytrace=False,
                    )
            return

        tail = ctx.log_tail()
        message = (
            f"{case_id}: server process exited during generation "
            f"(code {returncode})."
        )
        if tail:
            message += f"\n\nServer log tail:\n{tail}"
        pytest.fail(message, pytrace=False)

    def _run_generation_with_server_watchdog(
        self,
        ctx: ServerContext,
        case_id: str,
        generate_fn: Callable[[str, openai.Client], tuple[str, bytes]],
        client: openai.Client,
    ) -> tuple[str, bytes]:
        result_queue: queue.Queue[tuple[str, tuple[str, bytes] | BaseException]] = (
            queue.Queue(maxsize=1)
        )

        def _target() -> None:
            try:
                result_queue.put(("ok", generate_fn(case_id, client)))
            except BaseException as exc:
                result_queue.put(("error", exc))

        # native backend crashes can leave the HTTP client blocked until its read
        # timeout; keep the request in a daemon thread so the main test thread can
        # fail as soon as the server subprocess exits
        thread = threading.Thread(
            target=_target,
            name=f"diffusion-generation-{case_id}",
            daemon=True,
        )
        thread.start()

        while True:
            try:
                state, payload = result_queue.get(
                    timeout=_SERVER_EXIT_POLL_INTERVAL_SECS
                )
            except queue.Empty:
                self._fail_if_server_stopped_or_crashed(ctx, case_id)
                continue

            if state == "ok":
                if isinstance(payload, BaseException):
                    raise payload
                return payload

            self._fail_if_server_stopped_or_crashed(ctx, case_id)
            if not isinstance(payload, BaseException):
                pytest.fail(f"{case_id}: invalid generation result state: {state}")
            raise payload

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
        rid, content = self._run_generation_with_server_watchdog(
            ctx, case_id, generate_fn, client
        )

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

        if (
            not missing_scenario
            and not is_baseline_generation_mode
            and scenario.estimated_full_test_time_s is None
        ):
            _MISSING_ESTIMATED_TIME_CASES.add(case.id)

        validator_name = case.server_args.custom_validator or "default"
        validator_class = VALIDATOR_REGISTRY.get(validator_name, PerformanceValidator)

        validator = validator_class(
            scenario=scenario,
            tolerances=BASELINE_CONFIG.tolerances,
            step_fractions=BASELINE_CONFIG.step_fractions,
        )

        summary = validator.collect_metrics(perf_record)
        self._print_performance_log(case, summary, scenario)

        if case.run_perf_check:
            if is_baseline_generation_mode:
                _PENDING_BASELINE_DUMPS[case.id] = (summary, missing_scenario)
                return

            if missing_scenario:
                self._dump_baseline_for_testcase(case, summary, missing_scenario)
                if missing_scenario:
                    pytest.fail(
                        f"Testcase '{case.id}' not found in {get_perf_baseline_path()}"
                    )
                return

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

    def _print_performance_log(
        self,
        case: DiffusionTestCase,
        summary: PerformanceSummary,
        scenario: ScenarioConfig | None,
    ) -> None:
        lines = [
            "",
            f"--- Performance Log: {case.id} ---",
            (
                f"  e2e={summary.e2e_ms:.2f}ms, "
                f"avg_denoise={summary.avg_denoise_ms:.2f}ms, "
                f"median_denoise={summary.median_denoise_ms:.2f}ms"
            ),
        ]
        if scenario is not None:
            lines.append(
                "  baseline: "
                f"e2e={scenario.expected_e2e_ms:.2f}ms, "
                f"avg_denoise={scenario.expected_avg_denoise_ms:.2f}ms, "
                f"median_denoise={scenario.expected_median_denoise_ms:.2f}ms"
            )
        if summary.stage_metrics:
            stages = ", ".join(
                f"{name}={duration:.2f}ms"
                for name, duration in summary.stage_metrics.items()
            )
            lines.append(f"  stages: {stages}")
        if summary.all_denoise_steps:
            # ci retries need the exact outlier, not only sampled checkpoints
            steps = ", ".join(
                f"{idx}={duration:.2f}ms"
                for idx, duration in sorted(summary.all_denoise_steps.items())
            )
            lines.append(f"  denoise_steps: {steps}")
        lines.append(f"--- End Performance Log: {case.id} ---")
        print("\n".join(lines), flush=True)

    def _dump_baseline_for_testcase(
        self,
        case: DiffusionTestCase,
        summary: PerformanceSummary,
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
{action} this baseline in the "scenarios" section of {get_perf_baseline_path()}:

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

        if case.server_args.modality == "action":
            self._validate_action_consistency(case, content)
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

Add the expected file(s) to sgl-project/ci-data-diffusion in diffusion-ci/consistency_gt/sglang_generated/ with naming (n=num_gpus).
  Image: {case.id}_{{n}}gpu.<ext> (ext from output_format: png, jpg, webp)
  Video: {case.id}_{{n}}gpu_frame_0.png, {case.id}_{{n}}gpu_frame_mid.png, {case.id}_{{n}}gpu_frame_last.png

For this case, expected file(s): {names}

Repository: https://github.com/sgl-project/ci-data-diffusion (path: diffusion-ci/consistency_gt/sglang_generated/, with optional platform subdirectories such as 5090/)
Pinned revision used by this check: {SGL_TEST_FILES_CI_DATA_REVISION}

(Optional) Per-case override in {get_consistency_threshold_path()}:
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
            output_frames = pop_realtime_key_frames(case.id)
            if output_frames is None:
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
            artifact_path = save_consistency_failure_artifact(
                artifact_dir=os.environ.get("SGLANG_DIFFUSION_ARTIFACT_DIR"),
                case_id=case.id,
                num_gpus=num_gpus,
                output_frames=output_frames,
                gt_data=gt_data,
                result=result,
                is_video=is_video,
                output_format=output_format,
                gt_remote_files=gt_remote_files,
            )
            if artifact_path is not None:
                logger.info(
                    "[Artifact] Saved consistency failure comparison: %s",
                    artifact_path,
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

    def _extract_action_array(
        self,
        payload: dict[str, Any],
        expected_horizon: int,
        expected_dim: int,
    ) -> np.ndarray:
        action = payload["data"][0]["action"]
        values = action["values"]
        assert action["shape"] == [expected_horizon, expected_dim]
        array = np.asarray(values, dtype=np.float32)
        assert array.shape == (expected_horizon, expected_dim)
        assert np.isfinite(array).all()
        return array

    def _validate_action_consistency(
        self,
        case: DiffusionTestCase,
        content: bytes,
    ) -> None:
        payload = json.loads(content.decode("utf-8"))
        expected_horizon = int(case.sampling_params.extras.get("action_horizon", 50))
        expected_dim = int(case.sampling_params.extras.get("action_dim", 32))
        output = self._extract_action_array(payload, expected_horizon, expected_dim)

        num_gpus = case.server_args.num_gpus
        if not action_gt_exists(case.id, num_gpus):
            names = ", ".join(get_action_consistency_gt_candidates(case.id, num_gpus))
            logger.error(f"""
--- MISSING ACTION GROUND TRUTH DETECTED ---
GT action JSON not found for '{case.id}'.

Add the expected file to sgl-project/ci-data-diffusion in diffusion-ci/consistency_gt/sglang_generated/ with naming:
  Action: {case.id}_{{n}}gpu.json

For this case, expected file(s): {names}

Repository: https://github.com/sgl-project/ci-data-diffusion (path: diffusion-ci/consistency_gt/sglang_generated/, with optional platform subdirectories such as 5090/)
Pinned revision used by this check: {SGL_TEST_FILES_CI_DATA_REVISION}
""")
            pytest.fail(
                f"GT action JSON not found for {case.id}. See logs for instructions to add GT."
            )

        gt_payload = load_action_consistency_gt(case.id, num_gpus)
        gt = self._extract_action_array(gt_payload, expected_horizon, expected_dim)
        abs_diff = np.abs(output - gt)
        max_abs_diff = float(abs_diff.max())
        mean_abs_diff = float(abs_diff.mean())
        max_abs_threshold = float(
            case.sampling_params.extras.get("action_max_abs_diff_threshold", 0.05)
        )
        mean_abs_threshold = float(
            case.sampling_params.extras.get("action_mean_abs_diff_threshold", 0.005)
        )

        if max_abs_diff > max_abs_threshold or mean_abs_diff > mean_abs_threshold:
            gt_remote_info = "\n".join(
                f"    - {filename}: {url}"
                for filename, url in get_action_consistency_gt_remote_files(
                    case.id,
                    num_gpus,
                )
            )
            pytest.fail(
                f"Action consistency check failed for {case.id}:\n"
                f"  max_abs_diff={max_abs_diff:.6f} "
                f"(threshold {max_abs_threshold:.6f})\n"
                f"  mean_abs_diff={mean_abs_diff:.6f} "
                f"(threshold {mean_abs_threshold:.6f})\n"
                f"  Compared GT files and links:\n{gt_remote_info}"
            )

        logger.info(
            "[Consistency] %s: PASSED action GT check "
            "(shape=%sx%s, max_abs_diff=%.6f, mean_abs_diff=%.6f)",
            case.id,
            expected_horizon,
            expected_dim,
            max_abs_diff,
            mean_abs_diff,
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

        if case.server_args.modality == "3d":
            logger.info("Skipping GT save for mesh (3d) case: %s", case.id)
            return

        if case.server_args.modality == "action":
            output_path = out_dir / f"{case.id}_{num_gpus}gpu.json"
            output_path.write_bytes(content)
            logger.info(f"Saved GT action JSON: {output_path}")
            return

        if is_video:
            # realtime consistency uses websocket raw frames to avoid lossy mp4 drift
            frames = pop_realtime_key_frames(case.id)
            if frames is None:
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

    def _validate_lora_consistency(
        self, case: DiffusionTestCase, content: bytes, operation: str
    ) -> None:
        if not case.run_consistency_check:
            logger.info(
                "[LoRA Consistency] Skipping %s consistency for %s: disabled for case",
                operation,
                case.id,
            )
            return

        logger.info(
            "[LoRA Consistency] Validating %s output for %s", operation, case.id
        )
        self._validate_consistency(case, content)

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
        client = self._client(ctx)

        # Test 1: unmerge_lora_weights - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing unmerge_lora_weights for %s", case.id)
        resp = requests.post(
            f"{base_url}/unmerge_lora_weights", timeout=_CONTROL_API_TIMEOUT_SECS
        )
        assert resp.status_code == 200, f"unmerge_lora_weights failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after unmerge for %s", case.id)
        rid_after_unmerge, _ = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid_after_unmerge is not None, "Generation after unmerge failed"
        logger.info("[LoRA E2E] Generation after unmerge succeeded")

        # Test 2: merge_lora_weights - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing merge_lora_weights for %s", case.id)
        resp = requests.post(
            f"{base_url}/merge_lora_weights", timeout=_CONTROL_API_TIMEOUT_SECS
        )
        assert resp.status_code == 200, f"merge_lora_weights failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after re-merge for %s", case.id)
        rid_after_merge, content_after_merge = (
            self._run_generation_with_server_watchdog(ctx, case.id, generate_fn, client)
        )
        assert rid_after_merge is not None, "Generation after merge failed"
        self._validate_lora_consistency(case, content_after_merge, "merge_lora_weights")
        logger.info("[LoRA E2E] Generation after merge succeeded")

        # Test 3: set_lora (re-set the same adapter) - API should succeed and generation should work
        logger.info("[LoRA E2E] Testing set_lora for %s", case.id)
        resp = requests.post(
            f"{base_url}/set_lora",
            json={"lora_nickname": "default"},
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        assert resp.status_code == 200, f"set_lora failed: {resp.text}"

        logger.info("[LoRA E2E] Verifying generation after set_lora for %s", case.id)
        rid_after_set, content_after_set = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid_after_set is not None, "Generation after set_lora failed"
        self._validate_lora_consistency(case, content_after_set, "set_lora")
        logger.info("[LoRA E2E] Generation after set_lora succeeded")

        # Test 4: list_loras - API should return the expected list of LoRA adapters
        logger.info("[LoRA E2E] Testing list_loras for %s", case.id)
        resp = requests.get(f"{base_url}/list_loras", timeout=_CONTROL_API_TIMEOUT_SECS)
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
        client = self._client(ctx)

        # Test 1: Generate with initial LoRA
        logger.info(
            "[LoRA Switch E2E] Testing generation with initial LoRA for %s", case.id
        )
        rid_initial, content_initial = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid_initial is not None, "Generation with initial LoRA failed"
        self._validate_lora_consistency(
            case, content_initial, "dynamic switch initial LoRA"
        )
        logger.info("[LoRA Switch E2E] Generation with initial LoRA succeeded")

        # Test 2: Switch to second LoRA and generate
        logger.info(
            "[LoRA Switch E2E] Switching to second LoRA adapter for %s", case.id
        )
        resp = requests.post(
            f"{base_url}/set_lora",
            json={"lora_nickname": "lora2", "lora_path": second_lora_path},
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        assert (
            resp.status_code == 200
        ), f"set_lora to second adapter failed: {resp.text}"

        logger.info(
            "[LoRA Switch E2E] Verifying generation with second LoRA for %s", case.id
        )
        rid_second, _ = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid_second is not None, "Generation with second LoRA failed"
        logger.info("[LoRA Switch E2E] Generation with second LoRA succeeded")

        # Test 3: Switch back to original LoRA and generate
        logger.info("[LoRA Switch E2E] Switching back to original LoRA for %s", case.id)
        resp = requests.post(
            f"{base_url}/set_lora",
            json={"lora_nickname": "default"},
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        assert resp.status_code == 200, f"set_lora back to default failed: {resp.text}"

        logger.info(
            "[LoRA Switch E2E] Verifying generation after switching back for %s",
            case.id,
        )
        rid_switched_back, content_switched_back = (
            self._run_generation_with_server_watchdog(ctx, case.id, generate_fn, client)
        )
        assert rid_switched_back is not None, "Generation after switching back failed"
        self._validate_lora_consistency(
            case, content_switched_back, "dynamic switch default LoRA"
        )
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
            timeout=_CONTROL_API_TIMEOUT_SECS,
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
        client = self._client(ctx)

        # Test 1: Basic multi-LoRA with list format
        resp = requests.post(
            f"{base_url}/set_lora",
            json={
                "lora_nickname": ["default", "lora2"],
                "lora_path": [first_lora_path, second_lora_path],
                "target": "all",
                "strength": [0.5, 0.5],
            },
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        assert (
            resp.status_code == 200
        ), f"set_lora with multiple adapters failed: {resp.text}"
        rid, _ = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid is not None

        # Test 2: Different strengths
        resp = requests.post(
            f"{base_url}/set_lora",
            json={
                "lora_nickname": ["default", "lora2"],
                "lora_path": [first_lora_path, second_lora_path],
                "target": "all",
                "strength": [0.6, 0.35],
            },
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        assert (
            resp.status_code == 200
        ), f"set_lora with different strengths failed: {resp.text}"
        rid, _ = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid is not None

        # Test 3: Different targets
        requests.post(
            f"{base_url}/set_lora",
            json={"lora_nickname": "default"},
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        resp = requests.post(
            f"{base_url}/set_lora",
            json={
                "lora_nickname": ["default", "lora2"],
                "lora_path": [first_lora_path, second_lora_path],
                "target": ["transformer", "transformer_2"],
                "strength": [0.6, 0.35],
            },
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        assert (
            resp.status_code == 200
        ), f"set_lora with cached adapters failed: {resp.text}"
        rid, _ = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid is not None

        # Test 4: Switch back to single LoRA
        resp = requests.post(
            f"{base_url}/set_lora",
            json={"lora_nickname": "default"},
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
        assert (
            resp.status_code == 200
        ), f"set_lora back to single adapter failed: {resp.text}"
        rid, content = self._run_generation_with_server_watchdog(
            ctx, case.id, generate_fn, client
        )
        assert rid is not None
        self._validate_lora_consistency(case, content, "multi-LoRA default adapter")

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
        resp = requests.get(f"{base_url}/v1/models", timeout=_CONTROL_API_TIMEOUT_SECS)
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
        expected_task_type = get_model_task_type_for_server_args(case.server_args).name
        assert model["task_type"] == expected_task_type, (
            f"task_type mismatch: expected {expected_task_type}, "
            f"got {model['task_type']}"
        )
        logger.info(
            "[Models API] GET /v1/models returned valid response with extended fields"
        )

        # Test GET /v1/models/{model_path}
        model_path = model["id"]
        logger.info("[Models API] Testing GET /v1/models/%s", model_path)
        resp = requests.get(
            f"{base_url}/v1/models/{model_path}", timeout=_CONTROL_API_TIMEOUT_SECS
        )
        assert resp.status_code == 200, f"/v1/models/{model_path} failed: {resp.text}"

        single_model = resp.json()
        assert single_model["id"] == model_path, "Single model ID mismatch"
        assert single_model["object"] == "model", "Single model object type mismatch"

        # Verify extended fields on single model endpoint too
        assert "num_gpus" in single_model, "Single model missing 'num_gpus' field"
        assert "task_type" in single_model, "Single model missing 'task_type' field"
        assert single_model["task_type"] == expected_task_type, (
            f"Single model task_type mismatch: expected {expected_task_type}, "
            f"got {single_model['task_type']}"
        )
        logger.info(
            "[Models API] GET /v1/models/{model_path} returned valid response with extended fields"
        )

        # Test GET /v1/models/{non_existent_model} returns 404
        logger.info("[Models API] Testing GET /v1/models/non_existent_model")
        resp = requests.get(
            f"{base_url}/v1/models/non_existent_model",
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
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
        resp = requests.get(f"{base_url}/v1/models", timeout=_CONTROL_API_TIMEOUT_SECS)
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

        resp = requests.post(
            f"{base_url}/v1/videos",
            json=payload,
            timeout=_CONTROL_API_TIMEOUT_SECS,
        )
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
        try:
            self._test_diffusion_generation_impl(case, diffusion_server)
        except pytest.skip.Exception:
            _print_case_log_separator(case.id, "SKIPPED diffusion testcase")
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            _print_case_log_separator(case.id, "FAILED diffusion testcase")
            raise
        else:
            _print_case_log_separator(case.id, "PASSED diffusion testcase")

    def _test_diffusion_generation_impl(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        task_type = get_model_task_type_for_server_args(case.server_args)
        if task_type.is_text_gen():
            # Text generation has no media payload or diffusion perf record.
            # Exercise its native OpenAI endpoint and stop before image/video checks.
            self._test_chat_completion_smoke(diffusion_server, case)
            return

        # Check if we're in GT generation mode
        is_gt_gen_mode = os.environ.get("SGLANG_GEN_GT", "0") == "1"

        # GT generation also needs the dynamic set_lora step before generation.
        if case.run_lora_dynamic_load_check:
            self._test_dynamic_lora_loading(diffusion_server, case)

        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
            run_revised_prompt_check=case.run_revised_prompt_check,
        )

        # Single generation - output is reused for both validations
        is_realtime_case = case.sampling_params.realtime_num_chunks is not None
        perf_record, content = self.run_and_collect(
            diffusion_server,
            case.id,
            generate_fn,
            collect_perf=not is_gt_gen_mode and not is_realtime_case,
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

        if is_realtime_case:
            run_case_check(
                "performance",
                lambda: validate_realtime_perf_stats(
                    case.id,
                    pop_realtime_perf_stats(case.id),
                    case.sampling_params.realtime_perf_thresholds,
                    ignore_initial_chunks=(
                        case.sampling_params.realtime_perf_ignore_initial_chunks
                    ),
                ),
            )
        else:
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

        # Dynamic batching launches concurrent HTTP requests, so run it last and
        # let any failure immediately hand control to the server fixture teardown.
        if case.dynamic_batching_requests:
            self._test_dynamic_batching_smoke(diffusion_server, case)
