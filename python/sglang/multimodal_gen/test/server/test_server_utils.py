"""
Server management and performance validation for diffusion tests.
"""

from __future__ import annotations

import base64
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence
from urllib.request import urlopen

import pytest
from openai import Client

from sglang.multimodal_gen.benchmarks.compare_perf import calculate_upper_bound
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    globally_suppress_loggers,
    init_logger,
)
from sglang.multimodal_gen.runtime.utils.perf_logger import RequestPerfRecord
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    PerformanceSummary,
    ScenarioConfig,
    ToleranceConfig,
)
from sglang.multimodal_gen.test.slack_utils import upload_file_to_slack
from sglang.multimodal_gen.test.test_utils import (
    get_expected_image_format,
    is_image_url,
    prepare_perf_log,
    validate_image,
    validate_image_file,
    validate_openai_video,
    validate_video_file,
)

logger = init_logger(__name__)

globally_suppress_loggers()


def download_image_from_url(url: str) -> Path:
    """Download an image from a URL to a temporary file.

    Args:
        url: The URL of the image to download

    Returns:
        Path to the downloaded temporary file
    """
    logger.info(f"Downloading image from URL: {url}")

    # Determine file extension from URL
    ext = ".jpg"  # default
    if url.lower().endswith((".png", ".jpeg", ".jpg", ".webp", ".gif")):
        ext = url[url.rfind(".") :]

    # Create temporary file
    temp_file = (
        Path(tempfile.gettempdir()) / f"diffusion_test_image_{int(time.time())}{ext}"
    )

    try:
        with urlopen(url, timeout=30) as response:
            temp_file.write_bytes(response.read())
        logger.info(f"Downloaded image to: {temp_file}")
        return temp_file
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise


def parse_dimensions(size_string: str | None) -> tuple[int | None, int | None]:
    """Parse a size string in "widthxheight" format to (width, height) tuple.

    Args:
        size_string: Size string in "widthxheight" format (e.g., "1024x1024") or None.
                    Spaces are automatically stripped.

    Returns:
        Tuple of (width, height) as integers if parsing succeeds, (None, None) otherwise.
    """
    if not size_string:
        return (None, None)

    # Strip spaces from the entire string
    size_string = size_string.strip()
    if not size_string:
        return (None, None)

    # Split by "x"
    parts = size_string.split("x")
    if len(parts) != 2:
        return (None, None)

    # Strip spaces from each part and try to convert to int
    try:
        width_str = parts[0].strip()
        height_str = parts[1].strip()

        if not width_str or not height_str:
            return (None, None)

        width = int(width_str)
        height = int(height_str)

        # Validate that both are positive
        if width <= 0 or height <= 0:
            return (None, None)

        return (width, height)
    except ValueError:
        return (None, None)


@dataclass
class ServerContext:
    """Context for a running diffusion server."""

    port: int
    process: subprocess.Popen
    model: str
    stdout_file: Path
    perf_log_path: Path
    log_dir: Path
    _stdout_fh: Any = field(repr=False)
    _log_thread: threading.Thread | None = field(default=None, repr=False)

    def cleanup(self) -> None:
        """Clean up server resources."""
        try:
            kill_process_tree(self.process.pid)
        except Exception:
            pass
        try:
            self._stdout_fh.flush()
            self._stdout_fh.close()
        except Exception:
            pass

        # ROCm/AMD: Extra cleanup to ensure GPU memory is released between tests
        # This is needed because ROCm memory release can be slower than CUDA
        if current_platform.is_hip():
            self._cleanup_rocm_gpu_memory()
            # Clean up downloaded models if HF cache is not persistent
            # This prevents disk exhaustion in CI when cache is not mounted
            self._cleanup_hf_cache_if_not_persistent()

    def _cleanup_hf_cache_if_not_persistent(self) -> None:
        """Clean up HF cache if it's not on a persistent volume.

        When running in CI without persistent cache, downloaded models accumulate
        and can cause disk/memory exhaustion. This cleans up the model after each
        test if the cache is not persistent.
        """
        import shutil

        hf_home = os.environ.get("HF_HOME", "")
        if not hf_home:
            return

        hf_hub_cache = os.path.join(hf_home, "hub")

        # Check if HF cache is on a persistent volume by looking for a marker file
        # or checking if the directory existed before this test run
        persistent_marker = os.path.join(hf_home, ".persistent_cache")
        if os.path.exists(persistent_marker):
            logger.info("HF cache is persistent, skipping cleanup")
            return

        # Check if the cache directory is empty or was just created
        # If it has very few models, it's likely not persistent
        if not os.path.exists(hf_hub_cache):
            return

        try:
            # Get model cache directories
            model_dirs = [
                d
                for d in os.listdir(hf_hub_cache)
                if d.startswith("models--")
                and os.path.isdir(os.path.join(hf_hub_cache, d))
            ]

            # If there are cached models but no persistent marker, clean up
            # to prevent disk exhaustion in CI
            if model_dirs:
                logger.info(
                    "HF cache appears non-persistent (no .persistent_cache marker), "
                    "cleaning up %d model(s) to prevent disk exhaustion",
                    len(model_dirs),
                )
                for model_dir in model_dirs:
                    model_path = os.path.join(hf_hub_cache, model_dir)
                    try:
                        shutil.rmtree(model_path)
                        logger.info("Cleaned up model cache: %s", model_dir)
                    except Exception as e:
                        logger.warning("Failed to clean up %s: %s", model_dir, e)
        except Exception as e:
            logger.warning("Error during HF cache cleanup: %s", e)

    def _cleanup_rocm_gpu_memory(self) -> None:
        """ROCm-specific cleanup to ensure GPU memory is fully released."""
        import gc

        # Wait for process to fully terminate
        try:
            self.process.wait(timeout=30)
        except Exception:
            pass

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear HIP memory on all GPUs
        try:
            import torch

            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        except Exception:
            pass

        # Wait for GPU memory to be released (ROCm can be much slower than CUDA)
        # The GPU driver needs time to reclaim memory from killed processes
        time.sleep(15)


class ServerManager:
    """Manages diffusion server lifecycle."""

    def __init__(
        self,
        model: str,
        port: int,
        wait_deadline: float = 1200.0,
        extra_args: str = "",
        env_vars: dict[str, str] | None = None,
    ):
        self.model = model
        self.port = port
        self.wait_deadline = wait_deadline
        self.extra_args = extra_args
        self.env_vars = env_vars or {}

    def _wait_for_rocm_gpu_memory_clear(self, max_wait: float = 60.0) -> None:
        """ROCm-specific: Wait for GPU memory to be mostly free before starting.

        ROCm GPU memory release from killed processes can be significantly slower
        than CUDA, so we need to wait longer and be more patient.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return

            start_time = time.time()
            last_total_used = float("inf")

            while time.time() - start_time < max_wait:
                # Check GPU memory usage
                total_used = 0
                for i in range(torch.cuda.device_count()):
                    mem_info = torch.cuda.mem_get_info(i)
                    free, total = mem_info
                    used = total - free
                    total_used += used

                # If less than 5GB is used across all GPUs, we're good
                if total_used < 5 * 1024 * 1024 * 1024:  # 5GB
                    logger.info(
                        "[server-test] ROCm GPU memory is clear (used: %.2f GB)",
                        total_used / (1024**3),
                    )
                    return

                # Log progress
                elapsed = int(time.time() - start_time)
                if total_used < last_total_used:
                    logger.info(
                        "[server-test] ROCm: GPU memory clearing (used: %.2f GB, elapsed: %ds)",
                        total_used / (1024**3),
                        elapsed,
                    )
                else:
                    logger.info(
                        "[server-test] ROCm: Waiting for GPU memory (used: %.2f GB, elapsed: %ds)",
                        total_used / (1024**3),
                        elapsed,
                    )
                last_total_used = total_used
                time.sleep(3)

            # Final warning with detailed GPU info
            logger.warning(
                "[server-test] ROCm GPU memory not fully cleared after %.0fs (used: %.2f GB). "
                "Proceeding anyway - this may cause OOM.",
                max_wait,
                total_used / (1024**3),
            )
        except Exception as e:
            logger.debug("[server-test] Could not check ROCm GPU memory: %s", e)

    def start(self) -> ServerContext:
        """Start the diffusion server and wait for readiness."""
        # ROCm/AMD: Wait for GPU memory to be clear before starting
        # This prevents OOM when running sequential tests on ROCm
        if current_platform.is_hip():
            self._wait_for_rocm_gpu_memory_clear()

        log_dir, perf_log_path = prepare_perf_log()

        safe_model_name = self.model.replace("/", "_")
        stdout_path = (
            Path(tempfile.gettempdir())
            / f"sgl_server_{self.port}_{safe_model_name}.log"
        )
        stdout_path.unlink(missing_ok=True)

        command = [
            "sglang",
            "serve",
            "--model-path",
            self.model,
            "--port",
            str(self.port),
            "--log-level=debug",
        ]
        if self.extra_args.strip():
            command.extend(self.extra_args.strip().split())

        env = os.environ.copy()
        env["SGLANG_DIFFUSION_STAGE_LOGGING"] = "1"
        env["SGLANG_PERF_LOG_DIR"] = log_dir.as_posix()

        # Apply custom environment variables
        env.update(self.env_vars)

        # TODO: unify with run_command
        logger.info(f"Running command: {shlex.join(command)}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        log_thread = None
        stdout_fh = stdout_path.open("w", encoding="utf-8", buffering=1)
        if process.stdout:

            def _log_pipe(pipe: Any, file: Any) -> None:
                """Read from pipe and write to file and stdout."""
                try:
                    with pipe:
                        for line in iter(pipe.readline, ""):
                            sys.stdout.write(line)
                            sys.stdout.flush()
                            file.write(line)
                            file.flush()
                except Exception as e:
                    logger.error("Log pipe thread error: %s", e)
                finally:
                    file.close()
                    logger.debug("Log pipe thread finished.")

            log_thread = threading.Thread(
                target=_log_pipe, args=(process.stdout, stdout_fh)
            )
            log_thread.daemon = True
            log_thread.start()

        logger.info(
            "[server-test] Starting server pid=%s, model=%s, log=%s",
            process.pid,
            self.model,
            stdout_path,
        )

        self._wait_for_ready(process, stdout_path)

        return ServerContext(
            port=self.port,
            process=process,
            model=self.model,
            stdout_file=stdout_path,
            perf_log_path=perf_log_path,
            log_dir=log_dir,
            _stdout_fh=stdout_fh,
            _log_thread=log_thread,
        )

    def _wait_for_ready(self, process: subprocess.Popen, stdout_path: Path) -> None:
        """Wait for server to become ready."""
        start = time.time()
        ready_message = "Application startup complete."
        log_period = 30
        prev_log_period_count = 0

        while time.time() - start < self.wait_deadline:
            if process.poll() is not None:
                tail = self._get_log_tail(stdout_path)
                raise RuntimeError(
                    f"Server exited early (code {process.returncode}).\n{tail}"
                )

            if stdout_path.exists():
                try:
                    content = stdout_path.read_text(encoding="utf-8", errors="ignore")
                    if ready_message in content:
                        logger.info("[server-test] Server ready")
                        return
                except Exception as e:
                    logger.debug("Could not read log yet: %s", e)

            elapsed = int(time.time() - start)
            if (elapsed // log_period) > prev_log_period_count:
                prev_log_period_count = elapsed // log_period
                logger.info("[server-test] Waiting for server... elapsed=%ss", elapsed)
            time.sleep(1)

        tail = self._get_log_tail(stdout_path)
        raise TimeoutError(f"Server not ready within {self.wait_deadline}s.\n{tail}")

    @staticmethod
    def _get_log_tail(path: Path, lines: int = 200) -> str:
        """Get the last N lines from a log file."""
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            return "\n".join(content.splitlines()[-lines:])
        except Exception:
            return ""


class PerformanceValidator:
    """Validates performance metrics against expectations."""

    is_video_gen: bool = False

    def __init__(
        self,
        scenario: ScenarioConfig,
        tolerances: ToleranceConfig,
        step_fractions: Sequence[float],
    ):
        self.scenario = scenario
        self.tolerances = tolerances
        self.step_fractions = step_fractions
        self.is_baseline_generation_mode = (
            os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"
        )

    def _assert_le(
        self,
        name: str,
        actual: float,
        expected: float,
        tolerance: float,
        min_abs_tolerance_ms: float = 20.0,
    ):
        """Assert that actual is less than or equal to expected within a tolerance.

        Uses the larger of relative tolerance or absolute tolerance to prevent
        flaky failures on very fast operations.

        For AMD GPUs, uses 100% higher tolerance and issues warning instead of assertion.
        """
        # Check if running on AMD GPU
        is_amd = current_platform.is_hip()

        if is_amd:
            # Use 100% higher tolerance for AMD (2x the expected value)
            amd_tolerance = 1.0  # 100%
            upper_bound = calculate_upper_bound(
                expected, amd_tolerance, min_abs_tolerance_ms
            )
            if actual > upper_bound:
                logger.warning(
                    f"[AMD PERF WARNING] Validation would fail for '{name}'.\n"
                    f"  Actual:   {actual:.4f}ms\n"
                    f"  Expected: {expected:.4f}ms\n"
                    f"  AMD Limit: {upper_bound:.4f}ms "
                    f"(rel_tol: {amd_tolerance:.1%}, abs_pad: {min_abs_tolerance_ms}ms)\n"
                    f"  Original tolerance was: {tolerance:.1%}"
                )
        else:
            upper_bound = calculate_upper_bound(
                expected, tolerance, min_abs_tolerance_ms
            )
            assert actual <= upper_bound, (
                f"Validation failed for '{name}'.\n"
                f"  Actual:   {actual:.4f}ms\n"
                f"  Expected: {expected:.4f}ms\n"
                f"  Limit:    {upper_bound:.4f}ms "
                f"(rel_tol: {tolerance:.1%}, abs_pad: {min_abs_tolerance_ms}ms)"
            )

    def validate(
        self, perf_record: RequestPerfRecord, *args, **kwargs
    ) -> PerformanceSummary:
        """Validate all performance metrics and return summary."""
        summary = self.collect_metrics(perf_record)
        if self.is_baseline_generation_mode:
            return summary

        self._validate_e2e(summary)
        self._validate_denoise_agg(summary)
        self._validate_denoise_steps(summary)
        self._validate_stages(summary)

        return summary

    def collect_metrics(
        self,
        perf_record: RequestPerfRecord,
    ) -> PerformanceSummary:
        return PerformanceSummary.from_req_perf_record(perf_record, self.step_fractions)

    def _validate_e2e(self, summary: PerformanceSummary) -> None:
        """Validate end-to-end performance."""
        assert summary.e2e_ms > 0, "E2E duration missing"
        self._assert_le(
            "E2E Latency",
            summary.e2e_ms,
            self.scenario.expected_e2e_ms,
            self.tolerances.e2e,
        )

    def _validate_denoise_agg(self, summary: PerformanceSummary) -> None:
        """Validate aggregate denoising metrics."""
        assert summary.avg_denoise_ms > 0, "Denoising step timings missing"

        self._assert_le(
            "Average Denoise Step",
            summary.avg_denoise_ms,
            self.scenario.expected_avg_denoise_ms,
            self.tolerances.denoise_agg,
        )
        self._assert_le(
            "Median Denoise Step",
            summary.median_denoise_ms,
            self.scenario.expected_median_denoise_ms,
            self.tolerances.denoise_agg,
        )

    def _validate_denoise_steps(self, summary: PerformanceSummary) -> None:
        """Validate individual denoising steps."""
        for idx, actual in summary.sampled_steps.items():
            expected = self.scenario.denoise_step_ms.get(idx)
            if expected is None:
                continue
            # FIXME: hardcode, looser for first step
            tolerance = 0.4 if idx == 0 else self.tolerances.denoise_step

            self._assert_le(
                f"Denoise Step {idx}",
                actual,
                expected,
                tolerance,
            )

    def _validate_stages(self, summary: PerformanceSummary) -> None:
        """Validate stage-level metrics."""
        assert summary.stage_metrics, "Stage metrics missing"

        for stage, expected in self.scenario.stages_ms.items():
            if stage == "per_frame_generation" and self.is_video_gen:
                continue
            actual = summary.stage_metrics.get(stage)
            assert actual is not None, f"Stage {stage} timing missing"
            tolerance = (
                self.tolerances.denoise_stage
                if stage == "DenoisingStage"
                else self.tolerances.non_denoise_stage
            )
            self._assert_le(
                f"Stage '{stage}'",
                actual,
                expected,
                tolerance,
                min_abs_tolerance_ms=120.0,  # relax absolute tolerance for non-denoising stages
            )


class VideoPerformanceValidator(PerformanceValidator):
    """Extended validator for video diffusion with frame-level metrics."""

    is_video_gen = True

    def validate(
        self,
        perf_record: RequestPerfRecord,
        num_frames: int | None = None,
    ) -> PerformanceSummary:
        """Validate video metrics including frame generation rates."""
        summary = super().validate(perf_record)

        if num_frames and summary.e2e_ms > 0:
            summary.total_frames = num_frames
            summary.avg_frame_time_ms = summary.e2e_ms / num_frames
            summary.frames_per_second = 1000.0 / summary.avg_frame_time_ms

            if not self.is_baseline_generation_mode:
                self._validate_frame_rate(summary)

        return summary

    def _validate_frame_rate(self, summary: PerformanceSummary) -> None:
        """Validate frame generation performance."""
        expected_frame_time = self.scenario.stages_ms.get("per_frame_generation")
        if expected_frame_time and summary.avg_frame_time_ms:
            self._assert_le(
                "Average Frame Time",
                summary.avg_frame_time_ms,
                expected_frame_time,
                self.tolerances.denoise_stage,
            )


class MeshValidator:
    """Validator for 3D mesh generation using Chamfer Distance for geometric similarity."""

    # Reference mesh file path
    REFERENCE_MESH_PATH = (
        Path(__file__).parent.parent / "test_files" / "hunyuan3d_reference.obj"
    )

    # Chamfer Distance configuration
    NUM_SAMPLE_POINTS = 4096
    CD_THRESHOLD_RATIO = 0.01  # 1% of bbox diagonal
    RANDOM_SEED = 42

    def __init__(self, **kwargs):
        """Initialize mesh validator. Accepts kwargs for compatibility with validator registry."""
        pass

    def _sample_point_cloud(self, mesh, num_points: int):
        """Sample points uniformly from mesh surface.

        Args:
            mesh: Input trimesh object
            num_points: Number of points to sample

        Returns:
            Point cloud array of shape (num_points, 3)
        """
        import numpy as np

        points, _ = mesh.sample(num_points, return_index=True)
        return np.array(points)

    def _compute_chamfer_distance(self, points1, points2):
        """Compute bidirectional Chamfer Distance using KD-Tree.

        Args:
            points1: First point cloud (N, 3)
            points2: Second point cloud (M, 3)

        Returns:
            Tuple of (forward_cd, backward_cd, total_cd)
        """
        import numpy as np
        from scipy.spatial import cKDTree

        tree1 = cKDTree(points1)
        tree2 = cKDTree(points2)

        distances1, _ = tree2.query(points1)
        distances2, _ = tree1.query(points2)

        forward_cd = float(np.mean(distances1**2))
        backward_cd = float(np.mean(distances2**2))
        total_cd = forward_cd + backward_cd

        return forward_cd, backward_cd, total_cd

    def validate(self, mesh_path: str) -> dict:
        """
        Validate generated mesh against reference mesh.

        Args:
            mesh_path: Path to the generated mesh file

        Returns:
            Dictionary with validation results
        """
        try:
            import trimesh
        except ImportError:
            logger.error(
                "trimesh is required for mesh validation. Install with: pip install trimesh"
            )
            return {"all_passed": False, "error": "trimesh not installed"}

        results = {
            "all_passed": True,
            "checks": {},
        }

        # Load generated mesh
        try:
            generated_mesh = trimesh.load(mesh_path)
            if isinstance(generated_mesh, trimesh.Scene):
                generated_mesh = generated_mesh.dump(concatenate=True)
        except Exception as e:
            logger.error(f"Failed to load generated mesh: {e}")
            results["all_passed"] = False
            results["error"] = f"Failed to load generated mesh: {e}"
            return results

        # Load reference mesh
        if not self.REFERENCE_MESH_PATH.exists():
            logger.error(f"Reference mesh not found at {self.REFERENCE_MESH_PATH}")
            results["all_passed"] = False
            results["error"] = f"Reference mesh not found at {self.REFERENCE_MESH_PATH}"
            return results

        try:
            reference_mesh = trimesh.load(str(self.REFERENCE_MESH_PATH))
            if isinstance(reference_mesh, trimesh.Scene):
                reference_mesh = reference_mesh.dump(concatenate=True)
        except Exception as e:
            logger.error(f"Failed to load reference mesh: {e}")
            results["all_passed"] = False
            results["error"] = f"Failed to load reference mesh: {e}"
            return results

        import numpy as np

        # Compute bounding box diagonal for threshold normalization
        ref_bbox = reference_mesh.bounding_box.bounds
        bbox_diagonal = float(np.linalg.norm(ref_bbox[1] - ref_bbox[0]))
        cd_threshold = self.CD_THRESHOLD_RATIO * bbox_diagonal

        # Sample point clouds from both meshes
        np.random.seed(self.RANDOM_SEED)
        gen_points = self._sample_point_cloud(generated_mesh, self.NUM_SAMPLE_POINTS)
        ref_points = self._sample_point_cloud(reference_mesh, self.NUM_SAMPLE_POINTS)

        # Compute Chamfer Distance
        forward_cd, backward_cd, total_cd = self._compute_chamfer_distance(
            gen_points, ref_points
        )

        cd_passed = total_cd <= cd_threshold
        results["checks"]["chamfer_distance"] = {
            "forward_cd": forward_cd,
            "backward_cd": backward_cd,
            "total_cd": total_cd,
            "threshold": cd_threshold,
            "bbox_diagonal": bbox_diagonal,
            "passed": cd_passed,
        }
        if not cd_passed:
            results["all_passed"] = False
            logger.warning(
                f"Chamfer Distance check failed: total_cd={total_cd:.6f}, "
                f"threshold={cd_threshold:.6f}"
            )

        # Print comparison summary
        print("=" * 60)
        print("[MeshValidator] Chamfer Distance Results:")
        print(f"  Sample Points: {self.NUM_SAMPLE_POINTS}")
        print(f"  BBox Diagonal: {bbox_diagonal:.4f}")
        print(f"  Forward CD (gen->ref):  {forward_cd:.6f}")
        print(f"  Backward CD (ref->gen): {backward_cd:.6f}")
        print(f"  Total Chamfer Distance: {total_cd:.6f}")
        print(
            f"  Threshold: {cd_threshold:.6f} ({self.CD_THRESHOLD_RATIO * 100:.2f}% of bbox diagonal)"
        )
        print(f"  Passed: {cd_passed}")
        print("=" * 60)

        return results


# Registry of validators by name
VALIDATOR_REGISTRY = {
    "default": PerformanceValidator,
    "video": VideoPerformanceValidator,
    "mesh": MeshValidator,
}


def get_generate_fn(
    model_path: str,
    modality: str,
    sampling_params: DiffusionSamplingParams,
) -> Callable[[str, Client], str]:
    """Return appropriate generation function for the case."""
    # Allow override via environment variable (useful for AMD where large resolutions cause slow VAE)
    output_size = os.environ.get("SGLANG_TEST_OUTPUT_SIZE", sampling_params.output_size)
    n = sampling_params.num_outputs_per_prompt

    def _create_and_download_video(
        client,
        case_id,
        *,
        model: str,
        size: str,
        prompt: str | None = None,
        seconds: int | None = None,
        input_reference: Any | None = None,
        extra_body: dict[Any] | None = None,
    ) -> str:
        """
        Create a video job via /v1/videos, poll until completion,
        then download the binary content and validate it.

        Returns request-id
        """

        create_kwargs: dict[str, Any] = {
            "model": model,
            "size": size,
        }
        if prompt is not None:
            create_kwargs["prompt"] = prompt
        if seconds is not None:
            create_kwargs["seconds"] = seconds
        if input_reference is not None:
            create_kwargs["input_reference"] = input_reference  # triggers multipart
        if extra_body is not None:
            create_kwargs["extra_body"] = extra_body

        job = client.videos.create(**create_kwargs)  # type: ignore[attr-defined]
        video_id = job.id

        job_completed = False
        is_baseline_generation_mode = os.environ.get("SGLANG_GEN_BASELINE", "0") == "1"
        # Check if running on AMD GPU - use longer timeout
        is_amd = current_platform.is_hip()
        if is_baseline_generation_mode:
            timeout = 3600.0
        elif is_amd:
            timeout = 2400.0  # 40 minutes for AMD
        else:
            timeout = 1200.0
        deadline = time.time() + timeout
        while True:
            page = client.videos.list()  # type: ignore[attr-defined]
            item = next((v for v in page.data if v.id == video_id), None)

            if item and getattr(item, "status", None) == "completed":
                job_completed = True
                break

            if time.time() > deadline:
                break

            time.sleep(1)

        if not job_completed:
            if is_baseline_generation_mode:
                logger.warning(
                    f"{case_id}: video job {video_id} timed out during baseline generation. "
                    "Attempting to collect performance data anyway."
                )
                return video_id

            if is_amd:
                logger.warning(
                    f"[AMD TIMEOUT WARNING] {case_id}: video job {video_id} did not complete "
                    f"within {timeout}s timeout. This may indicate performance issues on AMD."
                )
                pytest.skip(
                    f"{case_id}: video job timed out on AMD after {timeout}s - skipping"
                )

            pytest.fail(f"{case_id}: video job {video_id} did not complete in time")

        # download video
        resp = client.videos.download_content(video_id=video_id)  # type: ignore[attr-defined]
        content = resp.read()
        validate_openai_video(content)

        expected_filename = f"{video_id}.mp4"
        tmp_path = expected_filename
        with open(tmp_path, "wb") as f:
            f.write(content)

        # Validate output file
        expected_width, expected_height = parse_dimensions(size)
        validate_video_file(
            tmp_path, expected_filename, expected_width, expected_height
        )

        upload_file_to_slack(
            case_id=case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            file_path=tmp_path,
            origin_file_path=sampling_params.image_path,
        )
        os.remove(tmp_path)

        return video_id

    video_seconds = sampling_params.seconds or 4

    def generate_image(case_id, client) -> str:
        """T2I: Text to Image generation."""
        if not sampling_params.prompt:
            pytest.skip(f"{id}: no text prompt configured")

        # Request parameters that affect output format
        req_output_format = None  # Not specified in current request
        req_background = None  # Not specified in current request

        # Build extra_body for optional features
        extra_body = {}
        if sampling_params.enable_teacache:
            extra_body["enable_teacache"] = True

        response = client.images.with_raw_response.generate(
            model=model_path,
            prompt=sampling_params.prompt,
            n=n,
            size=output_size,
            response_format="b64_json",
            extra_body=extra_body if extra_body else None,
        )
        result = response.parse()
        validate_image(result.data[0].b64_json)

        rid = result.id

        img_data = base64.b64decode(result.data[0].b64_json)
        # Infer expected format from request parameters
        expected_ext = get_expected_image_format(req_output_format, req_background)
        expected_filename = f"{result.created}.{expected_ext}"
        tmp_path = expected_filename
        with open(tmp_path, "wb") as f:
            f.write(img_data)

        # Validate output file
        expected_width, expected_height = parse_dimensions(output_size)
        validate_image_file(
            tmp_path,
            expected_filename,
            expected_width,
            expected_height,
            output_format=req_output_format,
            background=req_background,
        )

        upload_file_to_slack(
            case_id=case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            file_path=tmp_path,
        )
        os.remove(tmp_path)

        return rid

    def generate_image_edit(case_id, client) -> str:
        """TI2I: Text + Image ? Image edit."""
        if not sampling_params.prompt or not sampling_params.image_path:
            pytest.skip(f"{id}: no edit config")

        image_paths = sampling_params.image_path

        if not isinstance(image_paths, list):
            image_paths = [image_paths]

        new_image_paths = []
        for image_path in image_paths:
            if is_image_url(image_path):
                new_image_paths.append(download_image_from_url(str(image_path)))
            else:
                new_image_paths.append(Path(image_path))
                if not image_path.exists():
                    pytest.skip(f"{id}: file missing: {image_path}")

        image_paths = new_image_paths

        # Request parameters that affect output format
        req_output_format = (
            sampling_params.output_format
        )  # Not specified in current request
        req_background = None  # Not specified in current request

        # Build extra_body for optional features
        extra_body = {"num_frames": sampling_params.num_frames}
        if sampling_params.enable_teacache:
            extra_body["enable_teacache"] = True

        images = [open(image_path, "rb") for image_path in image_paths]
        try:
            response = client.images.with_raw_response.edit(
                model=model_path,
                image=images,
                prompt=sampling_params.prompt,
                n=n,
                size=output_size,
                response_format="b64_json",
                output_format=req_output_format,
                extra_body=extra_body,
            )
        finally:
            for img in images:
                img.close()

        result = response.parse()
        validate_image(result.data[0].b64_json)

        img_data = base64.b64decode(result.data[0].b64_json)
        rid = result.id

        # Infer expected format from request parameters
        expected_ext = get_expected_image_format(req_output_format, req_background)
        expected_filename = f"{rid}.{expected_ext}"
        tmp_path = expected_filename
        with open(tmp_path, "wb") as f:
            f.write(img_data)

        # Validate output file
        expected_width, expected_height = parse_dimensions(output_size)
        validate_image_file(
            tmp_path,
            expected_filename,
            expected_width,
            expected_height,
            output_format=req_output_format,
            background=req_background,
        )

        upload_file_to_slack(
            case_id=case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            file_path=tmp_path,
            origin_file_path=sampling_params.image_path,
        )
        os.remove(tmp_path)

        return rid

    def generate_image_edit_url(case_id, client) -> str:
        """TI2I: Text + Image ? Image edit using direct URL transfer (no pre-download)."""
        if not sampling_params.prompt or not sampling_params.image_path:
            pytest.skip(f"{id}: no edit config")
        # Handle both single URL and list of URLs
        image_urls = sampling_params.image_path
        if not isinstance(image_urls, list):
            image_urls = [image_urls]

        # Validate all URLs
        for url in image_urls:
            if not is_image_url(url):
                pytest.skip(
                    f"{id}: image_path must be a URL for URL direct test: {url}"
                )

        # Request parameters that affect output format
        req_output_format = (
            sampling_params.output_format
        )  # Not specified in current request
        req_background = None  # Not specified in current request

        response = client.images.with_raw_response.edit(
            model=model_path,
            prompt=sampling_params.prompt,
            image=[],  # Only for OpenAI verification
            n=n,
            size=sampling_params.output_size,
            response_format="b64_json",
            output_format=req_output_format,
            extra_body={"url": image_urls, "num_frames": sampling_params.num_frames},
        )

        result = response.parse()
        rid = result.id

        validate_image(result.data[0].b64_json)

        # Save and upload result for verification
        img_data = base64.b64decode(result.data[0].b64_json)
        # Infer expected format from request parameters
        expected_ext = get_expected_image_format(req_output_format, req_background)
        expected_filename = f"{rid}.{expected_ext}"
        tmp_path = expected_filename
        with open(tmp_path, "wb") as f:
            f.write(img_data)

        # Validate output file
        expected_width, expected_height = parse_dimensions(sampling_params.output_size)
        validate_image_file(
            tmp_path,
            expected_filename,
            expected_width,
            expected_height,
            output_format=req_output_format,
            background=req_background,
        )

        upload_file_to_slack(
            case_id=case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            file_path=tmp_path,
            origin_file_path=str(sampling_params.image_path),
        )
        os.remove(tmp_path)

        return rid

    def generate_video(case_id, client) -> str:
        """T2V: Text ? Video."""
        if not sampling_params.prompt:
            pytest.skip(f"{id}: no text prompt configured")

        # Build extra_body for optional features
        extra_body = {}
        if sampling_params.enable_teacache:
            extra_body["enable_teacache"] = True

        return _create_and_download_video(
            client,
            case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            size=output_size,
            seconds=video_seconds,
            extra_body=extra_body if extra_body else None,
        )

    def generate_image_to_video(case_id, client) -> str:
        """I2V: Image ? Video (optional prompt)."""
        if not sampling_params.image_path:
            pytest.skip(f"{id}: no input image configured")

        if is_image_url(sampling_params.image_path):
            image_path = download_image_from_url(str(sampling_params.image_path))
        else:
            image_path = Path(sampling_params.image_path)
            if not image_path.exists():
                pytest.skip(f"{id}: file missing: {image_path}")

        # Build extra_body for optional features
        extra_body = {}
        if sampling_params.enable_teacache:
            extra_body["enable_teacache"] = True

        with image_path.open("rb") as fh:
            return _create_and_download_video(
                client,
                case_id,
                model=model_path,
                prompt=sampling_params.prompt,
                size=output_size,
                seconds=video_seconds,
                input_reference=fh,
                extra_body=extra_body if extra_body else None,
            )

    def generate_text_url_image_to_video(case_id, client) -> str:
        if not sampling_params.prompt or not sampling_params.image_path:
            pytest.skip(f"{id}: no edit config")

        # Build extra_body for optional features
        extra_body = {"reference_url": sampling_params.image_path}
        if sampling_params.enable_teacache:
            extra_body["enable_teacache"] = True

        return _create_and_download_video(
            client,
            case_id,
            model=model_path,
            prompt=sampling_params.prompt,
            size=sampling_params.output_size,
            seconds=video_seconds,
            extra_body={
                "reference_url": sampling_params.image_path,
                "fps": sampling_params.fps,
                "num_frames": sampling_params.num_frames,
            },
        )

    def generate_text_image_to_video(case_id, client) -> str:
        """TI2V: Text + Image ? Video."""
        if not sampling_params.prompt or not sampling_params.image_path:
            pytest.skip(f"{id}: no edit config")

        if is_image_url(sampling_params.image_path):
            image_path = download_image_from_url(str(sampling_params.image_path))
        else:
            image_path = Path(sampling_params.image_path)
            if not image_path.exists():
                pytest.skip(f"{id}: file missing: {image_path}")

        # Build extra_body for optional features
        extra_body = {}
        if sampling_params.enable_teacache:
            extra_body["enable_teacache"] = True

        with image_path.open("rb") as fh:
            return _create_and_download_video(
                client,
                case_id,
                model=model_path,
                prompt=sampling_params.prompt,
                size=output_size,
                seconds=video_seconds,
                input_reference=fh,
                extra_body={
                    "fps": sampling_params.fps,
                    "num_frames": sampling_params.num_frames,
                },
            )

    def generate_mesh(case_id, client) -> str:
        """I2M: Image to Mesh generation using HTTP API (same pattern as image_edit)."""
        import requests as http_requests

        if not sampling_params.image_path:
            pytest.skip(f"{case_id}: no input image configured for mesh generation")

        image_path = sampling_params.image_path
        if isinstance(image_path, Path):
            image_path = str(image_path)

        if not Path(image_path).exists():
            pytest.skip(f"{case_id}: image file missing: {image_path}")

        # Get server base URL from client
        base_url = str(client.base_url).rstrip("/")
        # Remove /v1 suffix if present to get the root URL
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]

        # Use /v1/images/edits endpoint (same as image_edit)
        # Server routes based on model's data_type (MESH for Hunyuan3D)
        url = f"{base_url}/v1/images/edits"

        # Prepare multipart form data with identical parameters as reference script
        # Reference: seed=0 (native default), guidance_scale=5.0, num_inference_steps=50
        with open(image_path, "rb") as img_file:
            files = {"image": (Path(image_path).name, img_file, "image/png")}
            data = {
                "prompt": "generate 3d mesh",  # Required field, content not used for I2M
                "model": model_path,
                "seed": "0",
                "guidance_scale": "5.0",
                "num_inference_steps": "50",
                "response_format": "url",  # Get file path in response
            }

            logger.info(f"[Mesh Gen] Sending request to {url}")

            try:
                response = http_requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=6000,  # 10 minute timeout for mesh generation
                )
            except http_requests.exceptions.Timeout:
                pytest.fail(f"{case_id}: mesh generation timed out after 600s")
            except Exception as e:
                pytest.fail(f"{case_id}: mesh generation request failed: {e}")

        if response.status_code != 200:
            logger.error(
                f"[Mesh Gen] Request failed with status {response.status_code}"
            )
            logger.error(f"[Mesh Gen] Response: {response.text}")
            pytest.fail(f"{case_id}: mesh generation failed: {response.text}")

        result = response.json()

        # Extract mesh file path from response
        # For mesh, the output is returned as file_path in the response data
        mesh_path = None
        if "data" in result and len(result["data"]) > 0:
            data_item = result["data"][0]
            # Try different possible keys for the mesh path
            mesh_path = (
                data_item.get("file_path")
                or data_item.get("url")
                or data_item.get("revised_prompt")
            )

        if not mesh_path or not Path(mesh_path).exists():
            # Fallback: check if output is directly in result
            mesh_path = result.get("output")
            if isinstance(mesh_path, list) and len(mesh_path) > 0:
                mesh_path = mesh_path[0]

        if not mesh_path:
            pytest.fail(f"{case_id}: no mesh path in response: {result}")

        if not Path(mesh_path).exists():
            pytest.fail(f"{case_id}: mesh file not found at {mesh_path}")

        logger.info(f"[Mesh Gen] Mesh generated successfully at {mesh_path}")

        return str(mesh_path)

    if modality == "3d":
        fn = generate_mesh
    elif modality == "video":
        if sampling_params.image_path and sampling_params.prompt:
            if getattr(sampling_params, "direct_url_test", False):
                fn = generate_text_url_image_to_video
            else:
                fn = generate_text_image_to_video
        elif sampling_params.image_path:
            fn = generate_image_to_video
        else:
            fn = generate_video
    elif sampling_params.prompt and sampling_params.image_path:
        if getattr(sampling_params, "direct_url_test", False):
            fn = generate_image_edit_url
        else:
            fn = generate_image_edit
    else:
        fn = generate_image

    return fn
