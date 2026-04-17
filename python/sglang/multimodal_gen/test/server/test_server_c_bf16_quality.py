"""
BF16 quality checks for B200 ModelOpt diffusion CI cases.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.gpu_cases import ONE_GPU_CASES_C
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    ServerManager,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionServerArgs,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import (
    compute_clip_embedding,
    compute_mean_abs_diff,
    compute_psnr,
    compute_ssim,
    extract_key_frames_from_video,
    find_free_port,
    image_bytes_to_numpy,
)

logger = init_logger(__name__)

BF16_QUALITY_CASES = [case for case in ONE_GPU_CASES_C if case.run_bf16_quality_check]


def _build_server_extra_args(server_args: DiffusionServerArgs) -> str:
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
    if server_args.lora_path:
        extra_args += f" --lora-path {server_args.lora_path}"
    if server_args.enable_warmup:
        extra_args += " --warmup"

    extra_args += " --strict-ports"

    for arg in server_args.extras:
        extra_args += f" {arg}"

    return extra_args


def _build_server_env_vars(server_args: DiffusionServerArgs) -> dict[str, str]:
    env_vars = {}
    if server_args.enable_cache_dit:
        env_vars["SGLANG_CACHE_DIT_ENABLED"] = "true"
    env_vars.update(server_args.env_vars)
    return env_vars


def _safe_case_id(case_id: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in case_id)


def _save_quality_artifact(
    case: DiffusionTestCase,
    content: bytes,
    label: str,
) -> None:
    artifact_dir = os.environ.get("SGLANG_DIFFUSION_ARTIFACT_DIR")
    if not artifact_dir or not content:
        return

    suffix = "mp4" if case.server_args.modality == "video" else "png"
    dst_dir = Path(artifact_dir) / _safe_case_id(case.id)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{case.id}_{label}.{suffix}"
    dst.write_bytes(content)
    logger.info("[BF16 Quality] Preserved generated artifact: %s", dst)


def _content_to_quality_frames(
    case: DiffusionTestCase,
    content: bytes,
) -> list[Any]:
    if not content:
        raise AssertionError(f"{case.id}: content is empty")

    if case.server_args.modality == "video":
        return extract_key_frames_from_video(
            content, num_frames=case.sampling_params.num_frames
        )
    return [image_bytes_to_numpy(content)]


def _compare_bf16_quality(
    case: DiffusionTestCase,
    *,
    candidate_content: bytes,
    reference_content: bytes,
) -> None:
    thresholds = case.bf16_quality_thresholds
    if thresholds is None:
        raise AssertionError(f"{case.id}: missing BF16 quality thresholds")

    candidate_frames = _content_to_quality_frames(case, candidate_content)
    reference_frames = _content_to_quality_frames(case, reference_content)
    if len(candidate_frames) != len(reference_frames):
        raise AssertionError(
            f"{case.id}: BF16 quality frame count mismatch: "
            f"candidate={len(candidate_frames)}, reference={len(reference_frames)}"
        )

    failed_frames = []
    min_clip = float("inf")
    min_ssim = float("inf")
    min_psnr = float("inf")
    max_mean_abs_diff = 0.0

    for frame_idx, (candidate_frame, reference_frame) in enumerate(
        zip(candidate_frames, reference_frames)
    ):
        candidate_shape = getattr(candidate_frame, "shape", None)
        reference_shape = getattr(reference_frame, "shape", None)
        if candidate_shape != reference_shape:
            raise AssertionError(
                f"{case.id}: BF16 quality frame shape mismatch at frame "
                f"{frame_idx}: candidate={candidate_shape}, reference={reference_shape}"
            )

        candidate_embedding = compute_clip_embedding(candidate_frame)
        reference_embedding = compute_clip_embedding(reference_frame)
        clip_similarity = float(candidate_embedding.dot(reference_embedding))
        ssim = compute_ssim(candidate_frame, reference_frame)
        psnr = compute_psnr(candidate_frame, reference_frame)
        mean_abs_diff = compute_mean_abs_diff(candidate_frame, reference_frame)

        min_clip = min(min_clip, clip_similarity)
        min_ssim = min(min_ssim, ssim)
        min_psnr = min(min_psnr, psnr)
        max_mean_abs_diff = max(max_mean_abs_diff, mean_abs_diff)

        failed_metrics = []
        if clip_similarity < thresholds.clip_threshold:
            failed_metrics.append("clip")
        if ssim < thresholds.ssim_threshold:
            failed_metrics.append("ssim")
        if psnr < thresholds.psnr_threshold:
            failed_metrics.append("psnr")
        if mean_abs_diff > thresholds.mean_abs_diff_threshold:
            failed_metrics.append("mean_abs_diff")
        if failed_metrics:
            failed_frames.append(
                f"    - f{frame_idx} [{', '.join(failed_metrics)}] "
                f"clip={clip_similarity:.4f} "
                f"ssim={ssim:.4f} "
                f"psnr={psnr:.4f} "
                f"mean_abs_diff={mean_abs_diff:.4f}"
            )

    if failed_frames:
        pytest.fail(
            f"BF16 quality check failed for {case.id}:\n"
            f"  Metrics: clip={min_clip:.4f}, "
            f"ssim={min_ssim:.4f}, "
            f"psnr={min_psnr:.4f}, "
            f"mean_abs_diff={max_mean_abs_diff:.4f}\n"
            f"  Thresholds: clip>={thresholds.clip_threshold}, "
            f"ssim>={thresholds.ssim_threshold}, "
            f"psnr>={thresholds.psnr_threshold}, "
            f"mean_abs_diff<={thresholds.mean_abs_diff_threshold}\n"
            f"  Failed frames:\n" + "\n".join(failed_frames)
        )

    logger.info(
        f"[BF16 Quality] {case.id}: PASSED "
        f"(min_clip={min_clip:.4f}, "
        f"min_ssim={min_ssim:.4f}, "
        f"min_psnr={min_psnr:.4f}, "
        f"max_mean_abs_diff={max_mean_abs_diff:.4f})"
    )


class TestDiffusionServerOneGpuB200Bf16Quality(DiffusionServerBase):
    """BF16 reference quality checks for 1-GPU B200 ModelOpt diffusion cases."""

    @pytest.fixture(params=BF16_QUALITY_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each 1-GPU B200 quality test."""
        return request.param

    def test_bf16_quality(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,  # noqa: F811
    ) -> None:
        if os.environ.get("SGLANG_SKIP_BF16_QUALITY", "0") == "1":
            pytest.skip("SGLANG_SKIP_BF16_QUALITY=1")

        candidate_generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )
        _, candidate_content = self.run_and_collect(
            diffusion_server,
            f"{case.id}_quant",
            candidate_generate_fn,
            collect_perf=False,
        )
        _save_quality_artifact(case, candidate_content, "quant")

        logger.info(
            "[BF16 Quality] Cleaning up candidate server for %s before starting "
            "the BF16 reference server",
            case.id,
        )
        diffusion_server.cleanup()

        reference_server_args = replace(
            case.server_args,
            extras=[],
            env_vars={},
            enable_warmup=False,
        )
        reference_port = find_free_port()
        reference_manager = ServerManager(
            model=reference_server_args.model_path,
            port=reference_port,
            wait_deadline=float(os.environ.get("SGLANG_TEST_WAIT_SECS", "1200")),
            extra_args=_build_server_extra_args(reference_server_args),
            env_vars=_build_server_env_vars(reference_server_args),
        )

        logger.info(
            "[BF16 Quality] Starting BF16 reference server for %s on port %s",
            case.id,
            reference_port,
        )
        reference_ctx = reference_manager.start()
        try:
            reference_generate_fn = get_generate_fn(
                model_path=reference_server_args.model_path,
                modality=reference_server_args.modality,
                sampling_params=case.sampling_params,
            )
            _, reference_content = self.run_and_collect(
                reference_ctx,
                f"{case.id}_bf16_reference",
                reference_generate_fn,
                collect_perf=False,
            )
            _save_quality_artifact(case, reference_content, "bf16_reference")
        finally:
            reference_ctx.cleanup()

        _compare_bf16_quality(
            case,
            candidate_content=candidate_content,
            reference_content=reference_content,
        )
