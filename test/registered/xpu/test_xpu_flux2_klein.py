"""FLUX.2-klein on Intel XPU (manual, 1 GPU), 4B and 9B, t2i and ti2i.

The t2i cases mirror ``test_xpu_flux2_dev.py``: one fixed prompt, gated on a
CLIP score. The ti2i cases mirror CUDA's ``flux_2_ti2i``: the output is compared
against a stored reference image (CLIP similarity, SSIM, PSNR, mean abs diff).
There is no XPU reference set in sgl-project/ci-data-diffusion, so the
references are read from ``SGLANG_CONSISTENCY_GT_DIR``; a run without one saves
its output under ``$SGLANG_DIFFUSION_ARTIFACT_DIR/missing_consistency_gt/``.
"""

from __future__ import annotations

import io
import logging
import os

import pytest
import torch

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
    TI2I_sampling_params,
)
from sglang.multimodal_gen.test.test_utils import (
    ConsistencyThresholds,
    compare_with_gt,
    gt_exists,
    image_bytes_to_numpy,
    load_consistency_gt,
    save_missing_consistency_gt_artifact,
)
from sglang.test.ci.ci_register import register_xpu_ci

logger = logging.getLogger(__name__)

register_xpu_ci(
    est_time=2400,
    suite="nightly-xpu-1-gpu",
    nightly=True,
    disabled="Manual test, not part of the Intel nightly. Run by hand with pytest.",
)

# VBench all_dimension.txt row 691, the prompt CLIP_SCORE_THRESHOLD was measured on.
KLEIN_PROMPT = "A koala bear playing piano in the forest."


# CUDA flux_2_ti2i's edit, at klein's native size and a pinned seed so the
# stored reference is reproducible.
KLEIN_TI2I_SAMPLING_PARAMS = DiffusionSamplingParams(
    prompt=TI2I_sampling_params.prompt,
    image_path=TI2I_sampling_params.image_path,
    output_size="1024x1024",
    extras={"seed": 0},
)
KLEIN_4B_RESIDENCY = "dit=resident"
# A resident 16.91 GiB DiT leaves 6.84 GiB on a 24 GiB card, which
# 1024x1024 warmup exhausts; the encoder must stream layer by layer.
KLEIN_9B_RESIDENCY = "dit=resident,text_encoder=layerwise-offload"


# Klein is step-distilled, so 4 steps / guidance 1.0 are the defaults.
def _klein_extras(residency: str) -> list[str]:
    return [
        "--dit-precision",
        "bf16",
        "--vae-precision",
        "bf16",
        "--text-encoder-precisions",
        "bf16",
        "--component-residency",
        residency,
    ]


def _klein_case(
    case_id: str, model_path: str, residency: str, sampling_params
) -> DiffusionTestCase:
    return DiffusionTestCase(
        case_id,
        DiffusionServerArgs(
            model_path=model_path,
            modality="image",
            num_gpus=1,
            tp_size=1,
            extras=_klein_extras(residency),
        ),
        sampling_params,
        # XPU has no perf baseline or ci-data reference set; the CLIP score
        # and the local-reference comparison below are the accuracy checks.
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    )


KLEIN_T2I_SAMPLING_PARAMS = DiffusionSamplingParams(
    prompt=KLEIN_PROMPT,
    output_size="1024x1024",
)

XPU_FLUX2_KLEIN_CASES = [
    _klein_case(
        "flux2_klein_4b_image_t2i",
        "black-forest-labs/FLUX.2-klein-4B",
        KLEIN_4B_RESIDENCY,
        KLEIN_T2I_SAMPLING_PARAMS,
    ),
    _klein_case(
        "flux2_klein_9b_image_t2i",
        "black-forest-labs/FLUX.2-klein-9B",
        KLEIN_9B_RESIDENCY,
        KLEIN_T2I_SAMPLING_PARAMS,
    ),
    _klein_case(
        "flux2_klein_4b_image_ti2i",
        "black-forest-labs/FLUX.2-klein-4B",
        KLEIN_4B_RESIDENCY,
        KLEIN_TI2I_SAMPLING_PARAMS,
    ),
    _klein_case(
        "flux2_klein_9b_image_ti2i",
        "black-forest-labs/FLUX.2-klein-9B",
        KLEIN_9B_RESIDENCY,
        KLEIN_TI2I_SAMPLING_PARAMS,
    ),
]

# Measured on KLEIN_PROMPT: healthy 4B/9B score >= 0.4198,
# black/white/noise frames <= 0.1842.
CLIP_SCORE_THRESHOLD = 0.25

# CUDA flux_2_ti2i's H100 values. On a B60, a rerun scores clip 1.0000,
# ssim 0.9999, psnr >= 62.2, mad <= 0.02; a swapped input image scores
# clip <= 0.486, ssim <= 0.617, psnr <= 10.6, mad >= 56.5.
KLEIN_TI2I_THRESHOLDS = ConsistencyThresholds(
    clip_threshold=0.97,
    ssim_threshold=0.88,
    psnr_threshold=19.5,
    mean_abs_diff_threshold=13.5,
)

ARTIFACT_DIR = os.environ.get(
    "SGLANG_DIFFUSION_ARTIFACT_DIR", "/tmp/diffusion-artifacts"
)


def _save_image_and_write_summary(
    case_id: str, prompt: str, image_bytes: bytes, clip_score: float | None = None
):
    ext = "jpg" if image_bytes[:2] == b"\xff\xd8" else "png"
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    img_path = os.path.join(ARTIFACT_DIR, f"{case_id}.{ext}")
    with open(img_path, "wb") as f:
        f.write(image_bytes)
    logger.info("Saved image artifact: %s (%d bytes)", img_path, len(image_bytes))

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    clip_line = ""
    if clip_score is not None:
        status = "PASS" if clip_score >= CLIP_SCORE_THRESHOLD else "FAIL"
        clip_line = (
            f"| CLIP Score | {clip_score:.4f} "
            f"({status}, threshold: {CLIP_SCORE_THRESHOLD}) |\n"
        )

    md = (
        f"### FLUX.2-klein — `{case_id}`\n\n"
        f"| | |\n|---|---|\n"
        f"| Prompt | {prompt} |\n"
        f"| Size | {len(image_bytes):,} bytes |\n"
        f"{clip_line}"
        f"| Artifact | `{case_id}.{ext}` (download from Artifacts section above) |\n\n"
    )

    with open(summary_file, "a") as f:
        f.write(md)


def _compute_clip_score(image_bytes: bytes, prompt: str) -> float | None:
    try:
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        model.eval()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            score = outputs.logits_per_image.item() / 100.0

        logger.info("CLIP score for '%s': %.4f", prompt, score)
        return score
    except Exception as e:
        logger.warning("CLIP score computation failed: %s", e)
        return None


def _check_against_reference(case: DiffusionTestCase, content: bytes) -> None:
    gt_dir = os.environ.get("SGLANG_CONSISTENCY_GT_DIR")
    assert gt_dir, (
        f"{case.id}: set SGLANG_CONSISTENCY_GT_DIR to the XPU reference image dir"
    )
    num_gpus = case.server_args.num_gpus
    output_frames = [image_bytes_to_numpy(content)]
    if not gt_exists(case.id, num_gpus):
        saved = save_missing_consistency_gt_artifact(
            artifact_dir=ARTIFACT_DIR,
            case_id=case.id,
            num_gpus=num_gpus,
            output_frames=output_frames,
            is_video=False,
        )
        pytest.fail(
            f"{case.id}: no reference image in {gt_dir}. This run's output was "
            f"saved to {saved}; inspect it, then copy it into {gt_dir}."
        )

    result = compare_with_gt(
        output_frames=output_frames,
        gt_data=load_consistency_gt(case.id, num_gpus),
        thresholds=KLEIN_TI2I_THRESHOLDS,
        case_id=case.id,
    )
    assert result.passed, (
        f"{case.id}: output diverges from the reference in {gt_dir}: "
        f"clip={result.min_similarity:.4f} ssim={result.min_ssim:.4f} "
        f"psnr={result.min_psnr:.2f} mean_abs_diff={result.max_mean_abs_diff:.2f} "
        f"(thresholds {KLEIN_TI2I_THRESHOLDS})"
    )


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestFlux2KleinXPU(DiffusionServerBase):
    """Intel XPU manual test for FLUX.2-klein t2i and ti2i generation."""

    @classmethod
    def teardown_class(cls):
        try:
            super().teardown_class()
        except AttributeError:
            pass

    @pytest.fixture(params=XPU_FLUX2_KLEIN_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        return request.param

    def test_diffusion_generation(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )

        # No _validate_and_record: it fails any case absent from
        # perf_baselines/xpu_b60.json, and klein has no XPU perf baseline.
        _, content = self.run_and_collect(diffusion_server, case.id, generate_fn)

        self._test_v1_models_endpoint(diffusion_server, case)

        if case.sampling_params.image_path is not None:
            _check_against_reference(case, content)
            return

        prompt = case.sampling_params.prompt or ""
        clip_score = _compute_clip_score(content, prompt)
        # A scorer that failed to load or run must not read as a pass.
        assert clip_score is not None, (
            f"{case.id}: CLIP score not computed; see the warning above"
        )
        logger.info(
            "CLIP score: %.4f (threshold: %.2f)", clip_score, CLIP_SCORE_THRESHOLD
        )
        assert clip_score >= CLIP_SCORE_THRESHOLD, (
            f"CLIP score {clip_score:.4f} below threshold {CLIP_SCORE_THRESHOLD} "
            f"for prompt '{prompt}'"
        )

        _save_image_and_write_summary(case.id, prompt, content, clip_score)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
