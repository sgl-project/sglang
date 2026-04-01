"""AMD nightly test for Z-Image-Turbo diffusion model (text-to-image)."""

import io
import logging
import os

import pytest

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
)
from sglang.test.ci.ci_register import register_amd_ci

logger = logging.getLogger(__name__)

register_amd_ci(est_time=1800, suite="nightly-amd-1-gpu-zimage-turbo", nightly=True)

AMD_ZIMAGE_CASES = [
    DiffusionTestCase(
        "zimage_image_t2i",
        DiffusionServerArgs(model_path="Tongyi-MAI/Z-Image-Turbo", modality="image"),
        DiffusionSamplingParams(
            prompt="Doraemon is eating dorayaki",
            output_size="1024x1024",
        ),
    ),
]

CLIP_SCORE_THRESHOLD = 0.20


ARTIFACT_DIR = os.environ.get(
    "SGLANG_DIFFUSION_ARTIFACT_DIR", "/tmp/diffusion-artifacts"
)


def _save_image_and_write_summary(
    case_id: str, prompt: str, image_bytes: bytes, clip_score: float | None = None
):
    """Save generated image to artifact dir and write summary."""
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
        clip_line = f"| CLIP Score | {clip_score:.4f} ({status}, threshold: {CLIP_SCORE_THRESHOLD}) |\n"

    md = (
        f"### Z-Image-Turbo — `{case_id}`\n\n"
        f"| | |\n|---|---|\n"
        f"| Prompt | {prompt} |\n"
        f"| Size | {len(image_bytes):,} bytes |\n"
        f"{clip_line}"
        f"| Artifact | `{case_id}.{ext}` (download from Artifacts section above) |\n\n"
    )

    with open(summary_file, "a") as f:
        f.write(md)


def _compute_clip_score(image_bytes: bytes, prompt: str) -> float | None:
    """Compute CLIP cosine similarity between the image and prompt."""
    try:
        import torch
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


class TestZImageTurboAMD(DiffusionServerBase):
    """AMD nightly test for Z-Image-Turbo text-to-image generation."""

    @classmethod
    def teardown_class(cls):
        try:
            super().teardown_class()
        except AttributeError:
            pass

    @pytest.fixture(params=AMD_ZIMAGE_CASES, ids=lambda c: c.id)
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

        perf_record, content = self.run_and_collect(
            diffusion_server, case.id, generate_fn
        )

        self._validate_and_record(case, perf_record)
        self._test_v1_models_endpoint(diffusion_server, case)

        prompt = case.sampling_params.prompt or ""
        clip_score = _compute_clip_score(content, prompt)

        if clip_score is not None:
            logger.info(
                "CLIP score: %.4f (threshold: %.2f)", clip_score, CLIP_SCORE_THRESHOLD
            )
            assert clip_score >= CLIP_SCORE_THRESHOLD, (
                f"CLIP score {clip_score:.4f} below threshold {CLIP_SCORE_THRESHOLD} "
                f"for prompt '{prompt}'"
            )

        _save_image_and_write_summary(case.id, prompt, content, clip_score)
