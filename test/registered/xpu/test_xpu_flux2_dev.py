"""FLUX.2-dev text-to-image on Intel XPU (4-GPU nightly).

Mirrors ``test/registered/amd/test_zimage_turbo.py`` but targets FLUX.2-dev
with ``num_gpus=4`` and registers to the XPU 4-GPU nightly suite. The
diffusion server harness is device-agnostic; XPU dispatch is picked up by
``current_platform`` inside multimodal_gen at server launch.
"""

from __future__ import annotations

import logging
import os

import pytest
import torch

from sglang.multimodal_gen.test.quality_metrics import (
    QualityScores,
    format_quality_scores,
)
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
from sglang.test.ci.ci_register import register_xpu_ci

logger = logging.getLogger(__name__)

register_xpu_ci(est_time=3600, suite="nightly-xpu-4-gpu", nightly=True)

XPU_FLUX2_CASES = [
    DiffusionTestCase(
        "flux2_dev_image_t2i",
        DiffusionServerArgs(
            model_path="black-forest-labs/FLUX.2-dev",
            modality="image",
            num_gpus=4,
            tp_size=4,
            dit_layerwise_offload=True,
            extras=[
                "--dit-precision",
                "bf16",
                "--vae-precision",
                "bf16",
                "--text-encoder-precisions",
                "bf16",
            ],
        ),
        DiffusionSamplingParams(
            prompt="A curious raccoon in a top hat, oil painting",
            output_size="1024x1024",
        ),
        # XPU has no baseline under test/server/perf_baselines/, so the accuracy
        # check is the quality_thresholds.json floors, not latency/consistency.
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
        run_quality_check=True,
    ),
]

ARTIFACT_DIR = os.environ.get(
    "SGLANG_DIFFUSION_ARTIFACT_DIR", "/tmp/diffusion-artifacts"
)


def _save_image_and_write_summary(
    case_id: str,
    prompt: str,
    image_bytes: bytes,
    scores: QualityScores,
    failures: list[str],
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

    quality_status = "FAIL" if failures else "PASS"
    md = (
        f"### FLUX.2-dev — `{case_id}`\n\n"
        f"| | |\n|---|---|\n"
        f"| Prompt | {prompt} |\n"
        f"| Size | {len(image_bytes):,} bytes |\n"
        f"| Quality | {format_quality_scores(scores)} ({quality_status}) |\n"
        f"| Artifact | `{case_id}.{ext}` (download from Artifacts section above) |\n\n"
    )
    if failures:
        md += "".join(f"- {failure}\n" for failure in failures) + "\n"

    with open(summary_file, "a") as f:
        f.write(md)


@pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()),
    reason="Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestFlux2DevXPU(DiffusionServerBase):
    """Intel XPU nightly test for FLUX.2-dev text-to-image generation."""

    @classmethod
    def teardown_class(cls):
        try:
            super().teardown_class()
        except AttributeError:
            pass

    @pytest.fixture(params=XPU_FLUX2_CASES, ids=lambda c: c.id)
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
        scores = QualityScores()
        failures: list[str] = []
        if case.run_quality_check:
            # Score first, publish the numbers, then fail: a failing run is
            # exactly the one whose scores need to reach the step summary.
            scores, failures = self._score_quality(case, content)
            logger.info("Quality scores: %s", format_quality_scores(scores))
        _save_image_and_write_summary(case.id, prompt, content, scores, failures)

        assert not failures, (
            f"Quality check failed for {case.id}: {'; '.join(failures)}"
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
