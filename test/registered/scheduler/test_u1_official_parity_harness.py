# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sglang.srt.ug.parity import (
    UGParityArtifact,
    UGParityCase,
    UGTensorSummary,
    compare_ug_parity_artifacts,
    summarize_ug_image,
    write_ug_parity_bundle,
)


class TestU1OfficialParityHarness(unittest.TestCase):
    def test_u1_official_parity_harness(self):
        run_u1_official_parity_from_env(os.environ)


def run_u1_official_parity_from_env(env) -> Path:
    if env.get("SGLANG_TEST_U1_PARITY_DRY_RUN") == "1":
        output_dir = Path(
            env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
            or tempfile.mkdtemp(prefix="u1-parity-")
        )
        return _write_dry_run_bundle(output_dir)

    reference_path = env.get("SGLANG_TEST_U1_PARITY_REFERENCE_ARTIFACT")
    candidate_path = env.get("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT")
    output_dir = env.get("SGLANG_TEST_U1_PARITY_OUTPUT")
    missing = [
        name
        for name, value in (
            ("SGLANG_TEST_U1_PARITY_REFERENCE_ARTIFACT", reference_path),
            ("SGLANG_TEST_U1_PARITY_CANDIDATE_ARTIFACT", candidate_path),
            ("SGLANG_TEST_U1_PARITY_OUTPUT", output_dir),
        )
        if not value
    ]
    if missing:
        raise unittest.SkipTest(
            "U1 official parity harness is opt-in; missing env: "
            + ", ".join(missing)
        )

    reference = UGParityArtifact.from_json(Path(reference_path).read_text())
    candidate = UGParityArtifact.from_json(Path(candidate_path).read_text())
    case = _case_from_artifact(reference)
    report = compare_ug_parity_artifacts(reference, candidate)
    bundle = write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )
    if not report.passed:
        raise AssertionError(f"U1 parity failed; report: {bundle / 'report.json'}")
    return bundle


def _write_dry_run_bundle(output_dir: Path) -> Path:
    case = UGParityCase(
        case_id="u1-dry-run",
        model="sensenova-u1",
        task="vlm",
        prompt="describe this image",
        seed=1,
        sampling_params={"max_new_tokens": 4},
        dump_points=("text", "u_logits"),
    )
    reference = _dry_run_artifact(case, runner="official")
    candidate = _dry_run_artifact(case, runner="sglang")
    report = compare_ug_parity_artifacts(reference, candidate)
    return write_ug_parity_bundle(
        output_dir=output_dir,
        case=case,
        reference=reference,
        candidate=candidate,
        report=report,
    )


def _dry_run_artifact(case: UGParityCase, *, runner: str) -> UGParityArtifact:
    image = Image.fromarray(np.full((4, 4, 3), 7, dtype=np.uint8), "RGB")
    return UGParityArtifact(
        case_id=case.case_id,
        model=case.model,
        task=case.task,
        runner=runner,
        text="dry run",
        image=summarize_ug_image(image),
        tensors={"u_logits": UGTensorSummary.from_tensor(torch.ones(2, 2))},
        metadata={"dry_run": True},
    )


def _case_from_artifact(artifact: UGParityArtifact) -> UGParityCase:
    return UGParityCase(
        case_id=artifact.case_id,
        model=artifact.model,
        task=artifact.task,
        metadata={"source": "artifact_env"},
    )


if __name__ == "__main__":
    unittest.main()
