# SPDX-License-Identifier: Apache-2.0
"""Opt-in BAGEL official-vs-SGLang UG parity harness smoke.

Usage:
SGLANG_TEST_BAGEL_OFFICIAL_REPO=/data/repos/BAGEL \
SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL=/data/models/BAGEL-7B-MoT \
SGLANG_TEST_BAGEL_PARITY_OUTPUT=/tmp/ug-parity \
python3 test/registered/scheduler/test_bagel_official_parity_harness.py
"""

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang.srt.ug.parity import (
    UGParityArtifact,
    UGParityCase,
    compare_ug_parity_artifacts,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=120,
    suite="stage-b-test-1-gpu-large",
    disabled=(
        "Manual BAGEL official parity harness smoke; requires "
        "SGLANG_TEST_BAGEL_OFFICIAL_REPO and "
        "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
    ),
)

_OFFICIAL_REPO_ENV = "SGLANG_TEST_BAGEL_OFFICIAL_REPO"
_MODEL_ENV = "SGLANG_TEST_BAGEL_QWEN2_MOT_MODEL"
_OUTPUT_ENV = "SGLANG_TEST_BAGEL_PARITY_OUTPUT"


def _has_live_env() -> bool:
    return bool(os.getenv(_OFFICIAL_REPO_ENV) and os.getenv(_MODEL_ENV))


@unittest.skipUnless(
    _has_live_env(),
    f"Set {_OFFICIAL_REPO_ENV} and {_MODEL_ENV} for BAGEL parity harness smoke",
)
class TestBAGELOfficialParityHarness(CustomTestCase):
    def test_subprocess_probe_writes_artifacts_and_report(self):
        official_repo = Path(os.environ[_OFFICIAL_REPO_ENV]).expanduser()
        checkpoint_dir = Path(os.environ[_MODEL_ENV]).expanduser()
        self.assertTrue(official_repo.exists(), official_repo)
        self.assertTrue(checkpoint_dir.exists(), checkpoint_dir)

        output_dir = Path(
            os.getenv(_OUTPUT_ENV) or tempfile.mkdtemp(prefix="ug-parity-")
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        case = UGParityCase(
            case_id="bagel-parity-harness-probe",
            task="vlm",
            prompt="probe",
            seed=123,
            metadata={
                "official_repo": str(official_repo),
                "checkpoint": str(checkpoint_dir),
            },
        )
        case_path = output_dir / "case.json"
        official_path = output_dir / "reference.official.json"
        sglang_path = output_dir / "candidate.sglang.json"
        report_path = output_dir / "report.json"
        case.write_json(case_path)

        _run_probe_subprocess(
            case_path=case_path,
            output_path=official_path,
            runner="official",
            probe_path=official_repo,
        )
        _run_probe_subprocess(
            case_path=case_path,
            output_path=sglang_path,
            runner="sglang",
            probe_path=checkpoint_dir,
        )

        reference = UGParityArtifact.read_json(official_path)
        candidate = UGParityArtifact.read_json(sglang_path)
        report = compare_ug_parity_artifacts(reference, candidate)
        report.write_json(report_path)

        self.assertTrue(report.passed, report.to_json())
        self.assertTrue(case_path.exists())
        self.assertTrue(official_path.exists())
        self.assertTrue(sglang_path.exists())
        self.assertTrue(report_path.exists())


def _run_probe_subprocess(
    *,
    case_path: Path,
    output_path: Path,
    runner: str,
    probe_path: Path,
) -> None:
    code = r"""
import importlib.util
import sys
from pathlib import Path

import torch

from sglang.srt.ug.parity import UGParityArtifact, UGParityCase, UGTensorSummary

case = UGParityCase.read_json(sys.argv[1])
output_path = Path(sys.argv[2])
runner = sys.argv[3]
probe_path = Path(sys.argv[4])
metadata = {"probe_path": str(probe_path)}
if runner == "official":
    sys.path.insert(0, str(probe_path))
    metadata["official_inferencer_found"] = importlib.util.find_spec("inferencer") is not None
artifact = UGParityArtifact(
    case_id=case.case_id,
    runner=runner,
    task=case.task,
    text="parity_probe",
    tensors={"probe": UGTensorSummary.from_tensor(torch.tensor([1.0, 2.0]))},
    metadata=metadata,
)
artifact.write_json(output_path)
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(code),
            str(case_path),
            str(output_path),
            runner,
            str(probe_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
    )


if __name__ == "__main__":
    unittest.main(verbosity=3)
