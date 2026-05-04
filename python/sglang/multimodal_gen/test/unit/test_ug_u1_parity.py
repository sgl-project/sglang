# SPDX-License-Identifier: Apache-2.0

import json
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sglang.srt.ug.parity import (
    UGParityArtifact,
    UGParityCase,
    UGParityRunner,
    UGTensorSummary,
    compare_ug_parity_artifacts,
    summarize_ug_image,
    write_ug_parity_bundle,
)


class TestU1OfficialParityHarness(unittest.TestCase):
    def test_u1_parity_case_roundtrip(self):
        case = UGParityCase(
            case_id="u1-vlm-smoke",
            model="sensenova-u1",
            task="vlm",
            prompt="Describe this image.",
            image_path="/tmp/u1-input.png",
            seed=123,
            sampling_params={"max_new_tokens": 16},
            dump_points=("text", "u_logits"),
        )

        restored = UGParityCase.from_json(case.to_json())

        self.assertEqual(restored.case_id, case.case_id)
        self.assertEqual(restored.model, "sensenova-u1")
        self.assertEqual(restored.task, "vlm")
        self.assertEqual(restored.sampling_params["max_new_tokens"], 16)
        self.assertEqual(restored.dump_points, ("text", "u_logits"))

    def test_u1_tensor_and_image_summary_are_stable(self):
        tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        tensor_summary = UGTensorSummary.from_tensor(tensor)
        same_tensor_summary = UGTensorSummary.from_tensor(tensor.clone())
        image = Image.fromarray(np.full((3, 4, 3), 17, dtype=np.uint8), "RGB")
        image_summary = summarize_ug_image(image)

        self.assertEqual(tensor_summary, same_tensor_summary)
        self.assertEqual(tensor_summary.shape, (2, 2))
        self.assertEqual(tensor_summary.dtype, "torch.float32")
        self.assertEqual(image_summary.size, (4, 3))
        self.assertTrue(image_summary.sha256)

    def test_fake_runner_artifacts_pass_and_mismatch_fail(self):
        case = _case()
        reference = _FakeRunner(runner="official", text="a cup").run(case)
        candidate = _FakeRunner(runner="sglang", text="a cup").run(case)
        mismatch = _FakeRunner(runner="sglang", text="a vase").run(case)

        passed = compare_ug_parity_artifacts(reference, candidate)
        failed = compare_ug_parity_artifacts(reference, mismatch)

        self.assertTrue(passed.passed)
        self.assertFalse(failed.passed)
        self.assertIn("text", {diff.field for diff in failed.diffs})

    def test_write_u1_parity_bundle(self):
        case = _case()
        reference = _FakeRunner(runner="official", text="a cup").run(case)
        candidate = _FakeRunner(runner="sglang", text="a cup").run(case)
        report = compare_ug_parity_artifacts(reference, candidate)

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle = write_ug_parity_bundle(
                output_dir=tmpdir,
                case=case,
                reference=reference,
                candidate=candidate,
                report=report,
            )

            self.assertTrue((bundle / "case.json").exists())
            self.assertTrue((bundle / "reference.json").exists())
            self.assertTrue((bundle / "candidate.json").exists())
            report_json = json.loads((bundle / "report.json").read_text())
            self.assertTrue(report_json["passed"])

    def test_runtime_import_firewall_blocks_official_u1_imports(self):
        repo = Path(__file__).resolve().parents[5]
        runtime_root = repo / "python" / "sglang"
        forbidden = [
            re.compile(r"^\s*(from|import)\s+sensenova(?:\.|\s|$)", re.MULTILINE),
            re.compile(r"^\s*(from|import)\s+seed(?:\.|\s|$)", re.MULTILINE),
            re.compile(r"^\s*(from|import)\s+u1_official(?:\.|\s|$)", re.MULTILINE),
        ]
        offenders = []

        for path in runtime_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in forbidden):
                offenders.append(str(path.relative_to(repo)))

        self.assertEqual(offenders, [])


class _FakeRunner(UGParityRunner):
    def __init__(self, *, runner: str, text: str):
        self.runner = runner
        self.text = text

    def run(self, case: UGParityCase) -> UGParityArtifact:
        image = Image.fromarray(np.full((4, 4, 3), 91, dtype=np.uint8), "RGB")
        return UGParityArtifact(
            case_id=case.case_id,
            model=case.model,
            task=case.task,
            runner=self.runner,
            text=self.text,
            image=summarize_ug_image(image),
            tensors={"u_logits": UGTensorSummary.from_tensor(torch.ones(2, 2))},
            metadata={"seed": case.seed},
        )


def _case():
    return UGParityCase(
        case_id="u1-t2i-smoke",
        model="sensenova-u1",
        task="t2i",
        prompt="draw a cup",
        seed=7,
        sampling_params={"num_inference_steps": 2},
        dump_points=("image", "u_logits"),
    )


if __name__ == "__main__":
    unittest.main()
