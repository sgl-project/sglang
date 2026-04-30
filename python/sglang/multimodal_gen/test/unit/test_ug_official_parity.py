# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import torch

from sglang.srt.ug.parity import (
    UGImageSummary,
    UGParityArtifact,
    UGParityCase,
    UGParityReport,
    UGParityTolerance,
    UGTensorSummary,
    compare_ug_parity_artifacts,
    run_ug_parity_case,
)


class TestUGOfficialParityHarness(unittest.TestCase):
    def test_case_roundtrip(self):
        case = UGParityCase(
            case_id="vlm-smoke-001",
            task="vlm",
            prompt="Describe this image.",
            image_path="/tmp/image.png",
            seed=123,
            sampling_params={"max_new_tokens": 16},
            dump_points=("text", "logits"),
            metadata={"suite": "unit"},
        )

        loaded = UGParityCase.from_json(case.to_json())

        self.assertEqual(loaded, case)
        self.assertEqual(loaded.to_dict()["dump_points"], ["text", "logits"])

    def test_case_validation_rejects_bad_inputs(self):
        with self.assertRaisesRegex(ValueError, "case_id"):
            UGParityCase(case_id="", task="vlm")
        with self.assertRaisesRegex(ValueError, "Unsupported UG parity task"):
            UGParityCase(case_id="bad", task="audio")

    def test_interleave_task_alias_is_canonicalized(self):
        case = UGParityCase(case_id="legacy", task="interleaved")
        self.assertEqual(case.task, "interleave")
        self.assertEqual(case.to_dict()["task"], "interleave")

    def test_tensor_summary_is_stable(self):
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        first = UGTensorSummary.from_tensor(tensor)
        second = UGTensorSummary.from_tensor(tensor.clone())

        self.assertEqual(first, second)
        self.assertEqual(first.shape, (2, 2))
        self.assertEqual(first.dtype, "torch.float32")
        self.assertEqual(first.numel, 4)
        self.assertEqual(first.min, 1.0)
        self.assertEqual(first.max, 4.0)
        self.assertAlmostEqual(first.mean, 2.5)

    def test_artifact_json_roundtrip(self):
        artifact = _make_artifact(
            runner="official",
            text="a white cat",
            tensor=torch.ones(2, 2),
            image_bytes=b"image-bytes",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            artifact.write_json(path)
            loaded = UGParityArtifact.read_json(path)

        self.assertEqual(loaded, artifact)

    def test_report_passes_for_matching_fake_artifacts(self):
        case = UGParityCase(case_id="match", task="vlm", prompt="hello")
        runner = _FakeRunner(
            "official",
            text="same",
            tensor=torch.tensor([1.0, 2.0]),
            image_bytes=b"same-image",
        )

        report = run_ug_parity_case(
            case,
            reference_runner=runner,
            candidate_runner=_FakeRunner(
                "sglang",
                text="same",
                tensor=torch.tensor([1.0, 2.0]),
                image_bytes=b"same-image",
            ),
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.differences, ())
        self.assertEqual(report.reference_runner, "official")
        self.assertEqual(report.candidate_runner, "sglang")

    def test_report_fails_with_actionable_diff(self):
        reference = _make_artifact(
            runner="official",
            text="reference text",
            tensor=torch.tensor([1.0, 2.0]),
            image_bytes=b"same-image",
        )
        candidate = _make_artifact(
            runner="sglang",
            text="candidate text",
            tensor=torch.tensor([1.0, 3.0]),
            image_bytes=b"same-image",
        )

        report = compare_ug_parity_artifacts(reference, candidate)

        self.assertFalse(report.passed)
        fields = {diff.field for diff in report.differences}
        self.assertIn("text", fields)
        self.assertIn("tensors.logits.max", fields)
        self.assertIn("tensors.logits.mean", fields)
        roundtrip = UGParityReport.from_json(report.to_json())
        self.assertEqual(roundtrip, report)

    def test_report_can_ignore_debug_tensor_diffs(self):
        reference = _make_artifact(
            runner="official",
            text="same",
            tensor=torch.tensor([1.0, 2.0]),
            image_bytes=b"same-image",
        )
        candidate = _make_artifact(
            runner="sglang",
            text="same",
            tensor=torch.tensor([1.0, 3.0]),
            image_bytes=b"same-image",
        )

        report = compare_ug_parity_artifacts(
            reference,
            candidate,
            tolerance=UGParityTolerance(compare_tensors=False),
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.differences, ())

    def test_report_fails_with_token_id_diff(self):
        reference = UGParityArtifact(
            case_id="token-mismatch",
            runner="official",
            task="vlm",
            token_ids={"generated": (151644, 42, 151645)},
        )
        candidate = UGParityArtifact(
            case_id="token-mismatch",
            runner="sglang",
            task="vlm",
            token_ids={"generated": (151644, 43, 151645)},
        )

        report = compare_ug_parity_artifacts(reference, candidate)

        self.assertFalse(report.passed)
        self.assertEqual(report.differences[0].field, "token_ids.generated")
        self.assertEqual(report.differences[0].reference, [151644, 42, 151645])
        self.assertEqual(report.differences[0].candidate, [151644, 43, 151645])

    def test_import_firewall_blocks_runtime_official_imports(self):
        root = Path(__file__).resolve().parents[5] / "python" / "sglang"
        forbidden_patterns = (
            "SGLANG_TEST_BAGEL_OFFICIAL_REPO",
            "modeling.bagel",
            "modeling.autoencoder",
            "modeling.qwen2",
            "data.data_utils",
            "data.transforms",
            "InterleaveInferencer",
            "_build_official_bagel_inferencer",
        )
        hits = []
        for path in root.rglob("*.py"):
            if {"test", "tests", "__pycache__"} & set(path.parts):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in forbidden_patterns:
                if pattern in text:
                    hits.append(f"{path.relative_to(root)}: {pattern}")

        self.assertEqual(hits, [])


class _FakeRunner:
    def __init__(
        self,
        runner_name: str,
        *,
        text: str,
        tensor: torch.Tensor,
        image_bytes: bytes,
    ):
        self.runner_name = runner_name
        self.text = text
        self.tensor = tensor
        self.image_bytes = image_bytes

    def run(self, case: UGParityCase) -> UGParityArtifact:
        return _make_artifact(
            runner=self.runner_name,
            text=self.text,
            tensor=self.tensor,
            image_bytes=self.image_bytes,
            case_id=case.case_id,
            task=case.task,
            metadata={"seed": case.seed},
        )


def _make_artifact(
    *,
    runner: str,
    text: str,
    tensor: torch.Tensor,
    image_bytes: bytes,
    case_id: str = "vlm-smoke-001",
    task: str = "vlm",
    metadata: dict | None = None,
) -> UGParityArtifact:
    return UGParityArtifact(
        case_id=case_id,
        runner=runner,
        task=task,
        text=text,
        images={
            "output": UGImageSummary.from_bytes(
                image_bytes,
                width=16,
                height=16,
                mode="RGB",
            )
        },
        tensors={"logits": UGTensorSummary.from_tensor(tensor)},
        debug_counters={"prefill_count": 1},
        metadata=metadata or {},
    )


if __name__ == "__main__":
    unittest.main()
