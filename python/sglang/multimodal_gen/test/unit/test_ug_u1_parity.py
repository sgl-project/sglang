# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_u1_vlm_official_reference_mode_writes_candidate_error_bundle(self):
        run_from_env = _load_u1_official_parity_harness()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_vqa_repo(root, text="official answer")
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"

            bundle = run_from_env(
                {
                    "SGLANG_TEST_U1_PARITY_MODE": "vlm_official_reference",
                    "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                    "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                    "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                    "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                    "SGLANG_TEST_U1_VLM_IMAGE": str(image_path),
                    "SGLANG_TEST_U1_VLM_QUESTION": "what is here?",
                    "SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS": "4",
                    "SGLANG_TEST_U1_VLM_DEVICE": "cpu",
                }
            )

            reference = json.loads((bundle / "reference.json").read_text())
            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            self.assertEqual(reference["text"], "official answer")
            self.assertIsNone(reference["error"])
            self.assertIn("not wired", candidate["error"])
            self.assertFalse(report["passed"])
            self.assertEqual(report["diffs"][0]["field"], "error")

    def test_u1_vlm_official_reference_mode_can_run_sglang_candidate(self):
        run_from_env = _load_u1_official_parity_harness()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_vqa_repo(root, text="aligned answer")
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"

            bundle = run_from_env(
                {
                    "SGLANG_TEST_U1_PARITY_MODE": "vlm_official_reference",
                    "SGLANG_TEST_U1_PARITY_RUN_SGLANG_CANDIDATE": "1",
                    "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                    "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                    "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                    "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                    "SGLANG_TEST_U1_VLM_IMAGE": str(image_path),
                    "SGLANG_TEST_U1_VLM_QUESTION": "what is here?",
                    "SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS": "4",
                    "SGLANG_TEST_U1_VLM_DEVICE": "cpu",
                }
            )

            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            self.assertEqual(candidate["text"], "aligned answer")
            self.assertFalse(candidate["metadata"]["native_srt_model_runner"])
            self.assertEqual(candidate["debug_counters"]["prefill_count"], 1)
            self.assertTrue(report["passed"])

    def test_u1_vlm_official_reference_mode_can_run_native_srt_candidate(self):
        module = _load_u1_official_parity_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            official_repo = _write_fake_u1_vqa_repo(root, text="native answer")
            image_path = _write_fake_image(root)
            output_dir = root / "bundle"

            with patch.object(module, "_run_sglang_native_vlm_candidate") as run_native:
                run_native.return_value = UGParityArtifact(
                    case_id="u1-vlm-official-reference",
                    model="sensenova-u1",
                    task="vlm",
                    runner="sglang",
                    text="native answer",
                    image=summarize_ug_image(image_path),
                    metadata={
                        "candidate_backend": "u1_native_srt_vlm_full_prefill",
                        "native_srt_model_runner": True,
                        "kv_decode": False,
                    },
                )
                bundle = module.run_u1_official_parity_from_env(
                    {
                        "SGLANG_TEST_U1_PARITY_MODE": "vlm_official_reference",
                        "SGLANG_TEST_U1_PARITY_RUN_SGLANG_NATIVE_CANDIDATE": "1",
                        "SGLANG_TEST_U1_PARITY_OUTPUT": str(output_dir),
                        "SGLANG_TEST_U1_OFFICIAL_PY": sys.executable,
                        "SGLANG_TEST_U1_OFFICIAL_REPO": str(official_repo),
                        "SGLANG_TEST_U1_MODEL_PATH": "/fake/model",
                        "SGLANG_TEST_U1_VLM_IMAGE": str(image_path),
                        "SGLANG_TEST_U1_VLM_QUESTION": "what is here?",
                        "SGLANG_TEST_U1_VLM_MAX_NEW_TOKENS": "4",
                        "SGLANG_TEST_U1_VLM_DEVICE": "cpu",
                    }
                )

            candidate = json.loads((bundle / "candidate.json").read_text())
            report = json.loads((bundle / "report.json").read_text())

            run_native.assert_called_once()
            self.assertEqual(candidate["text"], "native answer")
            self.assertTrue(candidate["metadata"]["native_srt_model_runner"])
            self.assertEqual(
                candidate["metadata"]["candidate_backend"],
                "u1_native_srt_vlm_full_prefill",
            )
            self.assertFalse(candidate["metadata"]["kv_decode"])
            self.assertTrue(report["passed"])

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


def _load_u1_official_parity_harness():
    return _load_u1_official_parity_module().run_u1_official_parity_from_env


def _load_u1_official_parity_module():
    repo = Path(__file__).resolve().parents[5]
    path = repo / "test/registered/scheduler/test_u1_official_parity_harness.py"
    spec = importlib.util.spec_from_file_location("u1_official_parity_harness", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_fake_u1_vqa_repo(root: Path, *, text: str) -> Path:
    official_repo = root / "official"
    vqa_dir = official_repo / "examples/vqa"
    vqa_dir.mkdir(parents=True)
    fake_script = vqa_dir / "inference.py"
    fake_script.write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--model_path')",
                "parser.add_argument('--image')",
                "parser.add_argument('--question')",
                "parser.add_argument('--output')",
                "parser.add_argument('--max_new_tokens')",
                "parser.add_argument('--device')",
                "parser.add_argument('--dtype')",
                "parser.add_argument('--attn_backend')",
                "args = parser.parse_args()",
                f"Path(args.output).write_text({text!r})",
                "print('fake official u1 ok')",
            ]
        ),
        encoding="utf-8",
    )
    return official_repo


def _write_fake_image(root: Path) -> Path:
    image_path = root / "image.png"
    Image.fromarray(np.full((8, 8, 3), 23, dtype=np.uint8), "RGB").save(image_path)
    return image_path


if __name__ == "__main__":
    unittest.main()
