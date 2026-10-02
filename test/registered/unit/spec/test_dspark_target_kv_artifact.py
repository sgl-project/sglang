"""Deployment evidence must describe the exact checkpoint being audited."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.srt.speculative.dspark_components import dspark_target_kv_artifact as audit
from sglang.srt.training_capture.protocol import ContractError, digest_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dspark_target_kv_utils import make_target_kv_config
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTargetKVArtifact(CustomTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "validation").mkdir()
        save_file({"weight": torch.ones(2, 2)}, self.root / "model.safetensors")
        save_file(
            {"token_ids": torch.arange(8)},
            self.root / "validation/inputs.safetensors",
        )
        self.config = make_target_kv_config()
        self.config.update(
            num_hidden_layers=2,
            block_size=3,
            mask_token_id=255,
            vocab_size=256,
            dtype="bfloat16",
        )
        self.config["target_kv_contract"]["validation"]["golden_fixture_sha256"] = (
            self.digest("validation/inputs.safetensors")
        )
        self.write_config()
        self.report = {
            "status": "passed",
            "artifact_sha256": {name: self.digest(name) for name in audit._ARTIFACTS},
            "fixture_sha256": self.digest("validation/inputs.safetensors"),
            "weights_sha256": self.digest("model.safetensors"),
            "rtol": 1e-5,
            "atol": 1e-5,
            "stages": {
                name: {
                    "passed": True,
                    "elements": elements,
                    "mismatched": 0,
                    "nonfinite": 0,
                    "max_abs": 0.0,
                    "rms": 0.0,
                }
                for name, elements in (
                    ("layer.0", 72),
                    ("layer.1", 72),
                    ("hidden", 72),
                    ("base", 2304),
                    ("corrected", 2304),
                )
            },
            "anchors": [2, 3, 4],
            "valid_labels": 7,
            "loss": 1.0,
            "gradient_l1": {
                name: 1.0 for name in ("kv_encoder", "layers", "norm", "markov_head")
            },
            "dtype": "bfloat16",
            "reference_adapter": "specforge_backbone_with_shared_kv_encoder",
            "reference_attention": "flex_attention",
            "serving_attention": "triton",
            "specforge_sources": {
                f"specforge.modeling.draft.{name}": "a" * 64
                for name in ("dflash", "dflash_kernels", "dspark")
            },
        }
        self.write_report()

    def digest(self, name):
        return digest_bytes((self.root / name).read_bytes())

    def write_config(self):
        (self.root / "config.json").write_text(json.dumps(self.config))

    def write_report(self, value=None):
        (self.root / "validation/parity.json").write_text(
            json.dumps(self.report if value is None else value)
        )

    def test_receipt_binds_bytes_without_executing_models_or_writing_files(self):
        before = {p: p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        with patch.object(torch.cuda, "init", side_effect=AssertionError("no GPU")):
            receipt = audit.audit_target_kv_checkpoint(self.root)
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual(receipt["artifact_sha256"], self.report["artifact_sha256"])
        self.assertEqual(
            receipt["parity_report_sha256"], self.digest("validation/parity.json")
        )
        self.assertFalse(receipt["acceptance_report_bound"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_changed_or_missing_artifacts_cannot_reuse_a_pass(self):
        for name in (*audit._ARTIFACTS, "validation/parity.json"):
            path = self.root / name
            original = path.read_bytes()
            with self.subTest(name=name, change="missing"):
                path.unlink()
                with self.assertRaises(ContractError):
                    audit.audit_target_kv_checkpoint(self.root)
                path.write_bytes(original)
            if name != "validation/parity.json":
                with self.subTest(name=name, change="different"):
                    path.write_bytes(original + b" ")
                    with self.assertRaisesRegex(ContractError, "current artifacts"):
                        audit.audit_target_kv_checkpoint(self.root)
                    path.write_bytes(original)

    def test_incomplete_failed_or_weaker_numerical_reports_are_rejected(self):
        changes = [
            {"status": "failed"},
            {"status": "running"},
            {"stages": {}},
            {"stages": {k: v for k, v in self.report["stages"].items() if k != "base"}},
            {"rtol": 0.03},
            {"atol": float("nan")},
            {"anchors": []},
            {"anchors": [True, 3, 4]},
            {"valid_labels": 10},
            {"loss": float("inf")},
            {"gradient_l1": {"kv_encoder": 1.0}},
            {"gradient_l1": self.report["gradient_l1"] | {"layers": 0.0}},
            {"specforge_sources": {}},
            {"dtype": "float32"},
            {"weights_sha256": "f" * 64},
            {"fixture_sha256": "f" * 64},
            {"artifact_sha256": self.report["artifact_sha256"] | {"extra": "f" * 64}},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ContractError):
                self.write_report(self.report | change)
                audit.audit_target_kv_checkpoint(self.root)

    def test_unreported_weight_shards_or_index_cannot_change_loader_selection(self):
        for name in ("extra.safetensors", "model.safetensors.index.json"):
            path = self.root / name
            with self.subTest(name=name):
                path.write_text("{}")
                with self.assertRaisesRegex(ContractError, "single model.safetensors"):
                    audit.audit_target_kv_checkpoint(self.root)
                path.unlink()

    def test_every_stage_requires_full_finite_matching_elements(self):
        for change in (
            {"passed": False},
            {"passed": 1},
            {"elements": 71},
            {"elements": True},
            {"mismatched": 1},
            {"nonfinite": 1},
            {"max_abs": float("inf")},
            {"rms": -1},
        ):
            with self.subTest(change=change), self.assertRaises(ContractError):
                report = copy.deepcopy(self.report)
                report["stages"]["layer.1"].update(change)
                self.write_report(report)
                audit.audit_target_kv_checkpoint(self.root)

    def test_contract_and_report_must_agree_even_after_rehashing_config(self):
        original = copy.deepcopy(self.config)
        for change in (
            {"block_size": 4},
            {"mask_token_id": 254},
            {"vocab_size": 255},
            {"num_hidden_layers": True},
            {"num_hidden_layers": 4097},
            {"torch_dtype": "float16"},
        ):
            with self.subTest(change=change), self.assertRaises(ContractError):
                self.config = original | change
                self.write_config()
                report = copy.deepcopy(self.report)
                report["artifact_sha256"]["config.json"] = self.digest("config.json")
                self.write_report(report)
                audit.audit_target_kv_checkpoint(self.root)

    def test_replacement_while_other_files_are_read_is_rejected(self):
        original = audit._hash_artifact
        for name in ("config.json", "validation/parity.json"):
            path = self.root / name
            data = path.read_bytes()

            def replace_earlier_file(current, stamps, path=path, data=data):
                digest = original(current, stamps)
                if current.name == "model.safetensors":
                    path.write_bytes(data + b" ")
                return digest

            with (
                self.subTest(name=name),
                patch.object(audit, "_hash_artifact", side_effect=replace_earlier_file),
                self.assertRaisesRegex(ContractError, "changed during audit"),
            ):
                audit.audit_target_kv_checkpoint(self.root)
            path.write_bytes(data)

    def test_acceptance_is_an_explicit_opaque_pinned_artifact(self):
        with self.assertRaisesRegex(ContractError, "does not pin"):
            audit.audit_target_kv_checkpoint(self.root, require_acceptance=True)
        acceptance = self.root / "validation/acceptance.json"
        acceptance.write_text('{"benchmark_revision": "fixture"}')
        self.config["target_kv_contract"]["validation"]["acceptance_report_sha256"] = (
            self.digest("validation/acceptance.json")
        )
        self.write_config()
        self.report["artifact_sha256"]["config.json"] = self.digest("config.json")
        self.write_report()
        self.assertTrue(
            audit.audit_target_kv_checkpoint(self.root, require_acceptance=True)[
                "acceptance_report_bound"
            ]
        )
        acceptance.write_text("{}")
        with self.assertRaisesRegex(ContractError, "acceptance report differs"):
            audit.audit_target_kv_checkpoint(self.root)
        acceptance.unlink()
        with self.assertRaises(ContractError):
            audit.audit_target_kv_checkpoint(self.root)

    def test_cli_returns_structured_failure_and_nonzero_status(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(audit.main(["--checkpoint", str(self.root)]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "verified")
        self.write_report({"status": "passed"})
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(audit.main(["--checkpoint", str(self.root)]), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
