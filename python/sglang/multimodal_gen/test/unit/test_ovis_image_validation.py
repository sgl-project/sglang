# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for acceptance of real Ovis validation artifacts.

This runner contract needs no model, diffusion package, or GPU. Keep it in the
CPU lane separately from the native component and model-configuration tests.
"""

import contextlib
import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.test.test_utils import CustomTestCase

RUNNER = (
    Path(__file__).resolve().parents[5] / "test/manual/diffusion/validate_ovis_image.py"
)
SPEC = importlib.util.spec_from_file_location("ovis_image_validation", RUNNER)
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


class TestOvisImageValidation(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.args = SimpleNamespace(
            reference=root / "reference",
            output=root / "native",
            comparison_name="comparison.json",
        )
        tensor = torch.arange(8, dtype=torch.bfloat16).reshape(1, 2, 4) / 8
        timestep = torch.tensor([1000.0])
        self.capture = {
            "capture_schema": 2,
            "capture_metadata": [{"step": 0, "is_cfg_negative": False}],
            "initial_latents": tensor,
            "encoder_hidden_states": [tensor.clone()],
            "predictions": [tensor.clone()],
            "effective_timestep": [timestep.clone()],
            "trajectory": tensor.unsqueeze(0),
            "trajectory_timesteps": timestep.clone(),
            "images": torch.linspace(0, 1, 3 * 8 * 8).reshape(1, 3, 8, 8),
            "timestep_trace": [
                {
                    "step": 0,
                    "is_cfg_negative": False,
                    "timestep_convention": "scheduler_raw",
                    "timestep": timestep.clone(),
                    "effective_timestep": timestep.clone(),
                }
            ],
        }
        arguments = {
            "prompt": "A shop sign",
            "second_prompt": None,
            "height": 512,
            "width": 512,
            "steps": 1,
            "seed": 42,
            "guidance": 5.0,
            "text_length": 256,
            "outputs": 1,
            "vae_tiling": False,
        }
        self.metrics = {
            "arguments": arguments,
            "model_revision": validator.MODEL_REVISION,
            "source": {
                "repository": "/native-worktree",
                "commit": "a" * 40,
                "diff_sha256": "b" * 64,
            },
        }
        for path, mode in (
            (self.args.reference, "reference"),
            (self.args.output, "native"),
        ):
            path.mkdir()
            metrics = copy.deepcopy(self.metrics)
            metrics["arguments"]["mode"] = mode
            if mode == "reference":
                metrics["source"] = {
                    "repository": "/diffusers-reference",
                    "commit": "c" * 40,
                    "diff_sha256": "d" * 64,
                }
            self.save(path, self.capture, metrics)

    def save(self, directory, capture=None, metrics=None):
        if capture is not None:
            torch.save(capture, directory / "tensors.pt")
        if metrics is not None:
            (directory / "metrics.json").write_text(json.dumps(metrics))

    def read_report(self):
        return json.loads((self.args.output / self.args.comparison_name).read_text())

    def compare(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return validator.compare(self.args)

    def test_changed_sampling_configuration_cannot_pass_identical_tensors(self):
        """Previously guidance 5 and 6 could pass when first-step tensors matched."""
        metrics = json.loads((self.args.output / "metrics.json").read_text())
        for key, value in (
            ("prompt", "A different sign"),
            ("second_prompt", "Another sign"),
            ("height", 1024),
            ("width", 1024),
            ("steps", 2),
            ("seed", 43),
            ("guidance", 6.0),
            ("text_length", 128),
            ("outputs", 2),
            ("vae_tiling", True),
            ("model_revision", "different-weights"),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(metrics)
                if key == "model_revision":
                    changed[key] = value
                else:
                    changed["arguments"][key] = value
                self.save(self.args.output, metrics=changed)
                with self.assertRaisesRegex(
                    ValueError, "(?i)" + key.replace("_", "[ _]")
                ):
                    self.compare()
                report = self.read_report()
                self.assertFalse(report["acceptance"])
                self.assertFalse(report["configuration"][key]["matches"])

    def test_missing_metrics_cannot_reuse_a_previous_success_report(self):
        """Tensor artifacts alone cannot establish an equivalent request."""
        self.compare()
        self.assertTrue(self.read_report()["acceptance"])
        (self.args.reference / "metrics.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing metrics.json.*reference"):
            self.compare()
        report = self.read_report()
        self.assertFalse(report["acceptance"])
        self.assertIn("Missing metrics.json", report["acceptance_errors"][0])

    def test_exact_gates_persist_diagnostics_before_raising(self):
        """A BF16 change can pass the component tolerance but violate an exact gate."""
        for label in ("initial_latents", "effective_timestep"):
            with self.subTest(label=label):
                changed = copy.deepcopy(self.capture)
                if label == "initial_latents":
                    changed[label][0, 0, 0] += 0.0078125
                else:
                    for key in ("timestep", "effective_timestep"):
                        changed["timestep_trace"][0][key] += 0.0078125
                    changed[label][0] += 0.0078125
                self.save(self.args.output, capture=changed)
                with self.assertRaises(AssertionError):
                    self.compare()
                report = self.read_report()
                self.assertFalse(report["acceptance"])
                self.assertTrue(
                    any(label in error for error in report["acceptance_errors"])
                )
                stats = (
                    report["initial_latents"][0]
                    if label == "initial_latents"
                    else report["timestep_trace"][0]["effective"]
                )
                self.assertTrue(stats["within_tolerance"])
                self.assertFalse(stats["exact_equal"])

    def test_report_and_assertion_share_dtype_and_tolerance_checks(self):
        """Equal values with different dtypes must not be reported as accepted."""
        for change in ("dtype", "value"):
            with self.subTest(change=change):
                changed = copy.deepcopy(self.capture)
                if change == "dtype":
                    changed["encoder_hidden_states"][0] = changed[
                        "encoder_hidden_states"
                    ][0].float()
                else:
                    changed["encoder_hidden_states"][0][0, 0, 0] += 0.25
                self.save(self.args.output, capture=changed)
                with self.assertRaises(AssertionError):
                    self.compare()
                report = self.read_report()
                self.assertFalse(report["acceptance"])
                self.assertFalse(report["encoder_hidden_states"][0]["within_tolerance"])

    def test_native_baseline_requires_same_source_but_allows_different_topology(self):
        """Native comparisons must not silently cross dirty source revisions."""
        metrics = copy.deepcopy(self.metrics)
        metrics["arguments"].update(mode="native", tp=1, attention="torch_sdpa")
        self.save(self.args.reference, metrics=metrics)
        actual = copy.deepcopy(metrics)
        actual["source"]["repository"] = "/another-native-worktree"
        actual["arguments"].update(tp=2, attention="flash_attn", offload="component")
        self.save(self.args.output, metrics=actual)
        self.assertTrue(self.compare()["acceptance"])
        for key in ("commit", "diff_sha256"):
            with self.subTest(key=key):
                changed = copy.deepcopy(actual)
                changed["source"][key] = "different"
                self.save(self.args.output, metrics=changed)
                with self.assertRaisesRegex(ValueError, f"Native source {key} differs"):
                    self.compare()
                self.assertFalse(self.read_report()["acceptance"])

    def test_distributed_capture_rejects_mixed_rank_sources(self):
        """Rank-local imports can otherwise combine different versions into one result."""
        records = [
            {
                "rank_metrics": {"rank": rank},
                "source": copy.deepcopy(self.metrics["source"]),
            }
            for rank in range(2)
        ]
        self.assertEqual(len(validator.validate_rank_sources(records)), 2)
        for key in ("commit", "diff_sha256"):
            with self.subTest(key=key):
                changed = copy.deepcopy(records)
                changed[1]["source"][key] = "different"
                with self.assertRaisesRegex(ValueError, "Rank 1.*differs"):
                    validator.validate_rank_sources(changed)
        records[1].pop("source")
        with self.assertRaisesRegex(ValueError, "Rank 1.*missing source"):
            validator.validate_rank_sources(records)


if __name__ == "__main__":
    unittest.main()
