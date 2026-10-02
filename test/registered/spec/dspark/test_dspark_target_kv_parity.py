"""Compare the serving KV-input draft against the installed SpecForge source."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from sglang.srt.speculative.dspark_components.dspark_target_kv_export import (
    export_target_kv_checkpoint,
)
from sglang.srt.training_capture.protocol import ContractError, digest_bytes
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.dspark_target_kv_parity import (
    FixedInputParityError,
    ServingKVParityRunner,
    check_fixed_input_parity,
    compare_parity_outputs,
    make_parity_checkpoint,
    single_gpu_parity_context,
    validate_captured_checkpoint,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=40,
    stage="base-b",
    runner_config="1-gpu-small",
    disabled="requires the pinned SpecForge source checkout on PYTHONPATH",
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestTargetKVServingParity(CustomTestCase):
    def export_checkpoint(self, config, weights, tensors, directory):
        fixture = directory.with_suffix(".inputs.safetensors")
        save_file(
            {name: value.contiguous() for name, value in tensors.items()}, fixture
        )
        config = copy.deepcopy(config.to_dict())
        config["target_kv_contract"]["validation"]["golden_fixture_sha256"] = (
            digest_bytes(fixture.read_bytes())
        )
        return export_target_kv_checkpoint(
            config, weights, golden_fixture=fixture, output_dir=directory
        )

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if importlib.util.find_spec("specforge") is None:
            raise unittest.SkipTest("the pinned SpecForge checkout is required")
        cls.runtime = single_gpu_parity_context()
        cls.runtime.__enter__()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "runtime"):
            cls.runtime.__exit__(None, None, None)
        super().tearDownClass()

    def test_backbone_logits_and_training_gradients(self):
        for head_type in ("vanilla", "gated", "rnn"):
            with (
                self.subTest(head_type=head_type),
                tempfile.TemporaryDirectory() as temp,
            ):
                reference, embed, head, tensors = make_parity_checkpoint(
                    Path(temp) / "draft", head_type
                )
                reference, embed, head = (
                    module.cuda() for module in (reference, embed, head)
                )
                serving = ServingKVParityRunner(
                    Path(temp) / "draft", embed=embed, head=head
                )
                report = check_fixed_input_parity(
                    reference, serving, embed, head, tensors, [1, 5, 12]
                )
                self.assertEqual(report["valid_labels"], 7)
                print(json.dumps({"head": head_type, **report}), flush=True)
                before = (
                    reference.backbone.kv_encoder.projection.weight.detach().clone()
                )
                optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3)
                optimizer.step()
                self.assertFalse(
                    torch.equal(before, reference.backbone.kv_encoder.projection.weight)
                )
                updated_path = Path(temp) / "updated"
                self.export_checkpoint(
                    reference.backbone.config,
                    reference.backbone.state_dict(),
                    tensors,
                    updated_path,
                )
                updated = ServingKVParityRunner(updated_path, embed=embed, head=head)
                report = check_fixed_input_parity(
                    reference, updated, embed, head, tensors, [1, 5, 12]
                )
                print(
                    json.dumps(
                        {"head": head_type, "after_optimizer_step": True, **report}
                    ),
                    flush=True,
                )

    def test_packed_serving_weights_export_to_the_same_model(self):
        with tempfile.TemporaryDirectory() as temp:
            reference, embed, head, tensors = make_parity_checkpoint(
                Path(temp) / "draft", "gated"
            )
            reference, embed, head = (
                module.cuda() for module in (reference, embed, head)
            )
            serving = ServingKVParityRunner(
                Path(temp) / "draft", embed=embed, head=head
            )
            weights = {
                name: value
                for name, value in serving.model.named_parameters()
                if not name.startswith(("embed_tokens.", "lm_head."))
            }
            exported = Path(temp) / "packed-export"
            self.export_checkpoint(
                reference.backbone.config, weights, tensors, exported
            )
            reloaded = ServingKVParityRunner(exported, embed=embed, head=head)
            report = check_fixed_input_parity(
                reference, reloaded, embed, head, tensors, [1, 5, 12]
            )
            self.assertTrue(all(stage["passed"] for stage in report["stages"].values()))

    def test_future_tokens_and_kv_do_not_enter_backbone(self):
        with tempfile.TemporaryDirectory() as temp:
            reference, embed, head, tensors = make_parity_checkpoint(
                Path(temp) / "draft", "rnn"
            )
            reference, embed, head = (
                module.cuda() for module in (reference, embed, head)
            )
            serving = ServingKVParityRunner(
                Path(temp) / "draft", embed=embed, head=head
            )
            previous = tensors["token_ids"][5:8].long().cuda()[None]
            wanted = serving.forward(
                tensors=tensors, anchors=[5], previous_tokens=previous
            )
            changed = {name: value.clone() for name, value in tensors.items()}
            changed["token_ids"][6:] = 222
            for name, value in changed.items():
                if name.startswith("target_"):
                    value[5:] = float("nan")
            actual = serving.forward(
                tensors=changed, anchors=[5], previous_tokens=previous
            )
            for name in ("hidden", "base", "corrected"):
                torch.testing.assert_close(actual[name], wanted[name], rtol=0, atol=0)
            with torch.no_grad():
                expected = reference(
                    tensors=tensors,
                    anchor=5,
                    embed=embed,
                    head=head,
                    previous_tokens=previous,
                )
                observed = reference(
                    tensors=changed,
                    anchor=5,
                    embed=embed,
                    head=head,
                    previous_tokens=previous,
                )
            torch.testing.assert_close(
                observed["hidden"], expected["hidden"], rtol=0, atol=0
            )
            changed["target_v.3"][:5] = 50
            altered = serving.forward(
                tensors=changed, anchors=[5], previous_tokens=previous
            )
            self.assertGreater(
                (altered["hidden"].float() - wanted["hidden"].float())
                .abs()
                .max()
                .item(),
                0.03,
            )

    def test_serving_fused_stacked_and_individual_paths_agree(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "draft"
            _, embed, head, tensors = make_parity_checkpoint(directory, "rnn")
            serving = ServingKVParityRunner(
                directory, embed=embed.cuda(), head=head.cuda()
            )
            model = serving.model
            previous = torch.zeros(3, 3, device="cuda", dtype=torch.long)
            inputs = {
                "tensors": tensors,
                "anchors": [1, 5, 12],
                "previous_tokens": previous,
            }
            self.assertIsNotNone(model._fused_kv_write_bundle(serving.pool))
            fused = serving.forward(**inputs)
            expected = [
                {
                    name: (
                        [layer[row : row + 1] for layer in value]
                        if name == "layers"
                        else value[row : row + 1]
                    )
                    for name, value in fused.items()
                }
                for row in range(3)
            ]
            with patch.object(model, "_fused_kv_write_bundle", return_value=None):
                self.assertIsNotNone(model._stacked_ctx_kv_params())
                stacked = serving.forward(**inputs)
                compare_parity_outputs(stacked, expected, rtol=0.03, atol=0.03)
                with patch.object(model, "_stacked_ctx_kv_params", return_value=None):
                    for layer in model.layers:
                        layer.self_attn.use_table_qk_norm_rope = False
                    individual = serving.forward(**inputs)
                compare_parity_outputs(individual, expected, rtol=0.03, atol=0.03)

    def test_all_failing_stages_and_nonfinite_values_are_reported(self):
        expected = {
            "layers": [torch.zeros(1, 2), torch.zeros(1, 2)],
            "hidden": torch.zeros(1, 2),
            "base": torch.zeros(1, 3),
            "corrected": torch.zeros(1, 3),
        }
        actual = {
            "layers": [torch.zeros(1, 2), torch.ones(1, 2)],
            "hidden": torch.tensor([[float("nan"), 0]]),
            "base": torch.ones(1, 3),
            "corrected": torch.ones(1, 3),
        }
        with self.assertRaises(FixedInputParityError) as caught:
            compare_parity_outputs(actual, [expected], rtol=0.03, atol=0.03)
        stages = caught.exception.report["stages"]
        self.assertTrue(stages["layer.0"]["passed"])
        self.assertEqual(stages["hidden"]["nonfinite"], 1)
        for name in ("layer.1", "hidden", "base", "corrected"):
            self.assertFalse(stages[name]["passed"])
        json.dumps(caught.exception.report, allow_nan=False)

    def test_failed_identity_check_invalidates_previous_report(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "draft"
            _, _, _, tensors = make_parity_checkpoint(directory, "vanilla")
            validation = directory / "validation"
            validation.mkdir()
            save_file(tensors, str(validation / "inputs.safetensors"))
            report_path = validation / "parity.json"
            report_path.write_text(json.dumps({"status": "passed"}))
            invalid_target = Path(temp) / "target-without-weights"
            invalid_target.mkdir()
            with self.assertRaises(ContractError):
                validate_captured_checkpoint(directory, invalid_target)
            self.assertEqual(json.loads(report_path.read_text())["status"], "failed")
            (validation / "inputs.safetensors").unlink()
            with self.assertRaises(FileNotFoundError):
                validate_captured_checkpoint(directory, invalid_target)
            self.assertFalse(report_path.exists())


if __name__ == "__main__":
    unittest.main()
