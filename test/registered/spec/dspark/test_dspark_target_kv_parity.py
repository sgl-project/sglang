"""Compare the serving KV-input draft against the installed SpecForge source."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from sglang.srt.training_capture.protocol import ContractError
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
                reference.backbone.config.save_pretrained(updated_path)
                save_file(
                    {
                        name: value.detach().cpu().contiguous()
                        for name, value in reference.backbone.state_dict().items()
                    },
                    str(updated_path / "model.safetensors"),
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
