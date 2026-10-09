"""Consolidated training exports own their data and contain only draft weights."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from sglang.srt.speculative.dspark_components import dspark_target_kv_export as exporter
from sglang.srt.training_capture.protocol import ContractError, digest_bytes
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dspark_target_kv_utils import make_target_kv_config
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTargetKVExport(CustomTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "export"
        self.fixture = self.root / "inputs.safetensors"
        save_file({"token_ids": torch.arange(8)}, self.fixture)
        self.config = make_target_kv_config()
        self.config.update(
            model_type="qwen3",
            num_hidden_layers=1,
            intermediate_size=12,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            vocab_size=256,
            block_size=3,
            mask_token_id=255,
            markov_rank=3,
            markov_head_type="vanilla",
            dtype="bfloat16",
        )
        self.config["target_kv_contract"]["validation"]["golden_fixture_sha256"] = (
            digest_bytes(self.fixture.read_bytes())
        )
        self.weights = {
            "kv_encoder.projection.weight": torch.randn(8, 32),
            "kv_encoder.norm_weight": torch.ones(8),
            "norm.weight": torch.ones(8),
            "markov_head.markov_w1.weight": torch.randn(256, 3),
            "markov_head.markov_w2.weight": torch.randn(256, 3),
            "layers.0.self_attn.q_proj.weight": torch.randn(8, 8),
            "layers.0.self_attn.k_proj.weight": torch.randn(4, 8),
            "layers.0.self_attn.v_proj.weight": torch.randn(4, 8),
            "layers.0.self_attn.o_proj.weight": torch.randn(8, 8),
            "layers.0.self_attn.q_norm.weight": torch.ones(2),
            "layers.0.self_attn.k_norm.weight": torch.ones(2),
            "layers.0.input_layernorm.weight": torch.ones(8),
            "layers.0.post_attention_layernorm.weight": torch.ones(8),
            "layers.0.mlp.gate_proj.weight": torch.randn(12, 8),
            "layers.0.mlp.up_proj.weight": torch.randn(12, 8),
            "layers.0.mlp.down_proj.weight": torch.randn(8, 12),
        }

    def export(self, weights=None, **kwargs):
        return exporter.export_target_kv_checkpoint(
            self.config,
            self.weights if weights is None else weights,
            golden_fixture=self.fixture,
            output_dir=self.output,
            **kwargs,
        )

    def test_split_export_preserves_inputs_and_requires_new_parity(self):
        before = copy.deepcopy(self.config)
        receipt = self.export()
        self.assertEqual(self.config, before)
        self.assertTrue(receipt["requires_fixed_input_parity"])
        self.assertFalse((self.output / "validation/parity.json").exists())
        loaded = load_file(self.output / "model.safetensors")
        self.assertEqual(loaded.keys(), self.weights.keys())
        for name, value in loaded.items():
            self.assertEqual(value.dtype, torch.bfloat16)
            self.assertTrue(torch.equal(value, self.weights[name].bfloat16()))
            self.assertEqual(self.weights[name].dtype, torch.float32)
        for name, digest in receipt["artifact_sha256"].items():
            self.assertEqual(digest_bytes((self.output / name).read_bytes()), digest)
        self.assertEqual(
            json.loads((self.output / "export.json").read_bytes()), receipt
        )
        self.assertEqual(
            (self.output / "validation/inputs.safetensors").read_bytes(),
            self.fixture.read_bytes(),
        )

    def test_packed_gqa_and_mlp_match_split_values(self):
        packed = dict(self.weights)
        for prefix, parts, fused in (
            ("layers.0.self_attn.", ("q", "k", "v"), "qkv"),
            ("layers.0.mlp.", ("gate", "up"), "gate_up"),
        ):
            packed[prefix + fused + "_proj.weight"] = torch.cat(
                [packed.pop(prefix + part + "_proj.weight") for part in parts]
            )
        self.export({"model." + name: value for name, value in packed.items()})
        loaded = load_file(self.output / "model.safetensors")
        for name, value in self.weights.items():
            self.assertTrue(torch.equal(loaded[name], value.bfloat16()))

    def test_gated_rnn_and_attention_bias_parameters_are_not_mlp_aliases(self):
        for kind, extra in (
            (
                "gated",
                {
                    "gate_proj.weight": torch.randn(3, 11),
                    "gate_proj.bias": torch.randn(3),
                },
            ),
            (
                "rnn",
                {
                    "joint_proj.weight": torch.randn(9, 14),
                    "joint_proj.bias": torch.randn(9),
                },
            ),
        ):
            with self.subTest(kind=kind):
                self.output = self.root / kind
                self.config.update(markov_head_type=kind, attention_bias=True)
                weights = self.weights | {
                    "markov_head." + k: v for k, v in extra.items()
                }
                bias = torch.arange(16, dtype=torch.float32)
                weights["layers.0.self_attn.qkv_proj.bias"] = bias
                weights["layers.0.self_attn.o_proj.bias"] = torch.zeros(8)
                self.export(weights)
                loaded = load_file(self.output / "model.safetensors")
                self.assertTrue(
                    torch.equal(
                        loaded["layers.0.self_attn.k_proj.bias"], bias[8:12].bfloat16()
                    )
                )
                for name, value in extra.items():
                    self.assertTrue(
                        torch.equal(loaded["markov_head." + name], value.bfloat16())
                    )

    def test_missing_unknown_duplicate_and_mixed_weights_fail_before_publication(self):
        missing = dict(self.weights)
        missing.pop("norm.weight")
        mixed = self.weights | {
            "layers.0.self_attn.qkv_proj.weight": torch.randn(16, 8)
        }
        variants = [
            missing,
            mixed,
            list(self.weights.items()) + [("model.norm.weight", torch.ones(8))],
        ]
        for name in (
            "fc.weight",
            "hidden_norm.weight",
            "embed_tokens.weight",
            "lm_head.weight",
            "confidence_head.weight",
        ):
            variants.append(self.weights | {name: torch.ones(8)})
        for weights in variants:
            with self.subTest(weights=list(dict(weights))), self.assertRaises(
                ContractError
            ):
                self.export(weights)
            self.assertFalse(self.output.exists())

    def test_shapes_nonfinite_and_destination_overflow_are_rejected(self):
        for tensor in (
            torch.ones(9),
            torch.ones(8, dtype=torch.int64),
            torch.empty(8, device="meta"),
            torch.full((8,), float("nan")),
            torch.full((8,), float("inf")),
        ):
            with self.subTest(dtype=tensor.dtype), self.assertRaises(ContractError):
                self.export(self.weights | {"norm.weight": tensor})
            self.assertFalse(self.output.exists())
        self.config["dtype"] = "float16"
        with self.assertRaisesRegex(ContractError, "nonfinite"):
            self.export(self.weights | {"norm.weight": torch.full((8,), 1e20)})
        self.assertFalse(self.output.exists())

    def test_alias_and_dtype_conflicts_are_rejected(self):
        before = copy.deepcopy(self.config)
        for change in (
            {"model_type": "unregistered_kv_draft"},
            {"model_type": None},
            {"torch_dtype": "float16"},
            {"dspark_config": {"markov_rank": 7}},
            {"dflash_config": {"block_size": 4}},
            {"mask_token_id": 254},
            {"quantization_config": {}},
        ):
            self.config = before | change
            with self.subTest(change=change), self.assertRaises(ContractError):
                self.export()
            self.assertFalse(self.output.exists())

    def test_fixture_failure_and_write_failure_leave_no_output(self):
        self.fixture.write_bytes(b"changed")
        with self.assertRaisesRegex(ContractError, "golden fixture"):
            self.export()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".export.export-*")))
        save_file({"token_ids": torch.arange(8)}, self.fixture)
        with patch.object(
            exporter, "save_file", side_effect=OSError("full disk")
        ), self.assertRaises(OSError):
            self.export()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob(".export.export-*")))

    def test_existing_destination_is_preserved(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.export()
        sentinel = self.output / "keep"
        sentinel.write_text("existing")
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(sentinel.read_text(), "existing")

    def test_acceptance_requires_a_matching_pinned_artifact(self):
        acceptance = self.root / "acceptance.json"
        acceptance.write_text('{"quality": "fixture"}')
        with self.assertRaisesRegex(ContractError, "supplied together"):
            self.export(acceptance_report=acceptance)
        self.config["target_kv_contract"]["validation"]["acceptance_report_sha256"] = (
            digest_bytes(acceptance.read_bytes())
        )
        with self.assertRaisesRegex(ContractError, "supplied together"):
            self.export()
        self.export(acceptance_report=acceptance)
        self.assertEqual(
            (self.output / "validation/acceptance.json").read_bytes(),
            acceptance.read_bytes(),
        )

    def test_cli_uses_safetensors_without_unpickling_training_state(self):
        config_path, weights_path = (
            self.root / "config.json",
            self.root / "training.safetensors",
        )
        config_path.write_text(json.dumps(self.config))
        save_file(self.weights, weights_path)
        args = [
            "--config",
            str(config_path),
            "--weights",
            str(weights_path),
            "--golden-fixture",
            str(self.fixture),
            "--output-dir",
            str(self.output),
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(exporter.main(args), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "exported")
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(exporter.main(args), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "failed")


if __name__ == "__main__":
    unittest.main()
