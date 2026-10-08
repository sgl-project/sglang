"""Validate classification metadata against the exact saved PEFT artifacts."""

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from safetensors.torch import save_file

from sglang.srt.lora.classification_export import write_classification_manifest
from sglang.srt.lora.classification_head import (
    ClassificationLease,
    load_classification_head,
    prepare_classification_bundle,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WEIGHT_KEY = "base_model.model.score.weight"
BIAS_KEY = "base_model.model.score.bias"


class TestClassificationArtifacts(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.adapter = {
            "peft_type": "LORA",
            "task_type": "SEQ_CLS",
            "r": 2,
            "lora_alpha": 4,
            "target_modules": ["q_proj", "v_proj"],
            # PEFT adds these aliases and may repeat the requested module.
            "modules_to_save": ["score", "classifier", "score"],
        }
        self.tensors = {
            WEIGHT_KEY: torch.tensor([[1.0, 0.0, -1.0], [-1.0, 2.0, 0.5]]),
            BIAS_KEY: torch.tensor([0.25, -0.5]),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(
                2, 3
            ),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(
                3, 2
            ),
        }
        self.manifest = {
            "schema_version": 1,
            "problem_type": "single_label_classification",
            "num_labels": 2,
            "hidden_size": 3,
            "pooling": "last",
            "max_length": 3,
            "add_special_tokens": False,
            "id2label": {"0": "negative", "1": "positive"},
            "head": {"weight_key": WEIGHT_KEY, "bias_key": BIAS_KEY},
        }
        self.write_bundle()

    def write_bundle(self):
        (self.path / "adapter_config.json").write_text(json.dumps(self.adapter))
        save_file(self.tensors, self.path / "adapter_model.safetensors")
        self.manifest["artifacts"] = {
            name: hashlib.sha256((self.path / name).read_bytes()).hexdigest()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
        self.write_manifest()

    def write_manifest(self):
        (self.path / "classification_config.json").write_text(json.dumps(self.manifest))

    def test_saved_head_probabilities_and_bias(self):
        head = load_classification_head(str(self.path), 3)
        hidden = torch.tensor([0.5, 1.0, -0.5])
        expected = (self.tensors[WEIGHT_KEY] @ hidden + self.tensors[BIAS_KEY]).softmax(
            dim=0
        )
        result = head.classify([{"meta_info": {"hidden_states": hidden.tolist()}}])[0]
        torch.testing.assert_close(torch.tensor(result["probs"]), expected)
        self.assertEqual(result["label"], "negative")
        self.assertEqual(result["num_classes"], 2)

    def test_bias_free_classifier_alias(self):
        self.tensors["base_model.model.classifier.weight"] = self.tensors.pop(
            WEIGHT_KEY
        )
        self.tensors.pop(BIAS_KEY)
        self.manifest["head"] = {"weight_key": "base_model.model.classifier.weight"}
        self.write_bundle()
        head = load_classification_head(str(self.path), 3)
        self.assertIsNone(head.bias)
        self.assertEqual(head.labels, ("negative", "positive"))

    def test_generation_adapters_and_remote_identifiers_are_unchanged(self):
        (self.path / "classification_config.json").unlink()
        self.adapter["task_type"] = "CAUSAL_LM"
        (self.path / "adapter_config.json").write_text(json.dumps(self.adapter))
        self.assertIsNone(load_classification_head(str(self.path), 3))
        self.assertIsNone(prepare_classification_bundle(str(self.path), 3))
        self.assertIsNone(load_classification_head(str(self.path / "remote/model"), 3))

    def test_seq_cls_and_legacy_artifacts_cannot_silently_be_generation(self):
        (self.path / "classification_config.json").unlink()
        for loader in (load_classification_head, prepare_classification_bundle):
            with (
                self.subTest(loader=loader.__name__),
                self.assertRaisesRegex(ValueError, "SEQ_CLS"),
            ):
                loader(str(self.path), 3)
        self.adapter["task_type"] = "CAUSAL_LM"
        (self.path / "adapter_config.json").write_text(json.dumps(self.adapter))
        for marker in ("classification_head.pt", "label_mapping.json"):
            (self.path / marker).write_text("legacy")
            with (
                self.subTest(marker=marker),
                self.assertRaisesRegex(ValueError, "offline conversion"),
            ):
                prepare_classification_bundle(str(self.path), 3)
            (self.path / marker).unlink()

    def test_checksum_binds_both_adapter_files(self):
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            self.write_bundle()
            with (self.path / name).open("ab") as stream:
                stream.write(b" ")
            with (
                self.subTest(artifact=name),
                self.assertRaisesRegex(ValueError, "checksum mismatch"),
            ):
                prepare_classification_bundle(str(self.path), 3)

    def test_manifest_requires_exact_artifacts_and_hex_digests(self):
        original = dict(self.manifest["artifacts"])
        cases = [
            None,
            {},
            {"../adapter_config.json": "0" * 64},
            {**original, "adapter_model.bin": "0" * 64},
            {**original, "adapter_config.json": "z" * 64},
        ]
        for artifacts in cases:
            self.manifest["artifacts"] = artifacts
            self.write_manifest()
            with self.subTest(artifacts=artifacts), self.assertRaises(ValueError):
                prepare_classification_bundle(str(self.path), 3)

    def test_invalid_manifest_fields_fail_before_use(self):
        cases = {
            "schema_version": [0, 2, True, "1", None],
            "problem_type": ["multiclass", "regression", None],
            "num_labels": [True, 1, 2.0, "2"],
            "hidden_size": [True, 4, 3.0, "3"],
            "pooling": ["mean", None],
            "max_length": [0, True, 2.0],
            "add_special_tokens": [None, 0, "false"],
            "id2label": [
                ["negative", "positive"],
                {"0": "negative"},
                {"0": "same", "1": "same"},
                {"0": " ", "1": "positive"},
                {"0": 0, "1": 1},
            ],
        }
        for field, values in cases.items():
            original = self.manifest[field]
            for value in values:
                self.manifest[field] = value
                self.write_manifest()
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    load_classification_head(str(self.path), 3)
            self.manifest[field] = original

    def test_invalid_adapter_config_is_a_value_error(self):
        cases = [
            None,
            [],
            {**self.adapter, "task_type": "CAUSAL_LM"},
            {**self.adapter, "peft_type": "IA3"},
            {**self.adapter, "r": True},
            {**self.adapter, "lora_alpha": "4"},
            {**self.adapter, "lora_alpha": 10**400},
            {**self.adapter, "modules_to_save": "score"},
            {**self.adapter, "modules_to_save": ["other"]},
            {**self.adapter, "target_modules": {"q_proj": True}},
        ]
        for config in cases:
            self.adapter = config
            self.write_bundle()
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_classification_head(str(self.path), 3)

    def test_head_keys_cannot_omit_or_select_an_unrelated_tensor(self):
        cases = [
            None,
            {},
            {"weight_key": WEIGHT_KEY},
            {"weight_key": WEIGHT_KEY, "bias_key": None},
            {"weight_key": WEIGHT_KEY, "bias_key": "other.bias"},
            {"weight_key": "base_model.model.classifier.weight"},
            {
                "weight_key": "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
            },
        ]
        for head in cases:
            self.manifest["head"] = head
            self.write_manifest()
            with self.subTest(head=head), self.assertRaises(ValueError):
                load_classification_head(str(self.path), 3)

    def test_unsupported_peft_decoder_variants_are_rejected(self):
        original = dict(self.adapter)
        for field, value in {
            "use_rslora": True,
            "use_dora": True,
            "fan_in_fan_out": True,
            "lora_bias": True,
            "bias": "all",
            "rank_pattern": {"o_proj": 8},
            "alpha_pattern": {"o_proj": 16},
            "layer_replication": [[0, 1]],
            "target_parameters": ["experts.gate_up_proj"],
        }.items():
            self.adapter = {**original, field: value}
            self.write_bundle()
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "standard"),
            ):
                load_classification_head(str(self.path), 3)

    def test_exporter_preserves_peft_weights_and_never_overwrites_manifest(self):
        manifest = self.path / "classification_config.json"
        manifest.unlink()
        original = {
            name: (self.path / name).read_bytes()
            for name in ("adapter_config.json", "adapter_model.safetensors")
        }
        kwargs = dict(
            id2label={0: "negative", 1: "positive"},
            hidden_size=3,
            max_length=8,
            add_special_tokens=True,
        )
        write_classification_manifest(self.path, **kwargs)
        first_manifest = manifest.read_bytes()
        head = load_classification_head(str(self.path), 3)
        self.assertEqual(head.labels, ("negative", "positive"))
        self.assertTrue(head.add_special_tokens)
        self.assertEqual(head.max_length, 8)
        with self.assertRaises(FileExistsError):
            write_classification_manifest(self.path, **kwargs)
        self.assertEqual(manifest.read_bytes(), first_manifest)
        for name, content in original.items():
            self.assertEqual((self.path / name).read_bytes(), content)

    def test_exporter_does_not_publish_invalid_label_mapping(self):
        manifest = self.path / "classification_config.json"
        manifest.unlink()
        for labels in (
            {0: "negative"},
            {0: "a", "0": "b", 1: "c"},
            {False: "a", 1: "b"},
        ):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                write_classification_manifest(
                    self.path,
                    id2label=labels,
                    hidden_size=3,
                    max_length=3,
                    add_special_tokens=False,
                )
            self.assertFalse(manifest.exists())

    def test_explicit_padding_is_rejected_before_last_token_pooling(self):
        head = load_classification_head(str(self.path), 3)
        for ids in ([0, 2], [2, 0], [2, 0, 1]):
            obj = SimpleNamespace(text=None, input_ids=ids, is_single=True)
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "padding"):
                head.prepare_input(obj, None, pad_token_id=0)
        obj = SimpleNamespace(text=None, input_ids=[2, 1, 2, 0], is_single=True)
        head.prepare_input(obj, None, pad_token_id=0)
        self.assertEqual(obj.input_ids, [2, 1, 2])
        tokenizer = mock.Mock(return_value={"input_ids": [[2, 0]]})
        obj = SimpleNamespace(text="text [PAD]", input_ids=None, is_single=True)
        with self.assertRaisesRegex(ValueError, "padding"):
            head.prepare_input(obj, tokenizer, pad_token_id=0)

    def test_ambiguous_saved_heads_are_rejected(self):
        self.tensors["base_model.model.classifier.weight"] = torch.ones(2, 3)
        self.write_bundle()
        with self.assertRaisesRegex(ValueError, "keys do not match"):
            load_classification_head(str(self.path), 3)

    def test_invalid_head_shape_dtype_and_values_are_rejected(self):
        original_weight, original_bias = (
            self.tensors[WEIGHT_KEY],
            self.tensors[BIAS_KEY],
        )
        cases = [
            (WEIGHT_KEY, torch.ones(3, 2)),
            (WEIGHT_KEY, torch.ones(2, 3, dtype=torch.int32)),
            (WEIGHT_KEY, torch.full((2, 3), float("nan"))),
            (BIAS_KEY, torch.ones(2, 1)),
            (BIAS_KEY, torch.ones(2, dtype=torch.float16)),
            (BIAS_KEY, torch.full((2,), float("inf"))),
        ]
        for key, value in cases:
            self.tensors[WEIGHT_KEY], self.tensors[BIAS_KEY] = (
                original_weight,
                original_bias,
            )
            self.tensors[key] = value
            self.write_bundle()
            with (
                self.subTest(key=key, dtype=value.dtype, shape=value.shape),
                self.assertRaises(ValueError),
            ):
                load_classification_head(str(self.path), 3)

    def test_malformed_json_safetensors_and_missing_files_are_value_errors(self):
        for target in (
            "adapter_config.json",
            "adapter_model.safetensors",
            "classification_config.json",
        ):
            self.write_bundle()
            (self.path / target).write_bytes(b"not a valid artifact")
            if target != "classification_config.json":
                self.manifest["artifacts"][target] = hashlib.sha256(
                    (self.path / target).read_bytes()
                ).hexdigest()
                self.write_manifest()
            with self.subTest(target=target), self.assertRaises(ValueError):
                load_classification_head(str(self.path), 3)
        self.write_bundle()
        (self.path / "adapter_model.safetensors").unlink()
        with self.assertRaises(ValueError):
            prepare_classification_bundle(str(self.path), 3)

    def test_duplicate_manifest_keys_are_rejected(self):
        manifest = self.path / "classification_config.json"
        manifest.write_text('{"schema_version": 1, "schema_version": 1}')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            prepare_classification_bundle(str(self.path), 3)

    def test_snapshot_excludes_unbound_files_and_survives_source_replacement(self):
        (self.path / "adapter_model.bin").write_bytes(b"must not override safetensors")
        bundle = prepare_classification_bundle(str(self.path), 3)
        self.addCleanup(bundle.directory.cleanup)
        snapshot = Path(bundle.path)
        original_weights = bundle.head.weight.clone()
        self.assertEqual(
            {p.name for p in snapshot.iterdir()},
            {
                "adapter_config.json",
                "adapter_model.safetensors",
                "classification_config.json",
            },
        )
        self.tensors[WEIGHT_KEY] = torch.zeros(2, 3)
        self.manifest["id2label"] = {"0": "new-negative", "1": "new-positive"}
        self.write_bundle()
        torch.testing.assert_close(bundle.head.weight, original_weights)
        reloaded = load_classification_head(bundle.path, 3)
        torch.testing.assert_close(reloaded.weight, original_weights)
        self.assertEqual(reloaded.labels, ("negative", "positive"))
        self.assertEqual(
            load_classification_head(str(self.path), 3).labels,
            ("new-negative", "new-positive"),
        )

    def test_tokenizer_extensions_are_rejected(self):
        (self.path / "tokenizer.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "tokenizer/vocabulary"):
            prepare_classification_bundle(str(self.path), 3)

    def test_text_input_uses_lazy_private_tokenizer_and_right_truncation(self):
        class Tokenizer:
            def __init__(self):
                self.calls = 0

            def __call__(self, texts, **kwargs):
                self.calls += 1
                return {"input_ids": [[1, 2, 3, 4] for _ in texts]}

        manager = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=SimpleNamespace(pad_token_id=None)),
            tokenizer=Tokenizer(),
            classification_tokenizer=None,
            classification_tokenizer_lock=threading.Lock(),
        )
        lease = ClassificationLease(manager)
        lease.head = load_classification_head(str(self.path), 3)
        ids = SimpleNamespace(text=None, input_ids=[7, 8, 9, 10], is_single=True)
        with mock.patch(
            "sglang.srt.lora.classification_head.copy.deepcopy",
            side_effect=AssertionError("eager copy"),
        ):
            lease._prepare_input(ids)
        self.assertEqual(ids.input_ids, [7, 8, 9])
        self.assertIsNone(manager.classification_tokenizer)
        for _ in range(2):
            text = SimpleNamespace(text="input", input_ids=None, is_single=True)
            lease._prepare_input(text)
            self.assertEqual(text.input_ids, [1, 2, 3])
            self.assertIsNone(text.text)
        self.assertEqual(manager.tokenizer.calls, 0)
        self.assertEqual(manager.classification_tokenizer.calls, 2)


if __name__ == "__main__":
    unittest.main()
