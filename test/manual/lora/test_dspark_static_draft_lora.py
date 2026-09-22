import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

_SOURCE = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/speculative/dspark_components/dspark_lora.py"
)
_SPEC = importlib.util.spec_from_file_location("dspark_lora", _SOURCE)
lora = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(lora)


class TestStaticDraftLoRA(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        self.config = {
            "peft_type": "LORA",
            "r": 2,
            "lora_alpha": 6,
            "bias": "none",
            "target_modules": ["q_proj"],
        }
        self.a = torch.tensor([[1.0, 2, -1], [-2, 0.5, 1]])
        self.b = torch.tensor([[0.5, 1], [2, -1], [1, 3], [-1, 0.5]])
        self.module = "layers.0.self_attn.q_proj"
        self.tensors = self.pair(self.module)
        self.parameter_names = ["layers.0.self_attn.qkv_proj.weight"]
        self.weight = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10

    def pair(self, module, prefix="base_model.model."):
        return {
            f"{prefix}{module}.lora_A.weight": self.a.clone(),
            f"{prefix}{module}.lora_B.weight": self.b.clone(),
        }

    def write_adapter(self):
        (self.path / "adapter_config.json").write_text(json.dumps(self.config))
        save_file(self.tensors, str(self.path / "adapter_model.safetensors"))

    def merge(self, weights=None, parameter_names=None):
        self.write_adapter()
        return list(
            lora.merge_dspark_lora_weights(
                weights
                if weights is not None
                else [(self.module + ".weight", self.weight)],
                str(self.path),
                model_parameter_names=(
                    self.parameter_names if parameter_names is None else parameter_names
                ),
            )
        )

    def test_fp32_merge_matches_unmerged_lora_forward_and_preserves_input(self):
        before = self.weight.clone()
        merged = self.merge()[0][1]
        x = torch.tensor([[2.0, -1, 0.25], [1, 1, 1]])
        expected = x @ self.weight.T + 3 * (x @ self.a.T @ self.b.T)
        torch.testing.assert_close(x @ merged.T, expected)
        torch.testing.assert_close(self.weight, before, rtol=0, atol=0)
        self.assertNotEqual(merged.data_ptr(), self.weight.data_ptr())

    def test_bf16_and_fp16_round_once_after_fp32_merge(self):
        for dtype in [torch.bfloat16, torch.float16]:
            with self.subTest(dtype=dtype):
                self.weight = self.weight.to(dtype)
                actual = self.merge()[0][1]
                expected = (self.weight.float() + 3 * self.b @ self.a).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, dtype)

    def test_peft_and_checkpoint_prefix_variants(self):
        for adapter_prefix in [
            "",
            "model.",
            "base_model.model.",
            "base_model.model.model.",
        ]:
            for checkpoint_prefix in ["", "model."]:
                with self.subTest(
                    adapter_prefix=adapter_prefix, checkpoint_prefix=checkpoint_prefix
                ):
                    self.tensors = self.pair(self.module, adapter_prefix)
                    name = checkpoint_prefix + self.module + ".weight"
                    result = self.merge([(name, self.weight)])
                    self.assertEqual(result[0][0], name)
                    torch.testing.assert_close(
                        result[0][1], self.weight + 3 * self.b @ self.a
                    )

    def test_qkv_and_gate_up_packing_then_tp_slicing_preserves_updates(self):
        for projections, packed in [
            (["q_proj", "k_proj", "v_proj"], "self_attn.qkv_proj"),
            (["gate_proj", "up_proj"], "mlp.gate_up_proj"),
        ]:
            with self.subTest(projections=projections):
                kind = packed.split(".")[0]
                self.config["target_modules"] = projections
                weights = []
                self.tensors = {}
                for index, projection in enumerate(projections):
                    name = f"layers.0.{kind}.{projection}"
                    self.tensors.update(self.pair(name))
                    weights.append((name + ".weight", self.weight + index))
                merged = self.merge(weights, [f"layers.0.{packed}.weight"])
                for rank in [0, 1]:
                    actual = torch.cat([w.chunk(2, dim=0)[rank] for _, w in merged])
                    expected = torch.cat(
                        [
                            (w + 3 * self.b @ self.a).chunk(2, dim=0)[rank]
                            for _, w in weights
                        ]
                    )
                    torch.testing.assert_close(actual, expected)

    def test_context_projector_fc_is_supported(self):
        self.config["target_modules"] = ["fc"]
        self.tensors = self.pair("fc")
        merged = self.merge([("fc.weight", self.weight)], ["fc.weight"])
        torch.testing.assert_close(merged[0][1], self.weight + 3 * self.b @ self.a)

    def test_unadapted_target_shared_tensors_are_unchanged(self):
        target_embedding = torch.randn(7, 3)
        target_head = torch.randn(7, 3)
        embedding_before, head_before = target_embedding.clone(), target_head.clone()
        merged = dict(
            self.merge(
                [
                    ("embed_tokens.weight", target_embedding),
                    (self.module + ".weight", self.weight),
                    ("lm_head.weight", target_head),
                ]
            )
        )
        self.assertIs(merged["embed_tokens.weight"], target_embedding)
        self.assertIs(merged["lm_head.weight"], target_head)
        torch.testing.assert_close(target_embedding, embedding_before, atol=0, rtol=0)
        torch.testing.assert_close(target_head, head_before, atol=0, rtol=0)

    def test_rejects_shared_heads_and_non_backbone_modules(self):
        for module in ["lm_head", "embed_tokens", "markov_head.markov_w2", "norm"]:
            with self.subTest(module=module):
                self.config["target_modules"] = [module.split(".")[-1]]
                self.tensors = self.pair(module)
                with self.assertRaisesRegex(ValueError, "target_modules"):
                    self.merge()

    def test_rejects_unsupported_peft_semantics(self):
        options = {
            "fan_in_fan_out": True,
            "use_dora": True,
            "use_rslora": True,
            "use_qalora": True,
            "lora_bias": True,
            "rank_pattern": {"q_proj": 1},
            "alpha_pattern": {"q_proj": 1},
            "modules_to_save": ["lm_head"],
            "target_parameters": ["weight"],
            "layer_replication": [[0, 1]],
            "trainable_token_indices": [5],
            "alora_invocation_tokens": [42],
            "arrow_config": {"top_k": 3},
            "ensure_weight_tying": True,
            "megatron_config": {"tensor_model_parallel_size": 2},
            "loftq_config": {"loftq_bits": 4},
            "eva_config": {"rho": 2},
            "corda_config": {"corda_method": "ipm"},
        }
        for option, value in options.items():
            with self.subTest(option=option):
                self.config[option] = value
                with self.assertRaisesRegex(ValueError, option):
                    self.merge()
                del self.config[option]
        for changes in [
            {"peft_type": "IA3"},
            {"bias": "all"},
            {"target_modules": "all-linear"},
            {"r": 0},
            {"r": True},
            {"lora_alpha": float("nan")},
            {"lora_alpha": 0},
            {"init_lora_weights": "pissa"},
        ]:
            old = self.config.copy()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.config.update(changes)
                self.merge()
            self.config = old

    def test_rejects_missing_pair_and_extra_tensors(self):
        del self.tensors[next(iter(self.tensors))]
        with self.assertRaisesRegex(ValueError, "Missing LoRA"):
            self.merge()
        self.tensors = self.pair(self.module)
        self.tensors["base_model.model.lm_head.weight"] = self.weight
        with self.assertRaisesRegex(
            ValueError, "Unsupported DSpark draft adapter tensor"
        ):
            self.merge()

    def test_rejects_duplicate_aliases_and_checkpoint_weights(self):
        self.tensors.update(self.pair(self.module, "model."))
        with self.assertRaisesRegex(ValueError, "Duplicate DSpark draft adapter"):
            self.merge()
        self.tensors = self.pair(self.module)
        weights = [(self.module + ".weight", self.weight)] * 2
        with self.assertRaisesRegex(ValueError, "Duplicate draft checkpoint"):
            self.merge(weights)

    def test_rejects_rank_shape_quantization_and_nonfinite(self):
        for case in ["rank", "shape", "int8", "nan", "overflow"]:
            with self.subTest(case=case):
                self.setUp()
                if case == "rank":
                    self.config["r"] = 4
                elif case == "shape":
                    self.weight = torch.ones(8, 3)
                elif case == "int8":
                    self.weight = self.weight.to(torch.int8)
                elif case == "nan":
                    self.tensors[next(iter(self.tensors))][0, 0] = float("nan")
                else:
                    self.weight = self.weight.to(torch.float16)
                    self.config["lora_alpha"] = 1e10
                with self.assertRaises(ValueError):
                    self.merge()

    def test_rejects_unmatched_or_unloaded_parameters(self):
        with self.assertRaisesRegex(ValueError, "not found in checkpoint"):
            self.merge([])
        with self.assertRaisesRegex(ValueError, "absent from the DSpark draft model"):
            self.merge(parameter_names=[])
        with self.assertRaisesRegex(ValueError, "not found in checkpoint"):
            self.merge([("layers.0.self_attn.qkv_proj.weight", torch.ones(12, 3))])

    def test_rejects_undeclared_module_empty_adapter_and_remote_path(self):
        self.config["target_modules"] = ["k_proj"]
        with self.assertRaisesRegex(ValueError, "not declared"):
            self.merge()
        self.config["target_modules"] = ["q_proj"]
        self.tensors = {}
        with self.assertRaisesRegex(ValueError, "no LoRA"):
            self.merge()
        with self.assertRaisesRegex(ValueError, "local adapter directory"):
            list(
                lora.merge_dspark_lora_weights(
                    [], "org/not-a-local-model", model_parameter_names=[]
                )
            )

    def test_argument_gates(self):
        for algorithm in [None, "EAGLE3", "DFLASH", "UNO"]:
            with self.assertRaisesRegex(ValueError, "requires DSPARK"):
                lora.validate_dspark_lora_args(
                    algorithm=algorithm, draft_load_format="auto"
                )
        for load_format in [
            "dummy",
            "sharded_state",
            "bitsandbytes",
            "fastsafetensors",
        ]:
            with self.assertRaisesRegex(ValueError, "checkpoint loader"):
                lora.validate_dspark_lora_args(
                    algorithm="DSPARK", draft_load_format=load_format
                )
        for load_format in ["auto", "safetensors", "pt"]:
            lora.validate_dspark_lora_args(
                algorithm="dspark", draft_load_format=load_format
            )


if __name__ == "__main__":
    unittest.main()
