"""Regression tests for Phi quantization prefixes (issue #37848).

phi.py built its quantization-aware layers without a prefix, so the prefix
defaulted to "" -- a name matching no checkpoint submodule. Per-module decisions
keyed on that name (ignored_layers for FP8, exclude_modules for ModelOpt FP8)
were silently dropped and the layer stayed quantized. These tests build a real
tiny PhiForCausalLM with the real FP8 configs and assert each quantizable
layer's resolved quant_method.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

import contextlib
import socket
import unittest

import torch
from transformers import PhiConfig

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
    model_parallel_is_initialized,
)
from sglang.srt.environ import envs
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp8Config
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.models.phi import PhiForCausalLM
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

# A safetensors checkpoint stores the unfused qkv shards, never the fused
# qkv_proj; is_layer_skipped rewrites qkv_proj to these before matching.
QKV_SHARDS = ["q_proj", "k_proj", "v_proj"]

# Fp8Config defaults to the shared fallback mapping while the model declares the
# same mapping explicitly; both must resolve the fused qkv_proj identically.
PACKED_MAPPING_VARIANTS = {
    "fallback_mapping": {},
    "model_mapping": PhiForCausalLM.packed_modules_mapping,
}

LAYER_LINEARS = [
    "self_attn.qkv_proj",
    "self_attn.dense",
    "mlp.fc1",
    "mlp.fc2",
]


def phi_config() -> PhiConfig:
    # intermediate_size matches the 4 * hidden_size PhiMLP actually uses.
    return PhiConfig(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        max_position_embeddings=64,
        partial_rotary_factor=0.5,
    )


def linear_prefix(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


class TestPhiQuantizationPrefix(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # Registered first so the environment is restored even if setup below
        # fails part-way through.
        cls._cleanup = contextlib.ExitStack()
        cls.addClassCleanup(cls._cleanup.close)
        # An inherited value would be folded into every Fp8Config built here.
        cls._cleanup.enter_context(envs.SGLANG_FP8_IGNORED_LAYERS.override(""))
        # The srt layers read a published tp=1 config; this is scoped to the
        # class and never touches the global scheduler args.
        cls._cleanup.enter_context(get_context().override_server_args(tp_size=1))
        # Quantized parallel layers need real process groups even at tp=1.
        # Reuse whatever this worker already has; only tear down own creations.
        if not torch.distributed.is_initialized():
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend="gloo",
            )
            cls.addClassCleanup(destroy_distributed_environment)
        if not model_parallel_is_initialized():
            initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
            cls.addClassCleanup(destroy_model_parallel)

    def methods(self, quant_config) -> dict:
        """Map qualified module name -> resolved quant_method class name."""
        model = PhiForCausalLM(phi_config(), quant_config=quant_config)
        return {
            name: type(module.quant_method).__name__
            for name, module in model.named_modules()
            if isinstance(module, (LinearBase, ParallelLMHead))
        }

    def assert_layer_methods(self, methods: dict, layer_idx: int, expected: str):
        for suffix in LAYER_LINEARS:
            name = linear_prefix(layer_idx, suffix)
            with self.subTest(layer=name):
                self.assertEqual(methods[name], expected)

    def test_unquantized_baseline(self):
        methods = self.methods(None)
        self.assert_layer_methods(methods, 0, "UnquantizedLinearMethod")
        self.assert_layer_methods(methods, 1, "UnquantizedLinearMethod")
        self.assertEqual(methods["lm_head"], "UnquantizedEmbeddingMethod")

    def test_fp8_without_ignores_quantizes_every_linear(self):
        methods = self.methods(Fp8Config())
        self.assert_layer_methods(methods, 0, "Fp8LinearMethod")
        self.assert_layer_methods(methods, 1, "Fp8LinearMethod")

    def test_fp8_ignored_layers_by_qualified_name(self):
        # Core regression: these are the names a checkpoint's
        # modules_to_not_convert actually lists (the unfused QKV shards plus the
        # dense/MLP projections), and each must bypass FP8.
        ignored = [linear_prefix(1, f"self_attn.{shard}") for shard in QKV_SHARDS]
        ignored += [
            linear_prefix(1, "self_attn.dense"),
            linear_prefix(1, "mlp.fc1"),
            linear_prefix(1, "mlp.fc2"),
        ]
        methods = self.methods(Fp8Config(ignored_layers=ignored))

        self.assert_layer_methods(methods, 1, "UnquantizedLinearMethod")
        # Partial ignore: the untouched layer stays quantized.
        self.assert_layer_methods(methods, 0, "Fp8LinearMethod")

    def test_fp8_from_config_normalizes_bare_and_prefixed_names(self):
        # Fp8Config.from_config is the real checkpoint path; it keeps both the
        # bare and the "model."-prefixed variant of each entry.
        for entry in ("model.layers.1.mlp.fc1", "layers.1.mlp.fc1"):
            with self.subTest(entry=entry):
                config = Fp8Config.from_config(
                    {
                        "quant_method": "fp8",
                        "activation_scheme": "dynamic",
                        "ignored_layers": [entry],
                    }
                )
                methods = self.methods(config)
                self.assertEqual(
                    methods["model.layers.1.mlp.fc1"], "UnquantizedLinearMethod"
                )
                self.assertEqual(methods["model.layers.1.mlp.fc2"], "Fp8LinearMethod")

    def test_fused_qkv_is_skipped_only_through_all_shards(self):
        # The fused qkv_proj is skipped only when every shard it packs is
        # ignored; both mapping variants must agree.
        for label, mapping in PACKED_MAPPING_VARIANTS.items():
            with self.subTest(packed_mapping=label):
                ignored = [
                    linear_prefix(0, f"self_attn.{shard}") for shard in QKV_SHARDS
                ]
                methods = self.methods(
                    Fp8Config(ignored_layers=ignored, packed_modules_mapping=mapping)
                )
                self.assertEqual(
                    methods["model.layers.0.self_attn.qkv_proj"],
                    "UnquantizedLinearMethod",
                )
                # The sibling layer and the non-fused projections are untouched.
                self.assertEqual(
                    methods["model.layers.1.self_attn.qkv_proj"], "Fp8LinearMethod"
                )
                self.assertEqual(
                    methods["model.layers.0.self_attn.dense"], "Fp8LinearMethod"
                )
                self.assertEqual(methods["model.layers.0.mlp.fc1"], "Fp8LinearMethod")

    def test_fp8_ignoring_one_qkv_shard_raises(self):
        # Counter-example: a partial shard list is rejected, not silently
        # applied, in both mapping variants.
        for label, mapping in PACKED_MAPPING_VARIANTS.items():
            with self.subTest(packed_mapping=label):
                with self.assertRaises(ValueError) as ctx:
                    self.methods(
                        Fp8Config(
                            ignored_layers=[linear_prefix(0, "self_attn.q_proj")],
                            packed_modules_mapping=mapping,
                        )
                    )
                self.assertIn("model.layers.0.self_attn.qkv_proj", str(ctx.exception))

    def test_ignored_layers_env_var_is_scoped(self):
        target = linear_prefix(0, "mlp.fc1")
        with envs.SGLANG_FP8_IGNORED_LAYERS.override(target):
            self.assertEqual(
                self.methods(Fp8Config())[target], "UnquantizedLinearMethod"
            )
        # The override does not leak into configurations built afterwards.
        self.assertEqual(Fp8Config().ignored_layers, [])

    def test_modelopt_fp8_excludes_lm_head(self):
        # ModelOpt FP8 treats ParallelLMHead as quantizable and matches
        # exclude_modules against the prefix, so this is where the lm_head
        # prefix has a real resolver.
        def modelopt(exclude_modules):
            return ModelOptFp8Config(
                exclude_modules=exclude_modules,
                packed_modules_mapping=PhiForCausalLM.packed_modules_mapping,
            )

        cases = (
            (None, "ModelOptFp8LinearMethod", "ModelOptFp8LinearMethod"),
            (["lm_head"], "UnquantizedLinearMethod", "ModelOptFp8LinearMethod"),
        )
        for exclude_modules, lm_head_method, linear_method in cases:
            with self.subTest(exclude_modules=exclude_modules):
                methods = self.methods(modelopt(exclude_modules))
                self.assertEqual(methods["lm_head"], lm_head_method)
                # lm_head exclusion must not leak onto the body linears, and
                # vice versa.
                self.assert_layer_methods(methods, 0, linear_method)


if __name__ == "__main__":
    unittest.main()
