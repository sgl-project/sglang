"""Regression tests for ModelOpt quantization in the Qwen3 vision tower."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4EmbeddingMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models import qwen3_vl
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _NamedQuantConfig:
    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name


class TestQwen3VisionQuantConfig(CustomTestCase):
    @staticmethod
    def _construct_outer_model(quant_config):
        config = SimpleNamespace(
            language_model_only=False,
            encoder_only=True,
            language_only=False,
            vision_config=SimpleNamespace(deepstack_visual_indexes=[]),
            text_config=SimpleNamespace(rope_scaling={}),
        )
        parallel = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True, world_size=1)
        )

        with (
            patch.object(qwen3_vl, "get_parallel", return_value=parallel),
            patch.object(
                qwen3_vl,
                "get_mm",
                return_value=SimpleNamespace(mm_enable_dp_encoder=False),
            ),
            patch.object(
                qwen3_vl,
                "Qwen3VLMoeVisionModel",
                return_value=nn.Module(),
            ) as vision_cls,
            patch.object(qwen3_vl, "LogitsProcessor", return_value=nn.Module()),
            patch.object(qwen3_vl, "Pooler", return_value=nn.Module()),
        ):
            Qwen3VLForConditionalGeneration(
                config,
                quant_config=quant_config,
                language_model_cls=object,
            )
        return vision_cls

    def test_only_modelopt_configs_are_forwarded_to_vision(self):
        for name in ("modelopt_fp4", "modelopt_mixed"):
            with self.subTest(name=name):
                quant_config = _NamedQuantConfig(name)
                vision_cls = self._construct_outer_model(quant_config)

                self.assertIs(vision_cls.call_args.kwargs["quant_config"], quant_config)

        for name in ("fp8", "gptq", "bitsandbytes"):
            with self.subTest(name=name):
                vision_cls = self._construct_outer_model(_NamedQuantConfig(name))
                self.assertIsNone(vision_cls.call_args.kwargs["quant_config"])

        vision_cls = self._construct_outer_model(None)
        self.assertIsNone(vision_cls.call_args.kwargs["quant_config"])


class TestQwen3VisionModelOptPrefixes(CustomTestCase):
    @staticmethod
    def _mixed_config(*, exclude_modules=None):
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "exclude_modules": list(exclude_modules or []),
                "quantized_layers": {
                    "model.visual.blocks.0.attn.q_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "model.visual.blocks.0.attn.k_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "model.visual.blocks.0.attn.v_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "model.visual.blocks.0.mlp.linear_fc1": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                    "model.visual.pos_embed": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                },
                "packed_modules_mapping": {
                    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                },
            }
        )
        quant_config.apply_weight_name_mapper(
            Qwen3VLForConditionalGeneration.hf_to_sglang_mapper
        )
        return quant_config

    def test_mapper_handles_visual_root_glob_and_layer_names(self):
        mapper = Qwen3VLForConditionalGeneration.hf_to_sglang_mapper
        cases = {
            "model.visual": "visual",
            "model.visual*": "visual*",
            "model.visual.*": "visual.*",
            "model.visual.blocks.0.attn.q_proj": ("visual.blocks.0.attn.q_proj"),
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(mapper._map_name(source), expected)

    def test_uniform_visual_exclusions_survive_mapping(self):
        cases = {
            "model.visual": "visual",
            "model.visual*": "visual*",
            "model.visual.*": "visual.*",
            "model.visual.blocks.0.mlp.linear_fc1": ("visual.blocks.0.mlp.linear_fc1"),
        }

        for source, expected in cases.items():
            with self.subTest(source=source):
                quant_config = ModelOptFp4Config(
                    is_checkpoint_nvfp4_serialized=True,
                    group_size=16,
                    exclude_modules=[source],
                )
                quant_config.apply_weight_name_mapper(
                    Qwen3VLForConditionalGeneration.hf_to_sglang_mapper
                )

                self.assertEqual(quant_config.exclude_modules, [expected])
                self.assertTrue(
                    quant_config.is_layer_excluded(
                        "model.visual.blocks.0.mlp.linear_fc1"
                    )
                )

    def test_uniform_fused_visual_qkv_honors_split_exclusions(self):
        split_layers = [
            f"model.visual.blocks.0.attn.{projection}"
            for projection in ("q_proj", "k_proj", "v_proj")
        ]
        linear = ReplicatedLinear.__new__(ReplicatedLinear)

        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=split_layers,
            packed_modules_mapping={
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            },
        )
        quant_config.apply_weight_name_mapper(
            Qwen3VLForConditionalGeneration.hf_to_sglang_mapper
        )
        method = quant_config.get_quant_method(
            linear, "model.visual.blocks.0.attn.qkv_proj"
        )
        self.assertIsInstance(method, UnquantizedLinearMethod)

        partial = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=split_layers[:1],
            packed_modules_mapping={
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            },
        )
        partial.apply_weight_name_mapper(
            Qwen3VLForConditionalGeneration.hf_to_sglang_mapper
        )
        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            partial.get_quant_method(linear, "model.visual.blocks.0.attn.qkv_proj")

    def test_mixed_visual_metadata_dispatches_quantized_methods(self):
        quant_config = self._mixed_config()
        linear = ReplicatedLinear.__new__(ReplicatedLinear)

        self.assertEqual(
            quant_config.resolve_quant_algo("model.visual.blocks.0.attn.qkv_proj"),
            "NVFP4",
        )
        self.assertIsInstance(
            quant_config.get_quant_method(
                linear, "model.visual.blocks.0.attn.qkv_proj"
            ),
            ModelOptFp4LinearMethod,
        )
        self.assertEqual(
            quant_config.resolve_quant_algo("model.visual.blocks.0.mlp.linear_fc1"),
            "NVFP4",
        )
        self.assertIsInstance(
            quant_config.get_quant_method(
                linear, "model.visual.blocks.0.mlp.linear_fc1"
            ),
            ModelOptFp4LinearMethod,
        )

    def test_mixed_visual_position_embedding_dispatches_and_excludes(self):
        embedding = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        quant_config = self._mixed_config()

        self.assertEqual(
            quant_config.resolve_quant_algo("model.visual.pos_embed"), "NVFP4"
        )
        self.assertIsInstance(
            quant_config.get_quant_method(embedding, "model.visual.pos_embed"),
            ModelOptNvFp4EmbeddingMethod,
        )

        excluded = self._mixed_config(exclude_modules=["model.visual.pos_embed"])
        self.assertIsNone(
            excluded.get_quant_method(embedding, "model.visual.pos_embed")
        )


if __name__ == "__main__":
    unittest.main()
