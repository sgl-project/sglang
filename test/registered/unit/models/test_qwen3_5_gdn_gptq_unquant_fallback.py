import json
import tempfile
import unittest
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import PretrainedConfig

from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.qwen3_next import Qwen3NextConfig
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.gptq.gptq import (
    CPUGPTQConfig,
    GPTQAscendConfig,
    GPTQConfig,
    GPTQLinearMethod,
    GPTQMarlinConfig,
    GPTQMarlinLinearMethod,
)
from sglang.srt.layers.quantization.gptq.schemes import (
    GPTQAscendLinearScheme,
    GPTQIntelAMXLinearScheme,
)
from sglang.srt.layers.quantization.marlin_utils import (
    check_marlin_supports_layer,
    verify_marlin_supports_layer,
)
from sglang.srt.layers.quantization.moe_wna16 import MoeWNA16Config, MoeWNA16Method
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import (
    _DYNAMIC_FUSED_SHARDS,
    _FALLBACK_FUSED_SHARDS,
    get_dynamic_override,
    is_layer_skipped,
    override_config,
    validate_quant_param_compatibility,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.weight_utils import (
    get_quant_config,
    inspect_safetensors_tensor_names,
)
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_next import Qwen3GatedDeltaNet, Qwen3NextForCausalLM
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_gptq_marlin_config(dynamic=None, packed_modules_mapping=None):
    return GPTQMarlinConfig(
        weight_bits=4,
        group_size=128,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={} if dynamic is None else dynamic,
        full_config={},
        packed_modules_mapping=packed_modules_mapping,
    )


def _make_dummy_loader_module(
    proj_module, prefix="model.layers.0.linear_attn.in_proj_ba"
):
    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            parts = prefix.split(".")
            cur = self
            for part in parts[:-1]:
                child = getattr(cur, part, None)
                if child is None:
                    child = torch.nn.Module()
                    setattr(cur, part, child)
                cur = child
            setattr(cur, parts[-1], proj_module)

            self.config = SimpleNamespace(
                num_experts=0,
                tie_word_embeddings=False,
                encoder_only=False,
            )
            self.pp_group = SimpleNamespace(is_last_rank=False)
            self.enable_shared_expert_fusion = False
            self.num_fused_shared_experts = 0
            self.pooler = None

    return _DummyModel()


class TestQwen35GdnGptqUnquantFallback(CustomTestCase):
    def setUp(self):
        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu"),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)
        self.addCleanup(reset_context)
        self.quant_config = _make_gptq_marlin_config()

    def test_dynamic_fused_shards_and_is_layer_skipped_non_regression(self):
        """Verify _DYNAMIC_FUSED_SHARDS expands GDN shards without breaking is_layer_skipped (#23973)."""
        self.assertIn("in_proj_ba", _DYNAMIC_FUSED_SHARDS)
        self.assertEqual(
            _DYNAMIC_FUSED_SHARDS["in_proj_ba"], ["in_proj_b", "in_proj_a"]
        )
        self.assertIn("in_proj_qkvz", _DYNAMIC_FUSED_SHARDS)
        self.assertEqual(
            _DYNAMIC_FUSED_SHARDS["in_proj_qkvz"], ["in_proj_qkv", "in_proj_z"]
        )

        # _FALLBACK_FUSED_SHARDS must NOT include GDN fused layers so FP8
        # checkpoints listing unified `in_proj_ba`/`in_proj_qkvz` are skipped.
        self.assertNotIn("in_proj_ba", _FALLBACK_FUSED_SHARDS)
        self.assertNotIn("in_proj_qkvz", _FALLBACK_FUSED_SHARDS)
        self.assertTrue(
            is_layer_skipped(
                "model.layers.0.linear_attn.in_proj_ba",
                ["model.layers.0.linear_attn.in_proj_ba"],
                {},
            )
        )

    def test_get_dynamic_override_fused_shards_strict_validation(self):
        """Verify fused negative exclusions require all shards (or the fused name) and reject mixed precision."""
        # Excluded via all anchored shards (cannot accidentally match the fused name)
        config_all_shards = _make_gptq_marlin_config(
            dynamic={
                "-:.*in_proj_b$": {},
                "-:.*in_proj_a$": {},
                "-:.*in_proj_qkv$": {},
                "-:.*in_proj_z$": {},
            }
        )
        self.assertFalse(
            get_dynamic_override(
                config_all_shards, "model.layers.0.linear_attn.in_proj_ba"
            )
        )
        self.assertFalse(
            get_dynamic_override(
                config_all_shards, "model.layers.0.linear_attn.in_proj_qkvz"
            )
        )

        # Excluded via fused name directly
        config_fused = _make_gptq_marlin_config(dynamic={"-:.*in_proj_ba.*": {}})
        self.assertFalse(
            get_dynamic_override(config_fused, "model.layers.0.linear_attn.in_proj_ba")
        )

        # Partial exclusion: only in_proj_z excluded -> must NOT mark in_proj_qkv unquantized
        mixed_qkvz = _make_gptq_marlin_config(dynamic={"-:.*in_proj_z$": {}})
        with self.assertRaisesRegex(ValueError, "All shards of fused layers"):
            get_dynamic_override(mixed_qkvz, "model.layers.0.linear_attn.in_proj_qkvz")

        # Partial exclusion: only in_proj_b excluded -> must reject mixed precision
        mixed_ba = _make_gptq_marlin_config(dynamic={"-:.*in_proj_b$": {}})
        with self.assertRaisesRegex(ValueError, "All shards of fused layers"):
            get_dynamic_override(mixed_ba, "model.layers.0.linear_attn.in_proj_ba")

        # Mixed positive overrides across shards -> must reject
        mixed_pos = _make_gptq_marlin_config(
            dynamic={"+:.*in_proj_qkv$": {"group_size": 32}}
        )
        with self.assertRaisesRegex(ValueError, "mixed dynamic quantization overrides"):
            get_dynamic_override(mixed_pos, "model.layers.0.linear_attn.in_proj_qkvz")

        # Matching positive overrides across all shards -> succeeds
        consistent_pos = _make_gptq_marlin_config(
            dynamic={
                "+:.*in_proj_qkv$": {"group_size": 32},
                "+:.*in_proj_z$": {"group_size": 32},
            }
        )
        self.assertEqual(
            get_dynamic_override(
                consistent_pos,
                "model.layers.0.linear_attn.in_proj_qkvz",
                "group_size",
                128,
            ),
            32,
        )

    def test_gptq_marlin_is_linear_unquantized(self):
        """Verify is_linear_unquantized uses dynamic rules or checkpoint manifest rather than LinearBase(1, 1)."""
        self.assertFalse(
            self.quant_config.is_linear_unquantized("model.layers.0.self_attn.q_proj")
        )
        config = _make_gptq_marlin_config(
            dynamic={"-:.*in_proj_b$": {}, "-:.*in_proj_a$": {}}
        )
        self.assertTrue(
            config.is_linear_unquantized("model.layers.0.linear_attn.in_proj_ba")
        )
        self.assertFalse(
            config.is_linear_unquantized("model.layers.0.self_attn.q_proj")
        )

        # Partial exclusion must still fail loudly here too.
        partial_config = _make_gptq_marlin_config(dynamic={"-:.*in_proj_b$": {}})
        with self.assertRaises(ValueError):
            partial_config.is_linear_unquantized(
                "model.layers.0.linear_attn.in_proj_ba"
            )

    def test_check_marlin_supports_layer_tile_indivisible(self):
        """Verify check_marlin_supports_layer checks partition shape against Marlin tiles."""
        layer_96 = MergedColumnParallelLinear(
            input_size=5120,
            output_sizes=[48, 48],
            bias=False,
            quant_config=None,
            parallel_group="tp",
        )
        self.assertFalse(check_marlin_supports_layer(layer_96, group_size=128))

        layer_128 = ColumnParallelLinear(
            input_size=5120,
            output_size=128,
            bias=False,
            quant_config=None,
            parallel_group="tp",
        )
        self.assertTrue(check_marlin_supports_layer(layer_128, group_size=128))

    def test_check_marlin_supports_layer_resolves_parallel_group_pre_init(self):
        """Verify check_marlin_supports_layer resolves per-layer tp_group before partition attrs exist."""
        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)

        # Simulate pre-create_weights state during LinearBase.__init__.
        col_attn_tp = ColumnParallelLinear(
            input_size=5120,
            output_size=128,
            bias=False,
            quant_config=None,
            parallel_group="attn_tp",
        )
        del col_attn_tp.output_size_per_partition
        self.assertTrue(check_marlin_supports_layer(col_attn_tp, group_size=128))

        col_tp = ColumnParallelLinear(
            input_size=5120,
            output_size=128,
            bias=False,
            quant_config=None,
            parallel_group="tp",
        )
        del col_tp.output_size_per_partition
        self.assertFalse(check_marlin_supports_layer(col_tp, group_size=128))

        col_replicated = ColumnParallelLinear(
            input_size=5120,
            output_size=128,
            bias=False,
            quant_config=None,
            parallel_group="replicated",
        )
        del col_replicated.output_size_per_partition
        self.assertTrue(check_marlin_supports_layer(col_replicated, group_size=128))

        row_attn_tp = RowParallelLinear(
            input_size=256,
            output_size=128,
            bias=False,
            reduce_results=False,
            quant_config=None,
            parallel_group="attn_tp",
        )
        del row_attn_tp.input_size_per_partition
        self.assertTrue(check_marlin_supports_layer(row_attn_tp, group_size=128))

        row_tp = RowParallelLinear(
            input_size=256,
            output_size=128,
            bias=False,
            reduce_results=False,
            quant_config=None,
            parallel_group="tp",
        )
        del row_tp.input_size_per_partition
        self.assertFalse(check_marlin_supports_layer(row_tp, group_size=128))

    def test_gptq_marlin_rejects_quantized_incompatible_shape_without_exclusions(self):
        """Quantized layers with Marlin-incompatible TP partitions must raise instead of falling back to BF16."""
        layer_96 = MergedColumnParallelLinear(
            input_size=5120,
            output_sizes=[48, 48],
            bias=False,
            quant_config=None,
            parallel_group="tp",
        )
        with self.assertRaisesRegex(ValueError, "incompatible with GPTQMarlin"):
            self.quant_config.get_quant_method(
                layer_96, prefix="model.layers.0.linear_attn.in_proj_ba"
            )

        # Quantized 64-head in_proj_ba at TP=4 (32 per rank) must fail loudly rather than
        # allocating an uninitialized BF16 parameter that skips .qweight/.scales loading.
        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)
        with self.assertRaisesRegex(ValueError, "incompatible with GPTQMarlin"):
            MergedColumnParallelLinear(
                input_size=5120,
                output_sizes=[64, 64],
                bias=False,
                quant_config=self.quant_config,
                parallel_group="tp",
                prefix="model.layers.0.linear_attn.in_proj_ba",
            )

        # Compatible quantized layer succeeds (256 / TP=4 = 64); full negative
        # dynamic rules bypass Marlin check; positive overrides apply before
        # verify_marlin_supports_layer.
        layer_256 = ColumnParallelLinear(
            input_size=5120,
            output_size=256,
            bias=False,
            quant_config=None,
            parallel_group="tp",
        )
        quant_method = self.quant_config.get_quant_method(
            layer_256, prefix="model.layers.0.self_attn.o_proj"
        )
        self.assertIsInstance(quant_method, GPTQMarlinLinearMethod)

        dyn_config = _make_gptq_marlin_config(
            dynamic={
                "-:.*in_proj_b$": {},
                "-:.*in_proj_a$": {},
                "+:.*qkv_proj.*": {"group_size": 32},
            }
        )
        with patch(
            "sglang.srt.layers.quantization.marlin_utils.verify_marlin_supports_layer",
            wraps=verify_marlin_supports_layer,
        ) as mock_verify:
            skipped = dyn_config.get_quant_method(
                layer_96, prefix="model.layers.0.linear_attn.in_proj_ba"
            )
            self.assertIsInstance(skipped, UnquantizedLinearMethod)
            mock_verify.assert_not_called()

            overridden = dyn_config.get_quant_method(
                layer_256, prefix="model.layers.0.self_attn.qkv_proj"
            )
            self.assertIsInstance(overridden, GPTQMarlinLinearMethod)
            self.assertEqual(overridden.quant_config.group_size, 32)
            mock_verify.assert_called_once_with(
                layer_256, 32, prefix="model.layers.0.self_attn.qkv_proj"
            )

    def test_checkpoint_manifest_drives_gptq_unquant_init(self):
        """End-to-end get_quant_config detects dense vs quantized shards from safetensors metadata (#38854)."""
        # Case 1: #38854-style checkpoint whose `dynamic` omits `in_proj_b`/`in_proj_a`,
        # while `model.safetensors.index.json` stores them as BF16 `.weight`.
        quant_dict = {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "dynamic": {},
        }
        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.layers.0.linear_attn.in_proj_b.weight": "shard0.safetensors",
                            "model.layers.0.linear_attn.in_proj_a.weight": "shard0.safetensors",
                            "model.layers.0.linear_attn.in_proj_qkv.qweight": "shard0.safetensors",
                            "model.layers.0.linear_attn.in_proj_z.qweight": "shard0.safetensors",
                            "model.layers.0.self_attn.q_proj.qweight": "shard0.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            model_config = SimpleNamespace(
                quantization="gptq_marlin",
                hf_config=PretrainedConfig(quantization_config=dict(quant_dict)),
                model_path=model_path,
                revision=None,
                is_draft_model=False,
                is_draft_quantization_explicit=False,
            )
            quant_config = get_quant_config(
                model_config, LoadConfig(), Qwen3_5ForCausalLM.packed_modules_mapping
            )

        self.assertIsInstance(quant_config, GPTQMarlinConfig)
        # deepcopy stays O(1) by sharing immutable manifest instance
        self.assertIs(
            deepcopy(quant_config)._checkpoint_tensor_manifest,
            quant_config._checkpoint_tensor_manifest,
        )
        # Wrapper prefix canonicalization handles text/VLM/leading-dot prefixes without hitting MTP
        for runtime_prefix in (
            "model.layers.0.linear_attn.in_proj_ba",
            ".layers.0.linear_attn.in_proj_ba",
            "model.language_model.layers.0.linear_attn.in_proj_ba",
        ):
            self.assertTrue(quant_config.is_linear_unquantized(runtime_prefix))
        self.assertFalse(
            quant_config.is_linear_unquantized(
                "model.layers.0.linear_attn.in_proj_qkvz"
            )
        )
        self.assertFalse(
            quant_config.is_linear_unquantized("mtp.layers.0.linear_attn.in_proj_ba")
        )

        # Mapper propagation preserves match when model uses HfToSglang mapper
        quant_config.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_prefix={"model.": "language_model.model."})
        )
        self.assertTrue(
            quant_config.is_linear_unquantized(
                "language_model.model.layers.0.linear_attn.in_proj_ba"
            )
        )

        # Even at TP=4 with 64 heads (partition=32) or TP=1 with 48 heads (partition=96),
        # checkpoint-dense in_proj_ba is instantiated as UnquantizedLinearMethod.
        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)
        ba_tp4 = Qwen3_5GatedDeltaNet.create_ba_proj(
            None,
            hidden_size=5120,
            num_v_heads=64,
            quant_config=quant_config,
            prefix="model.layers.0.linear_attn.in_proj_ba",
            parallel_group="tp",
        )
        self.assertIsInstance(ba_tp4.quant_method, UnquantizedLinearMethod)
        self.assertEqual(ba_tp4.weight.shape, (32, 5120))

        # Case 2: quantize_config.json path + single model.safetensors header read,
        # plus direct GPTQConfig (non-Marlin) verification.
        with tempfile.TemporaryDirectory() as model_path:
            Path(model_path, "quantize_config.json").write_text(
                json.dumps(
                    {
                        "bits": 4,
                        "group_size": 128,
                        "desc_act": False,
                        "sym": True,
                    }
                ),
                encoding="utf-8",
            )
            save_file(
                {
                    "model.layers.0.linear_attn.in_proj_ba.weight": torch.zeros(1, 1),
                    "model.layers.0.self_attn.q_proj.qweight": torch.zeros(
                        1, 1, dtype=torch.int32
                    ),
                },
                str(Path(model_path) / "model.safetensors"),
            )
            model_config = SimpleNamespace(
                quantization="gptq_marlin",
                hf_config=PretrainedConfig(architectures=["Qwen3_5ForCausalLM"]),
                model_path=model_path,
                revision=None,
                is_draft_model=False,
                is_draft_quantization_explicit=False,
            )
            file_cfg = get_quant_config(
                model_config, LoadConfig(), Qwen3_5ForCausalLM.packed_modules_mapping
            )
            tensor_names = inspect_safetensors_tensor_names(model_path)

        self.assertIsInstance(file_cfg, GPTQMarlinConfig)
        self.assertTrue(
            file_cfg.is_linear_unquantized("model.layers.0.linear_attn.in_proj_ba")
        )
        self.assertFalse(
            file_cfg.is_linear_unquantized("model.layers.0.self_attn.q_proj")
        )

        gptq_cfg = GPTQConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "packed_modules_mapping": Qwen3_5ForCausalLM.packed_modules_mapping,
            }
        )
        gptq_cfg.set_checkpoint_tensor_names(tensor_names)
        layer_96 = MergedColumnParallelLinear(
            input_size=5120,
            output_sizes=[48, 48],
            bias=False,
            quant_config=None,
            parallel_group="tp",
        )
        self.assertIsInstance(
            gptq_cfg.get_quant_method(
                layer_96, prefix="model.layers.0.linear_attn.in_proj_ba"
            ),
            UnquantizedLinearMethod,
        )
        self.assertIsInstance(
            gptq_cfg.get_quant_method(
                layer_96, prefix="model.layers.0.self_attn.q_proj"
            ),
            GPTQLinearMethod,
        )

        # Case 3: Mixed dense/quantized fused shards in checkpoint must raise ValueError.
        mixed_manifest_cfg = _make_gptq_marlin_config(
            packed_modules_mapping=Qwen3_5ForCausalLM.packed_modules_mapping
        )
        mixed_manifest_cfg.set_checkpoint_tensor_names(
            {
                "model.layers.0.linear_attn.in_proj_b.qweight",
                "model.layers.0.linear_attn.in_proj_a.weight",
            }
        )
        with self.assertRaisesRegex(ValueError, "mixed dense and quantized shards"):
            mixed_manifest_cfg.get_quant_method(
                layer_96, prefix="model.layers.0.linear_attn.in_proj_ba"
            )

        # Case 4: Quantized shards at TP=4 must raise instead of falling back to BF16.
        quant_manifest_cfg = _make_gptq_marlin_config(
            packed_modules_mapping=Qwen3_5ForCausalLM.packed_modules_mapping
        )
        quant_manifest_cfg.set_checkpoint_tensor_names(
            {
                "model.layers.0.linear_attn.in_proj_b.qweight",
                "model.layers.0.linear_attn.in_proj_a.qweight",
            }
        )
        with self.assertRaisesRegex(ValueError, "incompatible with GPTQMarlin"):
            Qwen3_5GatedDeltaNet.create_ba_proj(
                None,
                hidden_size=5120,
                num_v_heads=64,
                quant_config=quant_manifest_cfg,
                prefix="model.layers.0.linear_attn.in_proj_ba",
                parallel_group="tp",
            )

        # Case 5: Remote path with dummy load_format must not attempt Hub network calls.
        with (
            patch(
                "sglang.srt.model_loader.weight_utils.hf_hub_download"
            ) as mock_download,
            patch("sglang.srt.model_loader.weight_utils.HfFileSystem") as mock_fs,
        ):
            self.assertIsNone(
                inspect_safetensors_tensor_names(
                    "org/nonexistent-qwen35-test",
                    local_files_only=(LoadFormat.DUMMY == LoadFormat.DUMMY),
                )
            )
            mock_download.assert_not_called()
            mock_fs.assert_not_called()

    def test_create_ba_proj_and_qwen3_next_respect_quant_config(self):
        """Verify create_ba_proj and Qwen3GatedDeltaNet honor dynamic/checkpoint rules and load cleanly."""
        dyn_config = _make_gptq_marlin_config(
            dynamic={"-:.*in_proj_b$": {}, "-:.*in_proj_a$": {}},
            packed_modules_mapping=Qwen3_5ForCausalLM.packed_modules_mapping,
        )
        ba_proj = Qwen3_5GatedDeltaNet.create_ba_proj(
            None,
            hidden_size=5120,
            num_v_heads=48,
            quant_config=dyn_config,
            prefix="model.layers.0.linear_attn.in_proj_ba",
            parallel_group="attn_tp",
        )
        self.assertIsInstance(ba_proj.quant_method, UnquantizedLinearMethod)
        self.assertTrue(hasattr(ba_proj, "weight"))
        self.assertEqual(ba_proj.weight.shape, (96, 5120))

        # Simulate loading in_proj_b (shard 0) and in_proj_a (shard 1)
        target_dtype = ba_proj.weight.dtype
        b_weight = torch.ones(48, 5120, dtype=target_dtype)
        a_weight = torch.ones(48, 5120, dtype=target_dtype) * 2.0
        ba_proj.weight.weight_loader(ba_proj.weight, b_weight, 0)
        ba_proj.weight.weight_loader(ba_proj.weight, a_weight, 1)

        self.assertTrue(torch.allclose(ba_proj.weight[:48], b_weight))
        self.assertTrue(torch.allclose(ba_proj.weight[48:], a_weight))

        x = torch.randn(2, 5120, dtype=target_dtype)
        out, bias = ba_proj(x)
        self.assertEqual(out.shape, (2, 96))
        self.assertIsNone(bias)

        # Multi-TP check (no shape fallback without ground truth)
        reset_context()
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=4, attn_dp_size=2),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)

        # Without exclusions or dense manifest, Qwen3_5GatedDeltaNet.create_ba_proj rejects 48 heads at attn_tp=2
        with self.assertRaisesRegex(ValueError, "incompatible with GPTQMarlin"):
            Qwen3_5GatedDeltaNet.create_ba_proj(
                None,
                hidden_size=5120,
                num_v_heads=48,
                quant_config=self.quant_config,
                prefix="model.layers.0.linear_attn.in_proj_ba",
                parallel_group="attn_tp",
            )

        # Qwen3GatedDeltaNet in qwen3_next.py likewise uses quant_config without shape fallback
        q3next_cfg = Qwen3NextConfig(
            hidden_size=256,
            linear_num_value_heads=48,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            hidden_act="silu",
        )
        q3next_cfg.linear_num_value_heads_cpu = 48
        q3next_cfg.linear_num_key_heads_cpu = 16
        # Patch CUDA GPU capability check on CPU hosts so in_proj_qkvz constructs while
        # verify_marlin_supports_layer still rejects incompatible quantized in_proj_ba.
        with patch(
            "sglang.srt.layers.quantization.gptq.schemes.gptq_marlin.verify_marlin_supported"
        ):
            with self.assertRaisesRegex(ValueError, "incompatible with GPTQMarlin"):
                Qwen3GatedDeltaNet(
                    config=q3next_cfg,
                    layer_id=0,
                    quant_config=self.quant_config,
                    prefix="model.layers.0.linear_attn",
                )

            q3next_gdn = Qwen3GatedDeltaNet(
                config=q3next_cfg,
                layer_id=0,
                quant_config=dyn_config,
                prefix="model.layers.0.linear_attn",
            )
        self.assertIsInstance(
            q3next_gdn.in_proj_ba.quant_method, UnquantizedLinearMethod
        )
        self.assertEqual(q3next_gdn.in_proj_ba.weight.shape, (48, 256))

    def test_load_weights_rejects_quant_precision_mismatch(self):
        """Verify load_weights fails loudly when checkpoint tensors conflict with dense/quantized params."""
        # Direct helper checks (including PP-missing layer safety)
        dense_params = {
            "model.layers.0.linear_attn.in_proj_ba.weight": torch.nn.Parameter(
                torch.empty(96, 256)
            )
        }
        quant_params = {
            "model.layers.0.linear_attn.in_proj_ba.qweight": torch.nn.Parameter(
                torch.empty(32, 128, dtype=torch.int32), requires_grad=False
            )
        }
        for suffix in ("qweight", "qzeros", "scales", "g_idx"):
            with self.assertRaisesRegex(
                ValueError, "Incompatible quantized checkpoint shard"
            ):
                validate_quant_param_compatibility(
                    f"model.layers.0.linear_attn.in_proj_ba.{suffix}", dense_params
                )
        with self.assertRaisesRegex(ValueError, "Incompatible dense checkpoint shard"):
            validate_quant_param_compatibility(
                "model.layers.0.linear_attn.in_proj_ba.weight", quant_params
            )
        # Missing layers (e.g. owned by another PP stage) must NOT raise.
        validate_quant_param_compatibility(
            "model.layers.99.linear_attn.in_proj_ba.qweight", dense_params
        )

        # Model load_weights paths (split shards + unified tensors, dense vs quantized)
        dense_ba = MergedColumnParallelLinear(
            input_size=256,
            output_sizes=[48, 48],
            bias=False,
            quant_config=None,
            parallel_group="tp",
            prefix="model.layers.0.linear_attn.in_proj_ba",
        )
        with patch(
            "sglang.srt.layers.quantization.gptq.schemes.gptq_marlin.verify_marlin_supported"
        ):
            quant_ba = MergedColumnParallelLinear(
                input_size=256,
                output_sizes=[64, 64],
                bias=False,
                quant_config=self.quant_config,
                parallel_group="tp",
                prefix="model.layers.0.linear_attn.in_proj_ba",
            )

        dense_loaders = (
            Qwen3_5ForCausalLM.load_weights,
            Qwen3_5MoeForCausalLM.load_weights,
            Qwen3_5ForConditionalGeneration.load_weights,
            Qwen3_5MoeForConditionalGeneration.load_weights,
            Qwen3NextForCausalLM.load_weights,
        )
        for loader_fn in dense_loaders:
            with self.subTest(
                loader=loader_fn.__qualname__, case="split_qweight_to_dense"
            ):
                mod = _make_dummy_loader_module(dense_ba)
                with self.assertRaisesRegex(ValueError, "Incompatible quantized"):
                    loader_fn(
                        mod,
                        [
                            (
                                "model.layers.0.linear_attn.in_proj_b.qweight",
                                torch.zeros(32, 48, dtype=torch.int32),
                            )
                        ],
                    )
            with self.subTest(
                loader=loader_fn.__qualname__, case="unified_qweight_to_dense"
            ):
                mod = _make_dummy_loader_module(dense_ba)
                with self.assertRaisesRegex(ValueError, "Incompatible quantized"):
                    loader_fn(
                        mod,
                        [
                            (
                                "model.layers.0.linear_attn.in_proj_ba.qweight",
                                torch.zeros(32, 96, dtype=torch.int32),
                            )
                        ],
                    )
            with self.subTest(
                loader=loader_fn.__qualname__, case="split_weight_to_quant"
            ):
                mod = _make_dummy_loader_module(quant_ba)
                with self.assertRaisesRegex(ValueError, "Incompatible dense"):
                    loader_fn(
                        mod,
                        [
                            (
                                "model.layers.0.linear_attn.in_proj_b.weight",
                                torch.zeros(64, 256),
                            )
                        ],
                    )
            with self.subTest(
                loader=loader_fn.__qualname__, case="unified_weight_to_quant"
            ):
                mod = _make_dummy_loader_module(quant_ba)
                with self.assertRaisesRegex(ValueError, "Incompatible dense"):
                    loader_fn(
                        mod,
                        [
                            (
                                "model.layers.0.linear_attn.in_proj_ba.weight",
                                torch.zeros(128, 256),
                            )
                        ],
                    )

    def test_cpu_and_ascend_gptq_configs_resolve_dense_and_dynamic_overrides(self):
        """CPUGPTQConfig and GPTQAscendConfig resolve dense layers, positive overrides, and ParallelLMHead safely."""
        for cfg_cls, expected_scheme_cls in (
            (CPUGPTQConfig, GPTQIntelAMXLinearScheme),
            (GPTQAscendConfig, GPTQAscendLinearScheme),
        ):
            with self.subTest(cfg_cls=cfg_cls.__name__):
                dyn_cfg = cfg_cls.from_config(
                    {
                        "bits": 4,
                        "group_size": 128,
                        "desc_act": False,
                        "lm_head": True,
                        "dynamic": {
                            "-:.*in_proj_b$": {},
                            "-:.*in_proj_a$": {},
                            "+:.*qkv_proj$": {"group_size": 32},
                        },
                        "packed_modules_mapping": Qwen3_5ForCausalLM.packed_modules_mapping,
                    }
                )
                ba_layer = MergedColumnParallelLinear(
                    input_size=256,
                    output_sizes=[48, 48],
                    bias=False,
                    quant_config=None,
                    parallel_group="tp",
                )
                ba_method = dyn_cfg.get_quant_method(
                    ba_layer, prefix="model.layers.0.linear_attn.in_proj_ba"
                )
                self.assertIsInstance(ba_method, UnquantizedLinearMethod)
                self.assertIsNone(ba_layer.scheme)

                qkv_layer = MergedColumnParallelLinear(
                    input_size=256,
                    output_sizes=[128, 128, 128],
                    bias=False,
                    quant_config=None,
                    parallel_group="tp",
                )
                qkv_method = dyn_cfg.get_quant_method(
                    qkv_layer, prefix="model.layers.0.self_attn.qkv_proj"
                )
                self.assertIsInstance(qkv_method, GPTQLinearMethod)
                self.assertIsInstance(qkv_layer.scheme, expected_scheme_cls)
                self.assertEqual(qkv_layer.scheme.quant_config.group_size, 32)
                self.assertEqual(dyn_cfg.group_size, 128)

                # Manifest-driven dense resolution when dynamic is empty
                manifest_cfg = cfg_cls.from_config(
                    {
                        "bits": 4,
                        "group_size": 128,
                        "desc_act": False,
                        "lm_head": True,
                        "packed_modules_mapping": Qwen3_5ForCausalLM.packed_modules_mapping,
                    }
                )
                manifest_cfg.set_checkpoint_tensor_names(
                    {
                        "model.layers.0.linear_attn.in_proj_b.weight",
                        "model.layers.0.linear_attn.in_proj_a.weight",
                        "model.layers.0.self_attn.q_proj.qweight",
                    }
                )
                ba_layer2 = MergedColumnParallelLinear(
                    input_size=256,
                    output_sizes=[48, 48],
                    bias=False,
                    quant_config=None,
                    parallel_group="tp",
                )
                self.assertIsInstance(
                    manifest_cfg.get_quant_method(
                        ba_layer2, prefix="model.layers.0.linear_attn.in_proj_ba"
                    ),
                    UnquantizedLinearMethod,
                )
                self.assertIsNone(ba_layer2.scheme)

                # ParallelLMHead must not hit LinearBase scheme assertions
                lm_head = ParallelLMHead(
                    num_embeddings=128,
                    embedding_dim=256,
                    quant_config=None,
                    prefix="lm_head",
                )
                self.assertIsNone(manifest_cfg.get_quant_method(lm_head, "lm_head"))

    def test_get_dynamic_override_noop_normalization_packed_mapping_and_3bit(self):
        """Fused positive overrides normalize against defaults, respect packed_modules_mapping, and preserve 3-bit Fraction."""
        # Single-shard no-op override matching global bits=4 passes on qkv_proj
        noop_cfg = _make_gptq_marlin_config(dynamic={"+:.*q_proj$": {"bits": 4}})
        self.assertEqual(
            get_dynamic_override(noop_cfg, "model.layers.0.self_attn.qkv_proj"),
            {"bits": 4},
        )
        self.assertEqual(
            get_dynamic_override(
                noop_cfg, "model.layers.0.self_attn.qkv_proj", "bits", 4
            ),
            4,
        )
        # Do not synthesize config defaults when caller asks with default_value=None
        self.assertIsNone(
            get_dynamic_override(
                noop_cfg, "model.layers.0.self_attn.qkv_proj", key="desc_act"
            )
        )

        # Shards with equivalent overrides after applying config defaults pass
        consistent_partial_keys = _make_gptq_marlin_config(
            dynamic={
                "+:.*q_proj$": {"bits": 8},
                "+:.*k_proj$": {"bits": 8, "group_size": 128},
                "+:.*v_proj$": {"bits": 8, "sym": True},
            }
        )
        self.assertEqual(
            get_dynamic_override(
                consistent_partial_keys,
                "model.layers.0.self_attn.qkv_proj",
                "bits",
                4,
            ),
            8,
        )
        self.assertEqual(
            get_dynamic_override(
                consistent_partial_keys,
                "model.layers.0.self_attn.qkv_proj",
                "group_size",
                128,
            ),
            128,
        )

        # Real mixed overrides (non-default bits or sym) still fail loudly
        for bad_dyn in (
            {"+:.*q_proj$": {"bits": 8}},
            {"+:.*q_proj$": {"sym": False}},
            {"+:.*in_proj_qkv$": {"bits": 8}},
        ):
            with self.subTest(bad_dyn=bad_dyn):
                bad_cfg = _make_gptq_marlin_config(dynamic=bad_dyn)
                target = (
                    "model.layers.0.linear_attn.in_proj_qkvz"
                    if "in_proj_qkv" in next(iter(bad_dyn))
                    else "model.layers.0.self_attn.qkv_proj"
                )
                with self.assertRaisesRegex(
                    ValueError, "mixed dynamic quantization overrides"
                ):
                    get_dynamic_override(bad_cfg, target)

        # dynamic=None guard
        none_dyn_cfg = _make_gptq_marlin_config()
        none_dyn_cfg.dynamic = None
        self.assertIsNone(
            get_dynamic_override(none_dyn_cfg, "model.layers.0.self_attn.qkv_proj")
        )

        # packed_modules_mapping consistency (and ignoring NPU nested dicts)
        custom_cfg = _make_gptq_marlin_config(
            dynamic={"-:.*shard_a$": {}},
            packed_modules_mapping={
                "custom_proj": ["shard_a", "shard_b"],
                "model": {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
            },
        )
        with self.assertRaisesRegex(ValueError, "All shards of fused layers"):
            get_dynamic_override(custom_cfg, "model.layers.0.attn.custom_proj")

        # override_config preserves Fraction(32, 3) on 3-bit GPTQConfig
        cfg_3bit = GPTQConfig(
            weight_bits=3,
            group_size=128,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={"+:.*q_proj$": {"group_size": 64}},
        )
        override_config(cfg_3bit, prefix="model.layers.0.self_attn.q_proj")
        self.assertEqual(cfg_3bit.group_size, 64)
        self.assertIsInstance(cfg_3bit.pack_factor, Fraction)
        self.assertEqual(cfg_3bit.pack_factor, Fraction(32, 3))

    def test_qwen3_5_gdn_gptq_in_proj_qkv_split_loading_gpu_and_cpu(self):
        """Qwen3_5GatedDeltaNet binds qweight/qzeros/scales/g_idx and loads split in_proj_qkv safely on GPU and CPU."""
        hidden_size = 128
        output_sizes = [128, 128, 256, 256]  # total=768, qkv=512, packed(4bit)=8
        gptq_cfg = GPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={},
        )

        for cpu_mode in (False, True):
            with (
                self.subTest(cpu_mode=cpu_mode),
                patch(
                    "sglang.srt.layers.quantization.gptq.schemes.gptq_linear.GPTQLinearScheme._init_kernel",
                    return_value=SimpleNamespace(),
                ),
                patch("sglang.srt.models.qwen3_5._is_cpu", cpu_mode),
                patch("sglang.srt.layers.linear._is_cpu", cpu_mode),
                patch("sglang.srt.layers.parameter._is_cpu", cpu_mode),
            ):
                proj = MergedColumnParallelLinear(
                    input_size=hidden_size,
                    output_sizes=output_sizes,
                    bias=False,
                    quant_config=gptq_cfg,
                    parallel_group="tp",
                    prefix="model.layers.0.linear_attn.in_proj_qkvz",
                )
                Qwen3_5GatedDeltaNet._bind_packed_weight_loaders(proj)

                qkv_qweight = torch.arange(
                    (hidden_size // 8) * 512, dtype=torch.int32
                ).reshape(hidden_size // 8, 512)
                z_qweight = torch.arange(
                    (hidden_size // 8) * 256, dtype=torch.int32
                ).reshape(hidden_size // 8, 256)
                qkv_qzeros = torch.arange(64, dtype=torch.int32).reshape(1, 64) + 100
                z_qzeros = torch.arange(32, dtype=torch.int32).reshape(1, 32) + 200
                qkv_scales = torch.arange(512, dtype=torch.float16).reshape(1, 512)
                z_scales = torch.arange(256, dtype=torch.float16).reshape(1, 256) + 10
                qkv_g_idx = torch.zeros(hidden_size, dtype=torch.int32)

                proj.qweight.weight_loader(proj.qweight, qkv_qweight, (0, 1, 2))
                proj.qweight.weight_loader(proj.qweight, z_qweight, 3)
                proj.qzeros.weight_loader(proj.qzeros, qkv_qzeros, (0, 1, 2))
                proj.qzeros.weight_loader(proj.qzeros, z_qzeros, 3)
                proj.scales.weight_loader(proj.scales, qkv_scales, (0, 1, 2))
                proj.scales.weight_loader(proj.scales, z_scales, 3)
                proj.g_idx.weight_loader(proj.g_idx, qkv_g_idx, (0, 1, 2))

                self.assertTrue(torch.equal(proj.qweight[:, :512], qkv_qweight))
                self.assertTrue(torch.equal(proj.qweight[:, 512:], z_qweight))
                self.assertTrue(torch.equal(proj.qzeros[:, :64], qkv_qzeros))
                self.assertTrue(torch.equal(proj.qzeros[:, 64:], z_qzeros))
                self.assertTrue(torch.equal(proj.scales[:, :512], qkv_scales))
                self.assertTrue(torch.equal(proj.scales[:, 512:], z_scales))
                self.assertTrue(torch.equal(proj.g_idx.data, qkv_g_idx))

        # Direct _load_fused_module_from_checkpoint check on CPU with qzeros (Qwen3Next path)
        with (
            patch(
                "sglang.srt.layers.quantization.gptq.schemes.gptq_linear.GPTQLinearScheme._init_kernel",
                return_value=SimpleNamespace(),
            ),
            patch("sglang.srt.layers.linear._is_cpu", True),
            patch("sglang.srt.layers.parameter._is_cpu", True),
        ):
            proj_next = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=output_sizes,
                bias=False,
                quant_config=gptq_cfg,
                parallel_group="tp",
                prefix="model.layers.0.linear_attn.in_proj_qkvz",
            )
            qkv_qzeros = torch.arange(64, dtype=torch.int32).reshape(1, 64) + 100
            proj_next.qzeros.weight_loader(proj_next.qzeros, qkv_qzeros, (0, 1, 2))
            self.assertTrue(torch.equal(proj_next.qzeros[:, :64], qkv_qzeros))

        # 3-bit Fraction split sizes must return integer values
        mock_mod = SimpleNamespace(output_sizes=[96, 96, 192, 192])
        mock_param = SimpleNamespace(
            packed_dim=1, output_dim=1, packed_factor=Fraction(32, 3)
        )
        split_sizes = Qwen3_5GatedDeltaNet._get_split_sizes_for_param(
            mock_mod, mock_param, (0, 1, 2)
        )
        self.assertEqual(split_sizes, [9, 9, 18])
        self.assertTrue(all(type(s) is int for s in split_sizes))

    def test_moe_wna16_config_gptq_manifest_and_dynamic(self):
        """MoeWNA16Config propagates checkpoint manifest/mapper/dynamic rules for GPTQ and skips manifest for AWQ."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        cfg = MoeWNA16Config.from_config(
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "desc_act": False,
                "sym": True,
                "packed_modules_mapping": Qwen3_5ForCausalLM.packed_modules_mapping,
            }
        )
        self.assertTrue(cfg.uses_checkpoint_tensor_manifest)
        cfg.set_checkpoint_tensor_names(
            {
                "model.layers.0.linear_attn.in_proj_b.weight",
                "model.layers.0.linear_attn.in_proj_a.weight",
                "model.layers.0.self_attn.q_proj.qweight",
            }
        )
        cfg.apply_weight_name_mapper(
            WeightsMapper(orig_to_new_prefix={"model.": "language_model.model."})
        )
        self.assertTrue(
            cfg.is_linear_unquantized(
                "language_model.model.layers.0.linear_attn.in_proj_ba"
            )
        )
        self.assertFalse(
            cfg.is_linear_unquantized("language_model.model.layers.0.self_attn.q_proj")
        )

        ba_layer = MergedColumnParallelLinear(
            input_size=256,
            output_sizes=[48, 48],
            bias=False,
            quant_config=None,
            parallel_group="tp",
        )
        self.assertIsInstance(
            cfg.get_quant_method(
                ba_layer, "language_model.model.layers.0.linear_attn.in_proj_ba"
            ),
            UnquantizedLinearMethod,
        )

        # FusedMoE still resolves to MoeWNA16Method (including AutoRound dicts missing desc_act)
        autoround_cfg = MoeWNA16Config.from_config(
            {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "lm_head": False,
            }
        )
        mock_moe = SimpleNamespace(__class__=FusedMoE)
        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.FusedMoE", SimpleNamespace
        ):
            self.assertIsInstance(
                autoround_cfg.get_quant_method(mock_moe, "model.layers.0.mlp.experts"),
                MoeWNA16Method,
            )

        with patch(
            "sglang.srt.layers.quantization.moe_wna16.GPTQMarlinConfig.is_gptq_marlin_compatible",
            return_value=True,
        ):
            marlin_cfg = MoeWNA16Config.from_config(
                {
                    "quant_method": "gptq",
                    "bits": 4,
                    "group_size": 128,
                    "desc_act": False,
                    "sym": True,
                    "packed_modules_mapping": Qwen3_5ForCausalLM.packed_modules_mapping,
                }
            )
        self.assertTrue(marlin_cfg.use_marlin)
        self.assertIsInstance(marlin_cfg._get_gptq_config(), GPTQMarlinConfig)
        marlin_cfg.set_checkpoint_tensor_names(
            {
                "model.layers.0.linear_attn.in_proj_b.weight",
                "model.layers.0.linear_attn.in_proj_a.weight",
                "model.layers.0.self_attn.q_proj.qweight",
            }
        )
        self.assertTrue(
            marlin_cfg.is_linear_unquantized("model.layers.0.linear_attn.in_proj_ba")
        )
        # Incompatible quantized BA still fails loudly under MoeWNA16 + Marlin
        with self.assertRaisesRegex(ValueError, "incompatible with GPTQMarlin"):
            marlin_cfg.get_quant_method(
                ba_layer, "model.layers.0.linear_attn.in_proj_qkvz"
            )

        with patch(
            "sglang.srt.layers.quantization.moe_wna16.get_device_capability",
            return_value=(8, 0),
        ):
            awq_cfg = MoeWNA16Config.from_config(
                {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "zero_point": True,
                }
            )
        self.assertFalse(awq_cfg.uses_checkpoint_tensor_manifest)


if __name__ == "__main__":
    unittest.main()
