"""LoRA targets for Beam-style layers: an attention output gate named gate_proj,
and dense MLP layers ahead of MoE layers."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.configs.load_config import LoadConfig
from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_config import LoRAConfig
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.lora.utils import (
    PRESERVED_LORA_TARGET_NAMES,
    get_normalized_target_modules,
    get_target_module_name,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.beam_fixture import make_beam_fixture
from sglang.test.beam_lora_fixture import PEFT_PREFIX, make_beam_lora_adapter
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class TestBeamLoRATargets(CustomTestCase):
    def test_bare_mlp_targets_do_not_add_the_attention_gate(self):
        peft_targets = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
        self.assertEqual(
            get_normalized_target_modules(peft_targets),
            {"qkv_proj", "gate_up_proj", "down_proj"},
        )

    def test_qualified_attention_gate_survives_full_prefixes(self):
        for prefix in ("", "model.layers.0.", f"{PEFT_PREFIX}model.layers.0."):
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    get_normalized_target_modules([f"{prefix}self_attn.gate_proj"]),
                    {"self_attn.gate_proj"},
                )
                self.assertEqual(
                    get_normalized_target_modules([f"{prefix}mlp.gate_proj"]),
                    {"gate_up_proj"},
                )

    def test_preserved_targets_are_accepted_by_the_cli(self):
        """Every preserved target must be a --lora-target-modules choice."""
        parser = argparse.ArgumentParser(prog="sglang serve")
        ServerArgs.add_cli_args(parser)
        targets = sorted(PRESERVED_LORA_TARGET_NAMES)
        try:
            args = parser.parse_args(
                ["--model-path", "unused", "--lora-target-modules", *targets]
            )
        except SystemExit:
            self.fail(f"--lora-target-modules rejects {targets}")
        self.assertEqual(sorted(args.lora_target_modules), targets)

    def test_adapter_keeps_the_attention_gate_apart_from_the_mlp_gate(self):
        """An attention gate must not be stacked with a zero up_proj into its
        layer's gate_up_proj, where it collides with the MLP gate's factors."""
        with (
            tempfile.TemporaryDirectory(prefix="beam-lora-targets-") as root,
            get_context().override_server_args(),
        ):
            config, state = make_beam_fixture(root)
            adapter_dir = Path(root) / "adapter"
            generated = make_beam_lora_adapter(
                adapter_dir, config, state, ["attention_gate", "dense_mlp"], rank=4
            ).weights
            lora_config = LoRAConfig(str(adapter_dir))
            adapter = LoRAAdapter(
                uid="beam",
                config=lora_config,
                base_hf_config=config,
                load_config=LoadConfig(),
                lora_backend=None,
            )
            adapter.initialize_weights()
        weights = adapter.layers[0].weights
        self.assertFalse(any("self_attn.gate_up_proj" in name for name in weights))
        targets = get_normalized_target_modules(lora_config.target_modules)
        attention_gate = f"{PEFT_PREFIX}model.layers.0.self_attn.gate_proj"
        mlp = f"{PEFT_PREFIX}model.layers.0.mlp"
        for side in ("A", "B"):
            with self.subTest(side=side):
                gate_name = f"{attention_gate}.lora_{side}.weight"
                self.assertTrue(torch.equal(weights[gate_name], generated[gate_name]))
                self.assertEqual(
                    get_target_module_name(gate_name, targets), "self_attn.gate_proj"
                )
                mlp_name = f"{mlp}.gate_up_proj.lora_{side}.weight"
                stacked = torch.cat(
                    [
                        generated[f"{mlp}.gate_proj.lora_{side}.weight"],
                        generated[f"{mlp}.up_proj.lora_{side}.weight"],
                    ]
                )
                self.assertTrue(torch.equal(weights[mlp_name], stacked))
                self.assertEqual(
                    get_target_module_name(mlp_name, targets), "gate_up_proj"
                )

    def test_loading_normalizes_targets_once_and_preserves_embedding_filter(self):
        # Fully qualified expert targets make rescanning the entire target list
        # per tensor quadratic in the size of an adapter.
        layer_modules = [
            f"model.layers.{layer}.mlp.experts.{expert}.down_proj"
            for layer in range(2)
            for expert in range(3)
        ]
        layer_weights = {
            f"{PEFT_PREFIX}{module}.lora_{side}.weight": (
                torch.arange(8, dtype=torch.float32).reshape(shape) + index
            )
            for index, module in enumerate(layer_modules)
            for side, shape in (("A", (2, 4)), ("B", (4, 2)))
        }
        embedding_weights = {
            f"{PEFT_PREFIX}{module}.lora_{side}.weight": torch.full(shape, value)
            for module, value in (("model.embed_tokens", 11.0), ("lm_head", 13.0))
            for side, shape in (("A", (2, 4)), ("B", (4, 2)))
        }
        tensors = layer_weights | embedding_weights
        for entrypoint in ("disk", "tensors"):
            for embedding_target in ("embed_tokens", "lm_head", "all-linear"):
                with self.subTest(
                    entrypoint=entrypoint, embedding_target=embedding_target
                ):
                    targets = (
                        "all-linear"
                        if embedding_target == "all-linear"
                        else layer_modules + [embedding_target]
                    )
                    adapter = LoRAAdapter(
                        uid="normalization-once",
                        config=LoRAConfig.from_dict(
                            {
                                "peft_type": "LORA",
                                "target_modules": targets,
                                "r": 2,
                                "lora_alpha": 4,
                            }
                        ),
                        base_hf_config=PretrainedConfig(num_hidden_layers=2),
                        load_config=LoadConfig(),
                        lora_backend=None,
                    )
                    with patch(
                        "sglang.srt.lora.utils.get_normalized_target_modules",
                        wraps=get_normalized_target_modules,
                    ) as normalize:
                        if entrypoint == "disk":
                            # Only replace checkpoint I/O; loading and weight
                            # normalization run through the real adapter.
                            with patch.object(
                                DefaultModelLoader,
                                "_get_weights_iterator",
                                return_value=iter(tensors.items()),
                            ):
                                adapter.initialize_weights()
                        else:
                            adapter.initialize_weights_from_tensors(tensors)

                    for layer_id, layer in enumerate(adapter.layers):
                        expected = {
                            name: weight
                            for name, weight in layer_weights.items()
                            if f".layers.{layer_id}." in name
                        }
                        self.assertEqual(set(layer.weights), set(expected))
                        for name, weight in expected.items():
                            torch.testing.assert_close(
                                layer.weights[name], weight, rtol=0, atol=0
                            )
                    expected_embeddings = {
                        name: weight
                        for name, weight in embedding_weights.items()
                        if embedding_target == "all-linear"
                        or f".{embedding_target}." in name
                    }
                    self.assertEqual(
                        set(adapter.embedding_layers), set(expected_embeddings)
                    )
                    for name, weight in expected_embeddings.items():
                        torch.testing.assert_close(
                            adapter.embedding_layers[name], weight, rtol=0, atol=0
                        )
                    self.assertEqual(normalize.call_count, 1)

    def test_dense_mlp_buffers_follow_decoder_layer_linears(self):
        def model(path, leaf, as_linear=True):
            root = parent = nn.Module()
            for part in path.split("."):
                parent.add_module(part, nn.Module())
                parent = getattr(parent, part)
            if as_linear:
                linear = MergedColumnParallelLinear(
                    8, [8, 8], bias=False, tp_rank=0, tp_size=1
                )
                parent.add_module(leaf, linear)
            else:
                parent.register_parameter(leaf, nn.Parameter(torch.zeros(2, 16, 8)))
            return root

        with get_context().override_server_args():
            cases = {
                # A leading dense layer in a MoE model owns a dense MLP.
                "dense_layer": (model("model.layers.0.mlp", "gate_up_proj"), True),
                # Packed routed experts hold gate_up_proj as a parameter.
                "packed_experts": (
                    model("model.layers.0.mlp.experts", "gate_up_proj", False),
                    False,
                ),
                # A vision tower's MLP is outside the decoder layers LoRA wraps.
                "vision_tower": (model("visual.blocks.0.mlp", "gate_up_proj"), False),
            }
        for case, (module_tree, expected) in cases.items():
            with self.subTest(case=case):
                self.assertEqual(
                    LoRAMemoryPool._has_dense_module(module_tree, "gate_up_proj"),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
