"""CPU accuracy test for AutoRound INT4 checkpoints in both packing formats
(auto_round:auto_gptq / auto_round:auto_awq)."""

import os
import unittest
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.parameter import PackedvLLMParameter
from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM
from sglang.srt.utils import cpu_has_amx_support, kill_process_tree
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cpu_ci(est_time=183, suite="stage-a-tp-test-cpu-intel")


class TestAutoRoundCPUConfig(CustomTestCase):
    def test_merged_gptq_zero_points_preserve_packed_shards(self):
        """Merged loading must preserve packed zero points instead of zeroing K/V."""
        checkpoint_sizes = [128, 128, 256, 256]
        checkpoint = torch.arange(192, dtype=torch.int32).reshape(2, 96)
        checkpoint_shards = checkpoint.split([16, 16, 32, 32], dim=1)
        for tp_size in (1, 3):
            output_sizes = [
                size if tp_size == 1 else size * 3 // 2 for size in checkpoint_sizes
            ]
            for tp_rank in range(tp_size):
                with self.subTest(tp_size=tp_size, tp_rank=tp_rank):
                    merged = MergedColumnParallelLinear.__new__(
                        MergedColumnParallelLinear
                    )
                    torch.nn.Module.__init__(merged)
                    merged.output_sizes = output_sizes
                    merged.tp_group = None
                    merged.use_presharded_weights = False
                    parameter = PackedvLLMParameter(
                        data=torch.full(
                            (2, sum(output_sizes) // tp_size // 8),
                            -1,
                            dtype=torch.int32,
                        ),
                        input_dim=0,
                        output_dim=1,
                        packed_dim=1,
                        packed_factor=Fraction(8, 1),
                        weight_loader=merged.weight_loader_v2,
                    )
                    with (
                        patch("sglang.srt.layers.linear._is_cpu", True),
                        patch(
                            "sglang.srt.layers.linear.get_group_rank_size",
                            return_value=(tp_rank, tp_size),
                        ),
                    ):
                        merged.weight_loader_v2(
                            parameter, checkpoint[:, :64], (0, 1, 2)
                        )
                        merged.weight_loader_v2(parameter, checkpoint_shards[3], 3)
                    expected_shards = []
                    for shard, output_size in zip(checkpoint_shards, output_sizes):
                        padded = torch.nn.functional.pad(
                            shard, (0, output_size // 8 - shard.shape[1])
                        )
                        expected_shards.append(padded.chunk(tp_size, dim=1)[tp_rank])
                    torch.testing.assert_close(
                        parameter.data,
                        torch.cat(expected_shards, dim=1),
                        rtol=0,
                        atol=0,
                    )

    def test_qwen35_fused_shared_expert_uses_source_bit_overrides(self):
        quant_config = AutoRoundConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "block_name_to_quantize": ["model.language_model.layers"],
                "extra_config": {
                    "model.language_model.layers.0.mlp.shared_expert.gate_proj": {
                        "bits": 16
                    },
                    "model.language_model.layers.0.mlp.shared_expert.up_proj": {
                        "bits": 16
                    },
                    "model.language_model.layers.0.mlp.shared_expert.down_proj": {
                        "bits": 16
                    },
                },
            }
        )
        layer = torch.nn.Linear(8, 8)
        parallel = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True, world_size=1)
        )
        config = SimpleNamespace(
            hidden_size=8,
            num_hidden_layers=1,
            layers_block_type=["attention"],
            rms_norm_eps=1e-6,
        )
        with (
            patch(
                "sglang.srt.models.qwen3_5.get_parallel",
                return_value=parallel,
            ),
            patch.object(
                Qwen3_5ForCausalLM,
                "_build_embed_tokens",
                return_value=torch.nn.Identity(),
            ),
            patch("sglang.srt.models.qwen3_5.make_pp_layers", return_value=([], 0, 1)),
            patch(
                "sglang.srt.models.qwen3_5._use_mnnvl_cutedsl_fusion",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.qwen3_5.GemmaRMSNorm",
                return_value=torch.nn.Identity(),
            ),
        ):
            Qwen3_5ForCausalLM(config=config, quant_config=quant_config)

        shared_gate_up = quant_config.get_layer_config(
            layer, "model.language_model.layers.0.mlp.shared_expert.gate_up_proj"
        )
        routed_gate_up = quant_config.get_layer_config(
            layer, "model.language_model.layers.0.mlp.experts.gate_up_proj_moe"
        )

        self.assertEqual(shared_gate_up[0], 16)
        self.assertEqual(routed_gate_up[0], 4)

    def test_gptq_defaults_are_explicit(self):
        quant_config = AutoRoundConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
            }
        )

        gptq_kwargs = quant_config.get_gptq_config_kwargs(4, 128)
        self.assertFalse(gptq_kwargs["desc_act"])
        self.assertFalse(gptq_kwargs["lm_head_quantized"])
        self.assertEqual(gptq_kwargs["dynamic"], {})

    def test_gptq_desc_act_is_rejected(self):
        quant_config = AutoRoundConfig.from_config(
            {
                "bits": 4,
                "group_size": 128,
                "sym": True,
                "packing_format": "auto_round:auto_gptq",
                "desc_act": True,
            }
        )

        with self.assertRaisesRegex(ValueError, "desc_act=False only"):
            quant_config.get_gptq_config_kwargs(4, 128)


@unittest.skipUnless(
    cpu_has_amx_support(),
    "AutoRound INT4 CPU inference requires the Intel AMX CPU backend.",
)
class TestAutoRoundCPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_mmlu(self):
        device = "cpu"
        for model in DEFAULT_AUTOROUND_MODEL_NAME_FOR_TEST:
            with self.subTest(model=model):
                print(f"\n[INFO] Launching server for model: {model}")
                process = popen_launch_server(
                    model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=["--trust-remote-code", "--quantization", "auto-round"],
                    device=device,
                )

                try:
                    args = SimpleNamespace(
                        base_url=self.base_url,
                        model=model,
                        eval_name="mmlu",
                        num_examples=32,
                        num_threads=32,
                        device=device,
                    )
                    metrics = run_sgl_eval(args)
                    self.assertGreaterEqual(metrics["score"], 0.25)
                finally:
                    kill_process_tree(process.pid)
                    print(f"[INFO] Server for {model} stopped.")


if __name__ == "__main__":
    os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")
    unittest.main()
