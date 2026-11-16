"""Configuration contracts for the opt-in RDMA MoE backend."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups import moe_hook
from sglang.srt.arg_groups.overrides import (
    _a2a_ep_size,
    _a2a_fusion_adjustments,
    resolution_result,
)
from sglang.srt.model_executor.cuda_graph_config import Backend, CudaGraphConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestFlashinferRdmaArgs(CustomTestCase):
    def config(self, **kwargs):
        defaults = dict(
            model_path="unused",
            tp_size=4,
            attn_dp_size=4,
            ep_size=4,
            cuda_graph_config=CudaGraphConfig(),
            moe_a2a_backend="flashinfer_rdma",
            speculative_algorithm="NEXTN",
            speculative_num_draft_tokens=4,
            flashinfer_rdma_max_tokens_per_rank=64,
        )
        defaults.update(kwargs)
        cfg = ServerArgs(**defaults)
        if cfg.model_path == "unused":
            cfg._model_config = SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Glm5NextForConditionalGeneration"]
                )
            )
        return cfg

    def test_cli_selects_backend_and_physical_capacity(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(
            [
                "--model-path",
                "unused",
                "--moe-a2a-backend",
                "flashinfer_rdma",
                "--flashinfer-rdma-max-tokens-per-rank",
                "68",
            ]
        )
        cfg = ServerArgs.from_cli_args(args)
        self.assertEqual(cfg.moe_a2a_backend, "flashinfer_rdma")
        self.assertEqual(cfg.flashinfer_rdma_max_tokens_per_rank, 68)

    def test_resolution_keeps_draft_separate_and_prefill_eager(self):
        cfg = self.config()
        with (
            patch.object(moe_hook, "resolved_view", return_value=cfg),
            patch.object(moe_hook, "declare_resolution") as declare,
        ):
            moe_hook.handle_flashinfer_rdma(cfg)
        resolution = declare.call_args.kwargs
        self.assertEqual(resolution["moe_runner_backend"], "flashinfer_cutlass")
        self.assertEqual(resolution["speculative_moe_a2a_backend"], "none")
        self.assertTrue(resolution["disable_overlap_schedule"])
        self.assertEqual(
            resolution["cuda_graph_config"].prefill.backend, Backend.DISABLED
        )
        self.assertEqual(_a2a_ep_size(cfg), {"ep_size": 4})
        self.assertEqual(
            _a2a_fusion_adjustments(cfg), {"disable_shared_experts_fusion": True}
        )

    def test_native_mxfp4_dspark_resolves_target_and_draft_rdma(self):
        from sglang.srt.arg_groups.model_override_base import dspark_dp_a2a_backends

        for architecture in ("DeepseekV4ForCausalLM", "DeepseekV41ForCausalLM"):
            self.assertIn("flashinfer_rdma", dspark_dp_a2a_backends(architecture))
            for ep in (4, 8, 16, 32):
                cfg = self.config(
                    tp_size=ep,
                    attn_dp_size=ep,
                    ep_size=ep,
                    moe_runner_backend="flashinfer_mxfp4",
                    speculative_algorithm="DSPARK",
                    speculative_num_draft_tokens=6,
                )
                cfg._model_config.hf_config.architectures = [architecture]
                with (
                    self.subTest(architecture=architecture, ep=ep),
                    patch.object(moe_hook, "resolved_view", return_value=cfg),
                    patch.object(moe_hook, "declare_resolution") as declare,
                ):
                    moe_hook.handle_flashinfer_rdma(cfg)
                resolution = declare.call_args.kwargs
                self.assertEqual(resolution["moe_runner_backend"], "flashinfer_mxfp4")
                self.assertEqual(
                    resolution["speculative_moe_a2a_backend"], "flashinfer_rdma"
                )
                cfg.speculative_moe_a2a_backend = "none"
                with (
                    patch.object(moe_hook, "resolved_view", return_value=cfg),
                    self.assertRaisesRegex(
                        ValueError, "flashinfer_rdma for this speculative model"
                    ),
                ):
                    moe_hook.handle_flashinfer_rdma(cfg)

    def test_native_mxfp4_auto_runner_keeps_other_quantization_defaults(self):
        from sglang.srt.arg_groups.model_overrides import deepseek_v4
        from sglang.srt.environ import envs

        for is_fp4, quantization, expected in (
            (True, "fp8", "flashinfer_mxfp4"),
            (True, None, "flashinfer_mxfp4"),
            (False, "fp8", None),
            (True, "modelopt_fp4", None),
        ):
            cfg = self.config(device="cuda", quantization=quantization)
            hf = SimpleNamespace(
                architectures=["DeepseekV4ForCausalLM"], model_type="deepseek_v41"
            )
            model = SimpleNamespace(is_fp4_experts=is_fp4, nvfp4_moe_meta=None)
            with (
                self.subTest(is_fp4=is_fp4, quantization=quantization),
                patch.object(deepseek_v4, "resolving_view", return_value=cfg),
                patch.object(deepseek_v4, "model_config_of", return_value=model),
                patch.object(
                    deepseek_v4,
                    "get_platform",
                    return_value=SimpleNamespace(
                        is_hip=False, is_sm90=False, is_sm100=False, is_sm120=True
                    ),
                ),
                envs.SGLANG_DSV4_FP4_DEQUANT.override(False),
            ):
                resolved = deepseek_v4._deepseek_v4_overrides(cfg, hf)
            self.assertEqual(resolved.get("moe_runner_backend"), expected)

    def test_unsupported_layouts_and_features_fail_before_allocation(self):
        for change in (
            dict(tp_size=2),
            dict(attn_dp_size=2),
            dict(pp_size=2),
            dict(attn_cp_size=2),
            dict(dcp_size=2),
            dict(moe_dp_size=2),
            dict(enable_two_batch_overlap=True),
            dict(enable_single_batch_overlap=True),
            dict(enable_eplb=True),
            dict(ep_num_redundant_experts=4),
            dict(moe_runner_backend="triton"),
            dict(speculative_moe_a2a_backend="deepep"),
            dict(flashinfer_rdma_max_tokens_per_rank=0),
        ):
            cfg = self.config(**change)
            with (
                self.subTest(change=change),
                patch.object(moe_hook, "resolved_view", return_value=cfg),
                self.assertRaises(ValueError),
            ):
                moe_hook.handle_flashinfer_rdma(cfg)
        cfg = self.config()
        cfg._model_config.hf_config.architectures = ["Qwen3MoeForCausalLM"]
        with (
            patch.object(moe_hook, "resolved_view", return_value=cfg),
            self.assertRaisesRegex(ValueError, "MoE families"),
        ):
            moe_hook.handle_flashinfer_rdma(cfg)

    def test_verify_graph_capacity_counts_all_draft_tokens(self):
        cfg = self.config(max_running_requests=64)
        cfg.cuda_graph_config.decode.max_bs = 16
        with patch.object(moe_hook, "resolved_view", return_value=cfg):
            moe_hook.validate_flashinfer_a2a_token_budget(cfg)
            cfg.flashinfer_rdma_max_tokens_per_rank = 63
            with self.assertRaisesRegex(ValueError, "requires 64 tokens"):
                moe_hook.validate_flashinfer_a2a_token_budget(cfg)
            cfg.cuda_graph_config.decode.backend = Backend.DISABLED
            with self.assertRaisesRegex(ValueError, "requires 64 tokens"):
                moe_hook.validate_flashinfer_a2a_token_budget(cfg)
            cfg.cuda_graph_config.decode.backend = Backend.BREAKABLE
            with self.assertRaisesRegex(ValueError, "full or disabled"):
                moe_hook.validate_flashinfer_a2a_token_budget(cfg)

    def test_full_resolution_caps_unset_admission_without_decode_graphs(self):
        # Exercise real declaration validation and the full resolution order;
        # no model weights or CUDA allocation are needed to resolve this config.
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["DeepseekV3ForCausalLM"],
                        "model_type": "deepseek_v3",
                        "hidden_size": 16,
                        "intermediate_size": 32,
                        "moe_intermediate_size": 16,
                        "n_routed_experts": 8,
                        "n_shared_experts": 1,
                        "num_experts_per_tok": 2,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "num_hidden_layers": 2,
                        "vocab_size": 128,
                        "max_position_embeddings": 2048,
                        "q_lora_rank": None,
                        "kv_lora_rank": 8,
                        "qk_nope_head_dim": 4,
                        "qk_rope_head_dim": 4,
                        "v_head_dim": 4,
                    }
                )
            )
            for algorithm, expected_requests in ((None, 76), ("NEXTN", 16)):
                with self.subTest(algorithm=algorithm):
                    graph_config = CudaGraphConfig()
                    graph_config.decode.backend = Backend.DISABLED
                    cfg = self.config(
                        model_path=folder,
                        device="cuda",
                        speculative_algorithm=algorithm,
                        speculative_num_steps=3 if algorithm else None,
                        speculative_eagle_topk=1 if algorithm else None,
                        speculative_num_draft_tokens=4 if algorithm else None,
                        moe_runner_backend="flashinfer_cutlass",
                        flashinfer_rdma_max_tokens_per_rank=19,
                        cuda_graph_config=graph_config,
                    )
                    self.assertIsNone(cfg.max_running_requests)
                    cfg.resolve_once()
                    self.assertIsNone(cfg.max_running_requests)
                    self.assertEqual(
                        resolution_result(cfg, "max_running_requests"),
                        expected_requests,
                    )


if __name__ == "__main__":
    unittest.main()
