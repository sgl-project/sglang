import json
import os
import subprocess
import sys
import textwrap
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    get_rl_on_policy_target,
    is_tp_invariant_target,
    is_true_on_policy_enabled,
    patch_prefill_only_deterministic_inference_for_cuda_graph,
    resolve_true_on_policy_runtime_policy,
    should_disable_flashinfer_allreduce_fusion,
    should_disable_mlp_allreduce_fusion_for_on_policy,
    should_disable_reduce_scatter_for_on_policy,
    should_use_tp_invariant_row_linear,
    should_use_tp_invariant_tree_all_reduce,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="stage-a-test-cpu")


def _run_server_args_script(argv: list[str]) -> dict[str, object]:
    stubbed_imports = textwrap.dedent("""
        import argparse
        import importlib.machinery
        import json
        import sys
        import types
        from types import SimpleNamespace
        from unittest.mock import patch

        from pydantic import BaseModel

        def install_openai_stubs():
            openai_mod = types.ModuleType("openai")
            openai_types_mod = types.ModuleType("openai.types")
            openai_responses_mod = types.ModuleType("openai.types.responses")
            openai_response_mod = types.ModuleType("openai.types.responses.response")
            openai_tool_mod = types.ModuleType("openai.types.responses.tool")

            openai_mod.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
            openai_types_mod.__spec__ = importlib.machinery.ModuleSpec("openai.types", loader=None)
            openai_responses_mod.__spec__ = importlib.machinery.ModuleSpec(
                "openai.types.responses", loader=None
            )
            openai_response_mod.__spec__ = importlib.machinery.ModuleSpec(
                "openai.types.responses.response", loader=None
            )
            openai_tool_mod.__spec__ = importlib.machinery.ModuleSpec(
                "openai.types.responses.tool", loader=None
            )

            for name in [
                "ResponseFunctionToolCall",
                "ResponseInputItemParam",
                "ResponseOutputItem",
                "ResponseOutputMessage",
                "ResponseOutputText",
                "ResponseReasoningItem",
            ]:
                setattr(openai_responses_mod, name, type(name, (BaseModel,), {}))

            openai_response_mod.ToolChoice = type("ToolChoice", (BaseModel,), {})
            openai_tool_mod.Tool = type("Tool", (BaseModel,), {})

            sys.modules.setdefault("openai", openai_mod)
            sys.modules.setdefault("openai.types", openai_types_mod)
            sys.modules.setdefault("openai.types.responses", openai_responses_mod)
            sys.modules.setdefault("openai.types.responses.response", openai_response_mod)
            sys.modules.setdefault("openai.types.responses.tool", openai_tool_mod)

        install_openai_stubs()

        hf_utils_mod = types.ModuleType("sglang.srt.utils.hf_transformers_utils")
        hf_utils_mod.__spec__ = importlib.machinery.ModuleSpec(
            "sglang.srt.utils.hf_transformers_utils", loader=None
        )
        hf_utils_mod.check_gguf_file = lambda *args, **kwargs: False
        sys.modules.setdefault("sglang.srt.utils.hf_transformers_utils", hf_utils_mod)

        from sglang.srt.server_args import ServerArgs

        def _mock_model_config():
            return SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["Qwen2ForCausalLM"])
            )

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        cli_args = parser.parse_args(ARGV)

        with patch("sglang.srt.server_args.get_device", return_value="cuda"), patch.object(
            ServerArgs, "get_model_config", return_value=_mock_model_config()
        ):
            server_args = ServerArgs.from_cli_args(cli_args)
            server_args._handle_deterministic_inference()

        print(
            json.dumps(
                {
                    "enable_deterministic_inference": server_args.enable_deterministic_inference,
                    "enable_prefill_only_deterministic_inference": server_args.enable_prefill_only_deterministic_inference,
                    "enable_flashinfer_allreduce_fusion": server_args.enable_flashinfer_allreduce_fusion,
                    "rl_on_policy_target": server_args.rl_on_policy_target,
                    "true_on_policy_contract": server_args.true_on_policy_contract,
                    "sampling_backend": server_args.sampling_backend,
                }
            )
        )
        """)

    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    repo_python = "python"
    env["PYTHONPATH"] = (
        f"{repo_python}{os.pathsep}{pythonpath}" if pythonpath else repo_python
    )
    script = f"ARGV = {argv!r}\n{stubbed_imports}"
    completed = subprocess.run(
        ["python", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return json.loads(completed.stdout)


class TestOnPolicyServerArgs(unittest.TestCase):
    def test_cli_parses_prefill_only_deterministic_flag(self):
        result = _run_server_args_script(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "triton",
                "--enable-prefill-only-deterministic-inference",
            ]
        )

        self.assertTrue(result["enable_prefill_only_deterministic_inference"])
        self.assertTrue(result["enable_deterministic_inference"])
        self.assertIsNone(result["rl_on_policy_target"])
        self.assertEqual(result["sampling_backend"], "pytorch")

    def test_cli_accepts_fsdp_and_fsdp_tp_targets(self):
        fsdp_tp_result = _run_server_args_script(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "triton",
                "--rl-on-policy-target",
                "fsdp_tp",
            ]
        )
        self.assertEqual(fsdp_tp_result["rl_on_policy_target"], "fsdp_tp")
        self.assertIsNone(fsdp_tp_result["true_on_policy_contract"])
        self.assertTrue(fsdp_tp_result["enable_deterministic_inference"])

        fsdp_result = _run_server_args_script(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "triton",
                "--rl-on-policy-target",
                "fsdp",
            ]
        )
        self.assertEqual(fsdp_result["rl_on_policy_target"], "fsdp")
        self.assertIsNone(fsdp_result["true_on_policy_contract"])
        self.assertTrue(fsdp_result["enable_deterministic_inference"])

    def test_cli_accepts_explicit_true_on_policy_contract(self):
        result = _run_server_args_script(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "triton",
                "--rl-on-policy-target",
                "fsdp_tp",
                "--true-on-policy-contract",
                QWEN3_DENSE_TRUE_ON_POLICY_V1,
            ]
        )

        self.assertEqual(result["rl_on_policy_target"], "fsdp_tp")
        self.assertEqual(result["true_on_policy_contract"], QWEN3_DENSE_TRUE_ON_POLICY_V1)
        self.assertTrue(result["enable_deterministic_inference"])

    def test_cli_rejects_contract_without_on_policy_target(self):
        with self.assertRaises(subprocess.CalledProcessError):
            _run_server_args_script(
                [
                    "--model-path",
                    "dummy",
                    "--attention-backend",
                    "triton",
                    "--true-on-policy-contract",
                    QWEN3_DENSE_TRUE_ON_POLICY_V1,
                ]
            )

    def test_fsdp_tp_disables_flashinfer_allreduce_fusion(self):
        result = _run_server_args_script(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "triton",
                "--rl-on-policy-target",
                "fsdp_tp",
                "--enable-flashinfer-allreduce-fusion",
            ]
        )
        self.assertFalse(result["enable_flashinfer_allreduce_fusion"])

    def test_fsdp_keeps_flashinfer_allreduce_fusion_available(self):
        result = _run_server_args_script(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "triton",
                "--rl-on-policy-target",
                "fsdp",
                "--enable-flashinfer-allreduce-fusion",
            ]
        )
        self.assertTrue(result["enable_flashinfer_allreduce_fusion"])


class TestDefaultPathUnchanged(unittest.TestCase):
    """Regression: when rl_on_policy_target=None, all helpers return
    'use default path', proving normal serving is unaffected by PR2."""

    def setUp(self):
        self.default_args = SimpleNamespace(rl_on_policy_target=None)

    def test_default_args_no_on_policy(self):
        self.assertIsNone(get_rl_on_policy_target(self.default_args))
        self.assertFalse(is_true_on_policy_enabled(self.default_args))
        self.assertFalse(is_tp_invariant_target(self.default_args))

    def test_default_args_row_linear_uses_quant_method(self):
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                256,
                server_args=self.default_args,
                row_linear_enable_inv=True,
            )
        )

    def test_default_args_tree_allreduce_not_selected(self):
        self.assertFalse(
            should_use_tp_invariant_tree_all_reduce(
                server_args=self.default_args,
                accl_binary_tree_enabled=False,
            )
        )

    def test_default_args_reduce_scatter_available(self):
        self.assertFalse(should_disable_reduce_scatter_for_on_policy(self.default_args))

    def test_default_args_mlp_fusion_available(self):
        self.assertFalse(
            should_disable_mlp_allreduce_fusion_for_on_policy(self.default_args)
        )

    def test_default_args_flashinfer_fusion_available(self):
        self.assertFalse(should_disable_flashinfer_allreduce_fusion(self.default_args))

    def test_default_server_args_cli_no_on_policy_flags(self):
        result = _run_server_args_script(
            ["--model-path", "dummy", "--attention-backend", "triton"]
        )
        self.assertIsNone(result["rl_on_policy_target"])
        self.assertFalse(result["enable_deterministic_inference"])
        self.assertFalse(result["enable_prefill_only_deterministic_inference"])


class TestOnPolicyHelpers(unittest.TestCase):
    def test_tp_invariant_row_linear_selection(self):
        server_args = SimpleNamespace(rl_on_policy_target="fsdp_tp")

        self.assertTrue(
            should_use_tp_invariant_row_linear(
                256,
                server_args=server_args,
                row_linear_enable_inv=True,
            )
        )

    def test_contract_resolver_preserves_legacy_target_behavior(self):
        policy = resolve_true_on_policy_runtime_policy(
            SimpleNamespace(
                rl_on_policy_target="fsdp_tp",
                true_on_policy_contract=None,
            )
        )

        self.assertEqual(policy.contract_name, QWEN3_DENSE_TRUE_ON_POLICY_V1)
        self.assertTrue(policy.enabled)
        self.assertTrue(policy.tp_invariant_row_linear)
        self.assertTrue(policy.deterministic_tree_all_reduce)

    def test_contract_resolver_accepts_explicit_qwen3_dense_contract(self):
        server_args = SimpleNamespace(
            rl_on_policy_target="fsdp",
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        )
        policy = resolve_true_on_policy_runtime_policy(
            server_args
        )

        self.assertTrue(policy.enabled)
        self.assertTrue(policy.force_bfloat16_dense_tensor_math)
        self.assertFalse(policy.tp_invariant_row_linear)
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                96,
                server_args=server_args,
                row_linear_enable_inv=True,
            )
        )
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                256,
                server_args=SimpleNamespace(rl_on_policy_target="fsdp"),
                row_linear_enable_inv=True,
            )
        )

    def test_contract_resolver_rejects_contract_without_target(self):
        with self.assertRaisesRegex(ValueError, "requires --rl-on-policy-target"):
            resolve_true_on_policy_runtime_policy(
                SimpleNamespace(
                    rl_on_policy_target=None,
                    true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
                )
            )

    def test_reduce_scatter_and_fusion_are_disabled_for_any_on_policy_target(self):
        self.assertTrue(
            should_disable_reduce_scatter_for_on_policy(
                SimpleNamespace(rl_on_policy_target="fsdp")
            )
        )
        self.assertTrue(
            should_disable_mlp_allreduce_fusion_for_on_policy(
                SimpleNamespace(rl_on_policy_target="fsdp_tp")
            )
        )
        self.assertFalse(
            should_disable_reduce_scatter_for_on_policy(
                SimpleNamespace(rl_on_policy_target=None)
            )
        )

    def test_tree_all_reduce_selection_requires_fsdp_tp_and_no_accl(self):
        self.assertTrue(
            should_use_tp_invariant_tree_all_reduce(
                server_args=SimpleNamespace(rl_on_policy_target="fsdp_tp"),
                accl_binary_tree_enabled=False,
            )
        )
        self.assertFalse(
            should_use_tp_invariant_tree_all_reduce(
                server_args=SimpleNamespace(rl_on_policy_target="fsdp_tp"),
                accl_binary_tree_enabled=True,
            )
        )
        self.assertFalse(
            should_use_tp_invariant_tree_all_reduce(
                server_args=SimpleNamespace(rl_on_policy_target="fsdp"),
                accl_binary_tree_enabled=False,
            )
        )

    def test_attention_handoff_tree_reduce_uses_attention_tp_group(self):
        from sglang.srt.layers.communicator import CommunicateWithAllReduceAndLayerNormFn

        hidden_states = torch.ones(2, 4)
        residual = torch.full((2, 4), 3.0)

        class FakeNorm:
            def __call__(self, x, residual):
                return x + residual, residual

        with (
            patch(
                "sglang.srt.layers.communicator.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch(
                "sglang.srt.layers.communicator.apply_aiter_all_reduce_fusion",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.communicator.should_use_tp_invariant_tree_all_reduce",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator.attention_tensor_model_parallel_tree_all_reduce",
                side_effect=lambda x: x + 10.0,
            ) as attn_tree_reduce,
            patch(
                "sglang.srt.layers.communicator.tensor_model_parallel_tree_all_reduce",
                side_effect=AssertionError("generic TP tree reduce must not handle attention output"),
            ),
        ):
            output, output_residual = (
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual(
                    hidden_states,
                    residual,
                    forward_batch=None,
                    layernorm=FakeNorm(),
                    context=SimpleNamespace(attn_dp_size=1, cache=None),
                    residual_input_mode=None,
                )
            )

        attn_tree_reduce.assert_called_once()
        torch.testing.assert_close(output, hidden_states + 10.0 + residual)
        torch.testing.assert_close(output_residual, residual)

    def test_prefill_only_cuda_graph_patch_temporarily_disables_on_policy_state(self):
        server_args = SimpleNamespace(
            enable_prefill_only_deterministic_inference=True,
            enable_deterministic_inference=True,
            enable_flashinfer_allreduce_fusion=False,
            rl_on_policy_target="fsdp_tp",
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
            disable_custom_all_reduce=True,
        )
        global_server_args = SimpleNamespace(
            enable_prefill_only_deterministic_inference=True,
            enable_deterministic_inference=True,
            enable_flashinfer_allreduce_fusion=False,
            rl_on_policy_target="fsdp_tp",
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
            disable_custom_all_reduce=True,
        )
        attn_backend = SimpleNamespace(num_splits=1)

        with patch.dict(
            os.environ,
            {
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
                "SGLANG_DISABLE_CUSTOM_ALL_REDUCE": "1",
                "NCCL_ALGO": "allreduce:tree",
            },
            clear=False,
        ):
            batch_pkg = types.ModuleType("sglang.srt.batch_invariant_ops")
            batch_mod = types.ModuleType(
                "sglang.srt.batch_invariant_ops.batch_invariant_ops"
            )
            tp_mod = types.ModuleType("sglang.srt.tp_invariant_ops")
            batch_mod.disable_batch_invariant_mode = unittest.mock.MagicMock()
            batch_mod.enable_batch_invariant_mode = unittest.mock.MagicMock()
            tp_mod.disable_tp_invariant_mode = unittest.mock.MagicMock()
            tp_mod.enable_tp_invariant_mode = unittest.mock.MagicMock()

            with patch.dict(
                sys.modules,
                {
                    "sglang.srt.batch_invariant_ops": batch_pkg,
                    "sglang.srt.batch_invariant_ops.batch_invariant_ops": batch_mod,
                    "sglang.srt.tp_invariant_ops": tp_mod,
                },
            ):
                with patch_prefill_only_deterministic_inference_for_cuda_graph(
                    server_args,
                    attn_backend=attn_backend,
                    global_server_args=global_server_args,
                ) as patched:
                    self.assertTrue(patched)
                    self.assertFalse(server_args.enable_deterministic_inference)
                    self.assertTrue(server_args.enable_flashinfer_allreduce_fusion)
                    self.assertIsNone(server_args.rl_on_policy_target)
                    self.assertIsNone(server_args.true_on_policy_contract)
                    self.assertIsNone(global_server_args.rl_on_policy_target)
                    self.assertIsNone(global_server_args.true_on_policy_contract)
                    self.assertEqual(attn_backend.num_splits, 0)
                    self.assertEqual(
                        os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"], "0"
                    )
                    self.assertEqual(
                        os.environ["SGLANG_DISABLE_CUSTOM_ALL_REDUCE"], "0"
                    )
                    batch_mod.disable_batch_invariant_mode.assert_called_once()
                    tp_mod.disable_tp_invariant_mode.assert_called_once()

                self.assertTrue(server_args.enable_deterministic_inference)
                self.assertFalse(server_args.enable_flashinfer_allreduce_fusion)
                self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")
                self.assertEqual(
                    server_args.true_on_policy_contract,
                    QWEN3_DENSE_TRUE_ON_POLICY_V1,
                )
                self.assertEqual(global_server_args.rl_on_policy_target, "fsdp_tp")
                self.assertEqual(
                    global_server_args.true_on_policy_contract,
                    QWEN3_DENSE_TRUE_ON_POLICY_V1,
                )
                self.assertTrue(server_args.disable_custom_all_reduce)
                self.assertEqual(attn_backend.num_splits, 1)
                self.assertEqual(
                    os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"], "1"
                )
                self.assertEqual(os.environ["SGLANG_DISABLE_CUSTOM_ALL_REDUCE"], "1")
                self.assertEqual(os.environ["NCCL_ALGO"], "allreduce:tree")
                batch_mod.enable_batch_invariant_mode.assert_called_once()
                tp_mod.enable_tp_invariant_mode.assert_called_once()

    def test_row_linear_k_alignment_edge_cases(self):
        server_args = SimpleNamespace(rl_on_policy_target="fsdp_tp")

        self.assertFalse(
            should_use_tp_invariant_row_linear(
                64,
                server_args=server_args,
                row_linear_enable_inv=True,
            ),
        )
        self.assertTrue(
            should_use_tp_invariant_row_linear(
                128,
                server_args=server_args,
                row_linear_enable_inv=True,
            ),
        )
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                300,
                server_args=server_args,
                row_linear_enable_inv=True,
            ),
        )
        self.assertTrue(
            should_use_tp_invariant_row_linear(
                3584,
                server_args=server_args,
                row_linear_enable_inv=True,
            ),
        )

    def test_row_linear_env_var_gate(self):
        server_args = SimpleNamespace(rl_on_policy_target="fsdp_tp")
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                256,
                server_args=server_args,
                row_linear_enable_inv=False,
            )
        )

    def test_flashinfer_allreduce_fusion_helpers(self):
        self.assertTrue(
            should_disable_flashinfer_allreduce_fusion(
                SimpleNamespace(rl_on_policy_target="fsdp_tp")
            )
        )
        self.assertFalse(
            should_disable_flashinfer_allreduce_fusion(
                SimpleNamespace(rl_on_policy_target="fsdp")
            )
        )
        self.assertFalse(
            should_disable_flashinfer_allreduce_fusion(
                SimpleNamespace(rl_on_policy_target=None)
            )
        )

    def test_get_rl_on_policy_target_returns_correct_value(self):
        self.assertEqual(
            get_rl_on_policy_target(SimpleNamespace(rl_on_policy_target="fsdp_tp")),
            "fsdp_tp",
        )
        self.assertEqual(
            get_rl_on_policy_target(SimpleNamespace(rl_on_policy_target="fsdp")),
            "fsdp",
        )
        self.assertIsNone(
            get_rl_on_policy_target(SimpleNamespace(rl_on_policy_target=None))
        )

    def test_is_true_on_policy_enabled_for_both_targets(self):
        self.assertTrue(
            is_true_on_policy_enabled(SimpleNamespace(rl_on_policy_target="fsdp"))
        )
        self.assertTrue(
            is_true_on_policy_enabled(SimpleNamespace(rl_on_policy_target="fsdp_tp"))
        )
        self.assertFalse(
            is_true_on_policy_enabled(SimpleNamespace(rl_on_policy_target=None))
        )

    def test_is_tp_invariant_target_only_fsdp_tp(self):
        self.assertTrue(
            is_tp_invariant_target(SimpleNamespace(rl_on_policy_target="fsdp_tp"))
        )
        self.assertFalse(
            is_tp_invariant_target(SimpleNamespace(rl_on_policy_target="fsdp"))
        )
        self.assertFalse(
            is_tp_invariant_target(SimpleNamespace(rl_on_policy_target=None))
        )

    def test_cuda_graph_patch_noop_when_disabled(self):
        server_args = SimpleNamespace(
            enable_prefill_only_deterministic_inference=False,
            enable_deterministic_inference=True,
            rl_on_policy_target="fsdp_tp",
        )
        with patch_prefill_only_deterministic_inference_for_cuda_graph(
            server_args,
        ) as patched:
            self.assertFalse(patched)
            self.assertTrue(server_args.enable_deterministic_inference)
            self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")

    def test_cuda_graph_patch_noop_when_dvr_verify(self):
        server_args = SimpleNamespace(
            enable_prefill_only_deterministic_inference=True,
            enable_deterministic_inference=True,
            enable_flashinfer_allreduce_fusion=False,
            rl_on_policy_target="fsdp_tp",
            disable_custom_all_reduce=True,
        )
        with patch_prefill_only_deterministic_inference_for_cuda_graph(
            server_args,
            dvr_target_verify_cuda_graph=True,
        ) as patched:
            self.assertFalse(patched)
            self.assertTrue(server_args.enable_deterministic_inference)
            self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")

    def test_tp_invariant_ops_import_is_available(self):
        import sglang.srt.tp_invariant_ops as tp_invariant_ops

        self.assertTrue(hasattr(tp_invariant_ops, "matmul_tp_inv"))

    def test_legacy_on_policy_utils_import_matches_true_on_policy_namespace(self):
        from sglang.srt.layers import on_policy_utils as legacy
        from sglang.srt import true_on_policy

        self.assertIs(
            legacy.should_use_tp_invariant_row_linear,
            true_on_policy.should_use_tp_invariant_row_linear,
        )
        self.assertIs(
            legacy.patch_prefill_only_deterministic_inference_for_cuda_graph,
            true_on_policy.patch_prefill_only_deterministic_inference_for_cuda_graph,
        )
        self.assertTrue(hasattr(torch.ops, "tp_inv_ops"))


if __name__ == "__main__":
    unittest.main()
