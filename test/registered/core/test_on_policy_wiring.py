import json
import os
import subprocess
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    get_rl_on_policy_target,
    get_true_on_policy_contract,
    is_tp_invariant_target,
    is_true_on_policy_enabled,
    patch_prefill_only_deterministic_inference_for_cuda_graph,
    resolve_true_on_policy_runtime_policy,
    should_disable_flashinfer_allreduce_fusion,
    should_disable_fused_qk_norm_mrope,
    should_disable_mlp_allreduce_fusion_for_on_policy,
    should_disable_reduce_scatter_for_on_policy,
    should_use_tp_invariant_row_linear,
    should_use_tp_invariant_tree_all_reduce,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="stage-a-test-cpu")

_PATCH_TARGET = "sglang.srt.server_args.get_global_server_args"


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
                "--true-on-policy-contract",
                QWEN3_DENSE_TRUE_ON_POLICY_V1,
            ]
        )

        self.assertIsNone(result["rl_on_policy_target"])
        self.assertEqual(
            result["true_on_policy_contract"], QWEN3_DENSE_TRUE_ON_POLICY_V1
        )
        self.assertTrue(result["enable_deterministic_inference"])

    def test_contract_tp_rollout_disables_flashinfer_allreduce_fusion(self):
        result = _run_server_args_script(
            [
                "--model-path",
                "dummy",
                "--attention-backend",
                "triton",
                "--tensor-parallel-size",
                "2",
                "--true-on-policy-contract",
                QWEN3_DENSE_TRUE_ON_POLICY_V1,
                "--enable-flashinfer-allreduce-fusion",
            ]
        )
        self.assertFalse(result["enable_flashinfer_allreduce_fusion"])

    def test_legacy_target_keeps_flashinfer_allreduce_fusion_available(self):
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
        self.assertTrue(result["enable_flashinfer_allreduce_fusion"])


def _mock_args(**kwargs):
    defaults = dict(
        rl_on_policy_target=None,
        true_on_policy_contract=None,
        tp_size=1,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _contract_args(*, tp_size: int = 1):
    return _mock_args(
        true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
        tp_size=tp_size,
    )


class TestDefaultPathUnchanged(unittest.TestCase):
    """Default serving must not enter true-on-policy policy paths."""

    def setUp(self):
        self.default_args = _mock_args()

    def test_default_args_no_on_policy(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertIsNone(get_rl_on_policy_target())
            self.assertFalse(is_true_on_policy_enabled())
            self.assertFalse(is_tp_invariant_target())

    def test_default_args_row_linear_uses_quant_method(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(
                should_use_tp_invariant_row_linear(
                    256,
                    row_linear_enable_inv=True,
                )
            )

    def test_default_args_tree_allreduce_not_selected(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(
                should_use_tp_invariant_tree_all_reduce(
                    accl_binary_tree_enabled=False,
                )
            )

    def test_default_args_reduce_scatter_available(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(should_disable_reduce_scatter_for_on_policy())

    def test_default_args_mlp_fusion_available(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(should_disable_mlp_allreduce_fusion_for_on_policy())

    def test_default_args_flashinfer_fusion_available(self):
        with patch(_PATCH_TARGET, return_value=self.default_args):
            self.assertFalse(should_disable_flashinfer_allreduce_fusion())

    def test_default_server_args_cli_no_on_policy_flags(self):
        result = _run_server_args_script(
            ["--model-path", "dummy", "--attention-backend", "triton"]
        )
        self.assertIsNone(result["rl_on_policy_target"])
        self.assertFalse(result["enable_deterministic_inference"])
        self.assertFalse(result["enable_prefill_only_deterministic_inference"])


class TestOnPolicyHelpers(unittest.TestCase):
    def test_tp_invariant_row_linear_selection(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(
                should_use_tp_invariant_row_linear(
                    256,
                    row_linear_enable_inv=True,
                )
            )

    def test_tp_invariant_row_linear_selection_is_contract_owned(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            with patch.dict(os.environ, {"ROW_LINEAR_ENABLE_INV": "0"}):
                self.assertTrue(should_use_tp_invariant_row_linear(256))

    def test_contract_resolver_ignores_legacy_target_without_contract(self):
        policy = resolve_true_on_policy_runtime_policy(
            _mock_args(rl_on_policy_target="fsdp_tp", tp_size=2)
        )

        self.assertIsNone(policy.contract_name)
        self.assertFalse(policy.enabled)
        self.assertFalse(policy.tp_invariant_row_linear)
        self.assertFalse(policy.deterministic_tree_all_reduce)

    def test_contract_resolver_accepts_explicit_qwen3_dense_contract(self):
        args_tp1 = _contract_args(tp_size=1)
        policy = resolve_true_on_policy_runtime_policy(args_tp1)

        self.assertTrue(policy.enabled)
        self.assertTrue(policy.force_bfloat16_dense_tensor_math)
        self.assertFalse(policy.tp_invariant_row_linear)
        with patch(_PATCH_TARGET, return_value=args_tp1):
            self.assertFalse(
                should_use_tp_invariant_row_linear(
                    96,
                    row_linear_enable_inv=True,
                )
            )
            self.assertFalse(
                should_use_tp_invariant_row_linear(
                    256,
                    row_linear_enable_inv=True,
                )
            )

    def test_contract_object_owns_sglang_runtime_policy_values(self):
        contract = get_true_on_policy_contract(QWEN3_DENSE_TRUE_ON_POLICY_V1)

        policy = contract.policy_for(_contract_args(tp_size=2))

        self.assertEqual(contract.schema.name, QWEN3_DENSE_TRUE_ON_POLICY_V1)
        self.assertEqual(contract.schema.model_family, "qwen3_dense")
        self.assertEqual(policy.contract_name, QWEN3_DENSE_TRUE_ON_POLICY_V1)
        self.assertTrue(policy.enabled)
        self.assertTrue(policy.force_bfloat16_dense_tensor_math)
        self.assertTrue(policy.force_bfloat16_lm_head)
        self.assertTrue(policy.disable_reduce_scatter)
        self.assertTrue(policy.disable_mlp_allreduce_fusion)
        self.assertTrue(policy.disable_flashinfer_allreduce_fusion)
        self.assertTrue(policy.tp_invariant_row_linear)
        self.assertTrue(policy.deterministic_tree_all_reduce)
        self.assertTrue(policy.disable_fused_qk_norm_mrope)

    def test_reduce_scatter_and_fusion_are_disabled_for_contract(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertTrue(should_disable_reduce_scatter_for_on_policy())
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(should_disable_mlp_allreduce_fusion_for_on_policy())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(should_disable_reduce_scatter_for_on_policy())

    def test_tree_all_reduce_selection_requires_tp_rollout_and_no_accl(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(
                should_use_tp_invariant_tree_all_reduce(
                    accl_binary_tree_enabled=False,
                )
            )
            self.assertFalse(
                should_use_tp_invariant_tree_all_reduce(
                    accl_binary_tree_enabled=True,
                )
            )
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertFalse(
                should_use_tp_invariant_tree_all_reduce(
                    accl_binary_tree_enabled=False,
                )
            )

    def test_tree_all_reduce_selection_is_contract_owned(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            with patch.dict(os.environ, {"ACCL_BINARY_TREE_ENABLE": "1"}):
                self.assertTrue(should_use_tp_invariant_tree_all_reduce())

    def test_attention_handoff_tree_reduce_uses_attention_tp_group(self):
        from sglang.srt.layers.communicator import (
            CommunicateWithAllReduceAndLayerNormFn,
        )

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
                "sglang.srt.layers.communicator.attention_tensor_model_parallel_all_reduce",
                side_effect=lambda x: x + 10.0,
            ) as attn_tree_reduce,
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

    def test_prefill_only_cuda_graph_patch_only_scopes_attention_splits(self):
        server_args = SimpleNamespace(
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
            with patch_prefill_only_deterministic_inference_for_cuda_graph(
                server_args,
                attn_backend=attn_backend,
            ) as patched:
                self.assertTrue(patched)
                self.assertTrue(server_args.enable_deterministic_inference)
                self.assertFalse(server_args.enable_flashinfer_allreduce_fusion)
                self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")
                self.assertEqual(
                    server_args.true_on_policy_contract,
                    QWEN3_DENSE_TRUE_ON_POLICY_V1,
                )
                self.assertEqual(attn_backend.num_splits, 0)
                self.assertEqual(
                    os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"], "1"
                )
                self.assertEqual(os.environ["SGLANG_DISABLE_CUSTOM_ALL_REDUCE"], "1")
                self.assertEqual(os.environ["NCCL_ALGO"], "allreduce:tree")

            self.assertTrue(server_args.enable_deterministic_inference)
            self.assertFalse(server_args.enable_flashinfer_allreduce_fusion)
            self.assertEqual(server_args.rl_on_policy_target, "fsdp_tp")
            self.assertEqual(
                server_args.true_on_policy_contract,
                QWEN3_DENSE_TRUE_ON_POLICY_V1,
            )
            self.assertTrue(server_args.disable_custom_all_reduce)
            self.assertEqual(attn_backend.num_splits, 1)
            self.assertEqual(os.environ["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"], "1")
            self.assertEqual(os.environ["SGLANG_DISABLE_CUSTOM_ALL_REDUCE"], "1")
            self.assertEqual(os.environ["NCCL_ALGO"], "allreduce:tree")

    def test_row_linear_k_alignment_edge_cases(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertFalse(
                should_use_tp_invariant_row_linear(64, row_linear_enable_inv=True),
            )
            self.assertTrue(
                should_use_tp_invariant_row_linear(128, row_linear_enable_inv=True),
            )
            self.assertFalse(
                should_use_tp_invariant_row_linear(300, row_linear_enable_inv=True),
            )
            self.assertTrue(
                should_use_tp_invariant_row_linear(3584, row_linear_enable_inv=True),
            )

    def test_row_linear_explicit_override_can_disable(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertFalse(
                should_use_tp_invariant_row_linear(256, row_linear_enable_inv=False)
            )

    def test_flashinfer_allreduce_fusion_helpers(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(should_disable_flashinfer_allreduce_fusion())
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertFalse(should_disable_flashinfer_allreduce_fusion())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(should_disable_flashinfer_allreduce_fusion())

    def test_fused_qk_norm_mrope_helper_follows_true_on_policy_contract(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertTrue(should_disable_fused_qk_norm_mrope())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(should_disable_fused_qk_norm_mrope())

    def test_get_rl_on_policy_target_returns_correct_value(self):
        with patch(
            _PATCH_TARGET, return_value=_mock_args(rl_on_policy_target="fsdp_tp")
        ):
            self.assertEqual(get_rl_on_policy_target(), "fsdp_tp")
        with patch(_PATCH_TARGET, return_value=_mock_args(rl_on_policy_target="fsdp")):
            self.assertEqual(get_rl_on_policy_target(), "fsdp")
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertIsNone(get_rl_on_policy_target())

    def test_is_true_on_policy_enabled_for_both_targets(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertTrue(is_true_on_policy_enabled())
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(is_true_on_policy_enabled())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(is_true_on_policy_enabled())

    def test_is_tp_invariant_target_only_fsdp_tp(self):
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=2)):
            self.assertTrue(is_tp_invariant_target())
        with patch(_PATCH_TARGET, return_value=_contract_args(tp_size=1)):
            self.assertFalse(is_tp_invariant_target())
        with patch(_PATCH_TARGET, return_value=_mock_args()):
            self.assertFalse(is_tp_invariant_target())

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
        from sglang.srt import true_on_policy
        from sglang.srt.layers import on_policy_utils as legacy

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
