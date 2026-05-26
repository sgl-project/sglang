import json
import os
import subprocess
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    QWEN3_MOE_TRUE_ON_POLICY_V1,
    get_moe_topk_tiebreak,
    get_true_on_policy_contract,
    is_tp_invariant_target,
    is_true_on_policy_enabled,
    resolve_true_on_policy_runtime_policy,
    should_disable_flashinfer_allreduce_fusion,
    should_disable_mlp_allreduce_fusion_for_on_policy,
    should_disable_reduce_scatter_for_on_policy,
    should_use_deterministic_moe_combine,
    should_use_deterministic_moe_routing,
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
                    "enable_flashinfer_allreduce_fusion": server_args.enable_flashinfer_allreduce_fusion,
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


class TestDefaultPathUnchanged(unittest.TestCase):
    """Default serving must not enter true-on-policy policy paths."""

    def setUp(self):
        self.default_args = SimpleNamespace(
            true_on_policy_contract=None,
            tp_size=1,
        )

    def test_default_args_no_on_policy(self):
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

    def test_default_args_moe_combine_uses_standard_allreduce(self):
        self.assertFalse(should_use_deterministic_moe_combine(self.default_args))

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
        self.assertFalse(result["enable_deterministic_inference"])


class TestOnPolicyHelpers(unittest.TestCase):
    def _contract_args(self, *, tp_size: int = 1) -> SimpleNamespace:
        return SimpleNamespace(
            true_on_policy_contract=QWEN3_DENSE_TRUE_ON_POLICY_V1,
            tp_size=tp_size,
            ep_size=1,
        )

    def _moe_contract_args(
        self, *, tp_size: int = 1, ep_size: int = 1
    ) -> SimpleNamespace:
        return SimpleNamespace(
            true_on_policy_contract=QWEN3_MOE_TRUE_ON_POLICY_V1,
            tp_size=tp_size,
            ep_size=ep_size,
        )

    def test_tp_invariant_row_linear_selection(self):
        self.assertTrue(
            should_use_tp_invariant_row_linear(
                256,
                server_args=self._contract_args(tp_size=2),
                row_linear_enable_inv=True,
            )
        )

    def test_tp_invariant_row_linear_selection_is_contract_owned(self):
        with patch.dict(os.environ, {"ROW_LINEAR_ENABLE_INV": "0"}):
            self.assertTrue(
                should_use_tp_invariant_row_linear(
                    256,
                    server_args=self._contract_args(tp_size=2),
                )
            )

    def test_contract_resolver_no_contract_disables_policy(self):
        policy = resolve_true_on_policy_runtime_policy(
            SimpleNamespace(
                true_on_policy_contract=None,
                tp_size=2,
            )
        )

        self.assertIsNone(policy.contract_name)
        self.assertFalse(policy.enabled)
        self.assertFalse(policy.tp_invariant_row_linear)
        self.assertFalse(policy.deterministic_tree_all_reduce)

    def test_contract_resolver_accepts_explicit_qwen3_dense_contract(self):
        policy = resolve_true_on_policy_runtime_policy(self._contract_args(tp_size=1))

        self.assertTrue(policy.enabled)
        self.assertTrue(policy.force_bfloat16_dense_tensor_math)
        self.assertFalse(policy.tp_invariant_row_linear)
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                96,
                server_args=self._contract_args(tp_size=1),
                row_linear_enable_inv=True,
            )
        )
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                256,
                server_args=self._contract_args(tp_size=1),
                row_linear_enable_inv=True,
            )
        )

    def test_contract_object_owns_sglang_runtime_policy_values(self):
        contract = get_true_on_policy_contract(QWEN3_DENSE_TRUE_ON_POLICY_V1)

        policy = contract.policy_for(self._contract_args(tp_size=2))

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

    def test_qwen3_moe_contract_splits_tp_and_ep_policy_values(self):
        policy = resolve_true_on_policy_runtime_policy(
            self._moe_contract_args(tp_size=1, ep_size=4)
        )

        self.assertEqual(policy.contract_name, QWEN3_MOE_TRUE_ON_POLICY_V1)
        self.assertTrue(policy.enabled)
        self.assertFalse(policy.tp_invariant_row_linear)
        self.assertFalse(policy.deterministic_tree_all_reduce)
        self.assertTrue(policy.deterministic_moe_routing)
        self.assertEqual(policy.moe_topk_tiebreak, "stable_sort")
        self.assertTrue(policy.ep_invariant_moe)
        self.assertTrue(policy.deterministic_moe_dispatch)
        self.assertTrue(policy.deterministic_moe_combine)
        self.assertTrue(should_use_deterministic_moe_routing(self._moe_contract_args()))
        self.assertTrue(
            should_use_deterministic_moe_combine(
                self._moe_contract_args(tp_size=1, ep_size=4)
            )
        )
        self.assertEqual(
            get_moe_topk_tiebreak(self._moe_contract_args()), "stable_sort"
        )

    def test_qwen3_moe_rollout_tp_still_enables_tp_policy_values(self):
        policy = resolve_true_on_policy_runtime_policy(
            self._moe_contract_args(tp_size=8, ep_size=1)
        )

        self.assertTrue(policy.tp_invariant_row_linear)
        self.assertTrue(policy.deterministic_tree_all_reduce)
        self.assertFalse(policy.ep_invariant_moe)

    def test_qwen3_moe_attention_uses_dense_qk_dtype_contract(self):
        repo_root = Path(__file__).resolve().parents[3]
        qwen3_moe_source = (
            repo_root / "python" / "sglang" / "srt" / "models" / "qwen3_moe.py"
        ).read_text()

        self.assertIn("weight_dtype=torch.float32", qwen3_moe_source)
        self.assertIn("fp32_residual=True", qwen3_moe_source)
        self.assertIn("override_orig_dtype=torch.float32", qwen3_moe_source)
        self.assertIn(
            "hidden_states.dtype != self.qkv_proj.weight.dtype", qwen3_moe_source
        )
        self.assertIn("q = q.to(v.dtype)", qwen3_moe_source)
        self.assertIn("k = k.to(v.dtype)", qwen3_moe_source)

    def test_qwen3_moe_experts_use_weight_dtype_under_deterministic_routing(self):
        repo_root = Path(__file__).resolve().parents[3]
        qwen3_moe_source = (
            repo_root / "python" / "sglang" / "srt" / "models" / "qwen3_moe.py"
        ).read_text()

        self.assertIn('getattr(self.experts, "w13_weight", None)', qwen3_moe_source)
        self.assertIn('getattr(self.gate, "weight", None)', qwen3_moe_source)
        self.assertIn("router_hidden_states.to(gate_weight.dtype)", qwen3_moe_source)
        self.assertIn("expert_hidden_states.to(expert_weight.dtype)", qwen3_moe_source)

    def test_qwen3_moe_ep_combine_uses_tree_reduce_under_contract(self):
        repo_root = Path(__file__).resolve().parents[3]
        qwen3_moe_source = (
            repo_root / "python" / "sglang" / "srt" / "models" / "qwen3_moe.py"
        ).read_text()

        self.assertIn("is_true_on_policy_enabled", qwen3_moe_source)
        self.assertIn("moe_expert_parallel_tree_all_reduce", qwen3_moe_source)


if __name__ == "__main__":
    unittest.main()
