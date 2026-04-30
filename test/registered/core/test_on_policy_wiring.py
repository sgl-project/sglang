import json
import os
import subprocess
import textwrap
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.true_on_policy import (
    QWEN3_DENSE_TRUE_ON_POLICY_V1,
    QWEN3_MOE_TRUE_ON_POLICY_V1,
    get_moe_topk_tiebreak,
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


def _moe_contract_args(*, tp_size: int = 1, ep_size: int = 1):
    return _mock_args(
        true_on_policy_contract=QWEN3_MOE_TRUE_ON_POLICY_V1,
        tp_size=tp_size,
        ep_size=ep_size,
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

    def test_default_args_moe_combine_uses_standard_allreduce(self):
        self.assertFalse(should_use_deterministic_moe_combine(self.default_args))

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

    def test_qwen3_moe_contract_splits_tp_and_ep_policy_values(self):
        policy = resolve_true_on_policy_runtime_policy(
            _moe_contract_args(tp_size=1, ep_size=4)
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
        self.assertTrue(should_use_deterministic_moe_routing(_moe_contract_args()))
        self.assertTrue(
            should_use_deterministic_moe_combine(
                _moe_contract_args(tp_size=1, ep_size=4)
            )
        )
        self.assertEqual(get_moe_topk_tiebreak(_moe_contract_args()), "stable_sort")

    def test_qwen3_moe_rollout_tp_still_enables_tp_policy_values(self):
        policy = resolve_true_on_policy_runtime_policy(
            _moe_contract_args(tp_size=8, ep_size=1)
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
        self.assertIn("should_force_bfloat16_dense_tensor_math", qwen3_moe_source)
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

        self.assertIn("should_use_deterministic_moe_combine", qwen3_moe_source)
        self.assertIn("moe_expert_parallel_tree_all_reduce", qwen3_moe_source)

    def test_true_on_policy_dp_attention_uses_max_len_padding(self):
        try:
            from sglang.srt.layers import dp_attention
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                self.skipTest("openai dependency is unavailable in this CPU test environment")
            raise

        with (
            patch(
                "sglang.srt.layers.dp_attention.get_attention_dp_size",
                return_value=4,
            ),
            patch(
                "sglang.srt.true_on_policy.is_true_on_policy_enabled",
                return_value=True,
            ),
        ):
            mode = dp_attention.DpPaddingMode.get_dp_padding_mode(
                is_extend_in_batch=True,
                global_num_tokens=[128, 0, 0, 0],
            )

        self.assertEqual(mode, dp_attention.DpPaddingMode.MAX_LEN)

    def test_dp_attention_decode_post_forward_trims_sampling_positions(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=1,
            input_ids=torch.tensor([10, 0, 0], dtype=torch.int64),
            req_pool_indices=torch.tensor([7, 0, 0], dtype=torch.int64),
            seq_lens=torch.tensor([12, 0, 0], dtype=torch.int64),
            out_cache_loc=torch.tensor([3, 0, 0], dtype=torch.int64),
            seq_lens_sum=12,
            seq_lens_cpu=torch.tensor([12, 0, 0], dtype=torch.int64),
            positions=torch.tensor([11, 0, 0], dtype=torch.int32),
            global_num_tokens_cpu=[3],
            global_num_tokens_for_logprob_cpu=[3],
        )
        batch._original_batch_size = 1
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.randn(3, 8),
            hidden_states=torch.randn(3, 4),
        )

        batch.post_forward_mlp_sync_batch(logits_output)

        self.assertEqual(batch.positions.shape[0], 1)
        self.assertEqual(batch.seq_lens.shape[0], 1)
        self.assertEqual(batch.req_pool_indices.shape[0], 1)
        self.assertEqual(batch.seq_lens_cpu.shape[0], 1)
        self.assertEqual(logits_output.next_token_logits.shape[0], 1)
        self.assertEqual(logits_output.hidden_states.shape[0], 1)

    def test_empty_idle_rank_stays_idle_under_max_len_mlp_sync(self):
        try:
            from sglang.srt.layers.dp_attention import DpPaddingMode
            from sglang.srt.model_executor.forward_batch_info import (
                ForwardBatch,
                ForwardMode,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                self.skipTest("openai dependency is unavailable in this CPU test environment")
            raise

        batch = ForwardBatch(
            forward_mode=ForwardMode.IDLE,
            batch_size=0,
            input_ids=torch.empty(0, dtype=torch.int64),
            req_pool_indices=torch.empty(0, dtype=torch.int64),
            seq_lens=torch.empty(0, dtype=torch.int64),
            out_cache_loc=torch.empty(0, dtype=torch.int64),
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty(0, dtype=torch.int64),
            positions=torch.empty(0, dtype=torch.int32),
            global_num_tokens_cpu=[16, 0, 0, 0],
            global_num_tokens_gpu=torch.zeros(4, dtype=torch.int64),
            global_num_tokens_for_logprob_cpu=[16, 0, 0, 0],
            global_num_tokens_for_logprob_gpu=torch.zeros(4, dtype=torch.int64),
            is_extend_in_batch=True,
            lora_ids=[],
        )
        model_runner = SimpleNamespace(
            attn_backend=SimpleNamespace(get_cuda_graph_seq_len_fill_value=lambda: 1),
            is_draft_worker=False,
        )

        with (
            patch(
                "sglang.srt.model_executor.forward_batch_info.get_attention_tp_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.get_attention_cp_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.get_attention_dp_rank",
                return_value=1,
            ),
            patch(
                "sglang.srt.model_executor.forward_batch_info.DpPaddingMode.get_dp_padding_mode",
                return_value=DpPaddingMode.MAX_LEN,
            ),
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap.TboForwardBatchPreparer.prepare"
            ),
        ):
            batch.prepare_mlp_sync_batch(model_runner)

        self.assertTrue(batch.forward_mode.is_idle())
        self.assertEqual(batch.batch_size, 16)
        self.assertEqual(batch.seq_lens_cpu.shape[0], 16)
        self.assertFalse(hasattr(batch, "_original_forward_mode"))

    def test_layer_postprocess_honors_on_policy_reduce_scatter_disable(self):
        try:
            from sglang.srt.layers.communicator import (
                LayerCommunicator,
                LayerScatterModes,
                ScatterMode,
            )
            from sglang.srt.layers.dp_attention import DpPaddingMode
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                self.skipTest("openai dependency is unavailable in this CPU test environment")
            raise

        communicator = object.__new__(LayerCommunicator)
        communicator.allow_reduce_scatter = True
        communicator.is_last_layer = False
        communicator.layer_scatter_modes = LayerScatterModes(
            layer_input_mode=ScatterMode.TP_ATTN_FULL,
            attn_mode=ScatterMode.TP_ATTN_FULL,
            mlp_mode=ScatterMode.FULL,
            middle_residual_mode=ScatterMode.TP_ATTN_FULL,
            layer_output_mode=ScatterMode.TP_ATTN_FULL,
        )
        communicator._context = SimpleNamespace()
        communicator._communicate_summable_tensor_pair_fn = lambda **kwargs: kwargs[
            "allow_reduce_scatter"
        ]
        forward_batch = SimpleNamespace(
            dp_padding_mode=DpPaddingMode.MAX_LEN,
        )

        with patch(
            "sglang.srt.layers.communicator.should_disable_reduce_scatter_for_on_policy",
            return_value=True,
        ):
            allow_reduce_scatter = communicator.postprocess_layer(
                hidden_states=torch.empty(1, 1),
                residual=torch.empty(1, 1),
                forward_batch=forward_batch,
            )

        self.assertFalse(allow_reduce_scatter)

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

    def test_dp_attention_gather_uses_post_norm_dtype(self):
        from sglang.srt.layers.communicator import (
            CommunicateWithAllReduceAndLayerNormFn,
        )

        hidden_states = torch.ones(2, 4, dtype=torch.bfloat16)
        residual = torch.full((2, 4), 3.0, dtype=torch.bfloat16)
        captured_dtype = None

        class FakeNorm:
            def __call__(self, x, residual):
                x = x.float() + residual.float()
                return x, x

        def fake_global_dp_buffer(dtype=None):
            nonlocal captured_dtype
            captured_dtype = dtype
            return torch.empty(8, 4, dtype=dtype or torch.bfloat16)

        def fake_dp_gather_partial(global_tokens, local_tokens, forward_batch):
            self.assertEqual(global_tokens.dtype, local_tokens.dtype)
            global_tokens[: local_tokens.shape[0]].copy_(local_tokens)

        with (
            patch(
                "sglang.srt.layers.communicator.get_attn_tp_context",
                return_value=SimpleNamespace(input_scattered=False),
            ),
            patch(
                "sglang.srt.layers.communicator.use_symmetric_memory",
                side_effect=lambda *args, **kwargs: nullcontext(),
            ),
            patch(
                "sglang.srt.layers.communicator.get_tp_group",
                return_value=object(),
            ),
            patch(
                "sglang.srt.layers.communicator.is_allocation_symmetric",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.communicator.get_global_dp_buffer",
                side_effect=fake_global_dp_buffer,
            ),
            patch(
                "sglang.srt.layers.communicator.dp_gather_partial",
                side_effect=fake_dp_gather_partial,
            ),
        ):
            output, output_residual = (
                CommunicateWithAllReduceAndLayerNormFn._gather_hidden_states_and_residual(
                    hidden_states,
                    residual,
                    forward_batch=None,
                    layernorm=FakeNorm(),
                    context=SimpleNamespace(
                        attn_dp_size=4,
                        attn_tp_size=1,
                        attn_tp_rank=0,
                        cache=None,
                    ),
                    residual_input_mode=None,
                )
            )

        self.assertEqual(captured_dtype, torch.float32)
        self.assertEqual(output.dtype, torch.float32)
        self.assertEqual(output_residual.dtype, torch.float32)

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


class TestOnPolicyLinearWiring(unittest.TestCase):
    def test_unquantized_linear_uses_explicit_batch_invariant_path(self):
        try:
            from sglang.srt.layers.quantization import unquant as unquant_module
            from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                self.skipTest("openai dependency is unavailable in this CPU test environment")
            raise

        method = UnquantizedLinearMethod()
        layer = SimpleNamespace(weight=torch.empty(3, 2))
        x = torch.empty(4, 2)
        sentinel = torch.empty(4, 3)

        with patch.object(
            unquant_module, "_should_use_batch_invariant_linear", return_value=True
        ), patch.object(
            unquant_module, "_apply_batch_invariant_linear", return_value=sentinel
        ) as apply_batch_invariant_linear:
            output = method.apply(layer, x)

        self.assertEqual(output.data_ptr(), sentinel.data_ptr())
        self.assertEqual(output.shape, sentinel.shape)
        apply_batch_invariant_linear.assert_called_once_with(layer, x, None)

    def test_batch_invariant_linear_casts_weight_to_input_dtype(self):
        try:
            from sglang.srt.layers.quantization import unquant as unquant_module
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                self.skipTest("openai dependency is unavailable in this CPU test environment")
            raise

        layer = SimpleNamespace(
            weight=torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16
            )
        )
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            dtype=torch.float32,
        )

        with patch(
            "sglang.srt.batch_invariant_ops.batch_invariant_ops.matmul_persistent",
            side_effect=AssertionError("fp32 router linears must not JIT persistent matmul"),
        ):
            output = unquant_module._apply_batch_invariant_linear(layer, x, None)

        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, torch.einsum("bi,oi->bo", x, layer.weight.float()))

    def test_batch_invariant_linear_uses_persistent_for_non_fp32(self):
        try:
            from sglang.srt.layers.quantization import unquant as unquant_module
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                self.skipTest("openai dependency is unavailable in this CPU test environment")
            raise

        layer = SimpleNamespace(weight=torch.empty(3, 2, dtype=torch.bfloat16))
        x = torch.empty(4, 2, dtype=torch.bfloat16)
        sentinel = torch.empty(4, 3, dtype=torch.bfloat16)

        def fake_matmul(a, b, bias):
            self.assertEqual(a.dtype, torch.bfloat16)
            self.assertEqual(b.dtype, torch.bfloat16)
            self.assertIsNone(bias)
            return sentinel

        with patch(
            "sglang.srt.batch_invariant_ops.batch_invariant_ops.matmul_persistent",
            side_effect=fake_matmul,
        ):
            output = unquant_module._apply_batch_invariant_linear(layer, x, None)

        self.assertEqual(output.data_ptr(), sentinel.data_ptr())
        self.assertEqual(output.shape, sentinel.shape)


if __name__ == "__main__":
    unittest.main()
