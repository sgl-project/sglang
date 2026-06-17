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
    get_true_on_policy_contract,
    is_tp_invariant_target,
    is_true_on_policy_enabled,
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

    def test_true_on_policy_dp_attention_uses_max_len_padding(self):
        try:
            from sglang.srt.layers import dp_attention
        except ModuleNotFoundError as exc:
            if exc.name == "openai":
                self.skipTest(
                    "openai dependency is unavailable in this CPU test environment"
                )
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
                self.skipTest(
                    "openai dependency is unavailable in this CPU test environment"
                )
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
                self.skipTest(
                    "openai dependency is unavailable in this CPU test environment"
                )
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
        self.assertTrue(
            should_disable_reduce_scatter_for_on_policy(self._contract_args(tp_size=1))
        )
        self.assertTrue(
            should_disable_mlp_allreduce_fusion_for_on_policy(
                self._contract_args(tp_size=2)
            )
        )
        self.assertFalse(
            should_disable_reduce_scatter_for_on_policy(
                SimpleNamespace(true_on_policy_contract=None, tp_size=1)
            )
        )

    def test_tree_all_reduce_selection_requires_tp_rollout_and_no_accl(self):
        self.assertTrue(
            should_use_tp_invariant_tree_all_reduce(
                server_args=self._contract_args(tp_size=2),
                accl_binary_tree_enabled=False,
            )
        )
        self.assertFalse(
            should_use_tp_invariant_tree_all_reduce(
                server_args=self._contract_args(tp_size=2),
                accl_binary_tree_enabled=True,
            )
        )
        self.assertFalse(
            should_use_tp_invariant_tree_all_reduce(
                server_args=self._contract_args(tp_size=1),
                accl_binary_tree_enabled=False,
            )
        )

    def test_tree_all_reduce_selection_is_contract_owned(self):
        with patch.dict(os.environ, {"ACCL_BINARY_TREE_ENABLE": "1"}):
            self.assertTrue(
                should_use_tp_invariant_tree_all_reduce(
                    server_args=self._contract_args(tp_size=2),
                )
            )

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
                "sglang.srt.layers.communicator.should_use_tp_invariant_tree_all_reduce",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.communicator.attention_tensor_model_parallel_tree_all_reduce",
                side_effect=lambda x: x + 10.0,
            ) as attn_tree_reduce,
            patch(
                "sglang.srt.layers.communicator.tensor_model_parallel_tree_all_reduce",
                side_effect=AssertionError(
                    "generic TP tree reduce must not handle attention output"
                ),
                create=True,
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

    def test_row_linear_k_alignment_edge_cases(self):
        server_args = self._contract_args(tp_size=2)

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

    def test_row_linear_explicit_override_can_disable(self):
        server_args = self._contract_args(tp_size=2)
        self.assertFalse(
            should_use_tp_invariant_row_linear(
                256,
                server_args=server_args,
                row_linear_enable_inv=False,
            )
        )

    def test_flashinfer_allreduce_fusion_helpers(self):
        self.assertTrue(
            should_disable_flashinfer_allreduce_fusion(self._contract_args(tp_size=2))
        )
        self.assertFalse(
            should_disable_flashinfer_allreduce_fusion(self._contract_args(tp_size=1))
        )
        self.assertFalse(
            should_disable_flashinfer_allreduce_fusion(
                SimpleNamespace(true_on_policy_contract=None, tp_size=1)
            )
        )

    def test_fused_qk_norm_mrope_helper_follows_true_on_policy_contract(self):
        self.assertTrue(
            should_disable_fused_qk_norm_mrope(self._contract_args(tp_size=1))
        )
        self.assertFalse(
            should_disable_fused_qk_norm_mrope(
                SimpleNamespace(true_on_policy_contract=None, tp_size=1)
            )
        )

    def test_is_true_on_policy_enabled_for_both_targets(self):
        self.assertTrue(is_true_on_policy_enabled(self._contract_args(tp_size=1)))
        self.assertTrue(is_true_on_policy_enabled(self._contract_args(tp_size=2)))
        self.assertFalse(
            is_true_on_policy_enabled(
                SimpleNamespace(true_on_policy_contract=None, tp_size=1)
            )
        )

    def test_is_tp_invariant_target_only_fsdp_tp(self):
        self.assertTrue(is_tp_invariant_target(self._contract_args(tp_size=2)))
        self.assertFalse(is_tp_invariant_target(self._contract_args(tp_size=1)))
        self.assertFalse(
            is_tp_invariant_target(
                SimpleNamespace(true_on_policy_contract=None, tp_size=1)
            )
        )

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
        self.assertTrue(hasattr(torch.ops, "tp_inv_ops"))


class TestStableTopKSoftmax(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fused_stable_topk_softmax_matches_reference_ids(self):
        from sglang.srt.tp_invariant_ops import stable_topk, stable_topk_softmax

        logits = torch.tensor(
            [
                [1.0, 1.0, 0.5, -0.25, 0.0, 2.0, 2.0, -1.0],
                [0.0, -0.5, 0.25, 0.25, 0.25, -1.0, 3.0, 2.0],
            ],
            device="cuda",
            dtype=torch.float32,
        )
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        expected_values, expected_ids = stable_topk(scores, 3)
        expected_values = expected_values / expected_values.sum(dim=-1, keepdim=True)

        with patch.dict(
            os.environ,
            {"SGLANG_TRUE_ON_POLICY_STABLE_TOPK_SOFTMAX": "1"},
        ):
            actual_values, actual_ids = stable_topk_softmax(logits, 3)

        torch.testing.assert_close(actual_ids, expected_ids)
        torch.testing.assert_close(actual_values, expected_values, rtol=1e-6, atol=1e-6)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_fused_stable_topk_softmax_can_emit_int32_ids(self):
        from sglang.srt.tp_invariant_ops import stable_topk, stable_topk_softmax

        logits = torch.randn(8, 16, device="cuda", dtype=torch.float32)
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
        expected_values, expected_ids = stable_topk(scores, 4)
        expected_values = expected_values / expected_values.sum(dim=-1, keepdim=True)

        with patch.dict(
            os.environ,
            {"SGLANG_TRUE_ON_POLICY_STABLE_TOPK_SOFTMAX": "1"},
        ):
            actual_values, actual_ids = stable_topk_softmax(
                logits, 4, ids_dtype=torch.int32
            )

        self.assertEqual(actual_ids.dtype, torch.int32)
        torch.testing.assert_close(actual_ids.long(), expected_ids)
        torch.testing.assert_close(actual_values, expected_values, rtol=1e-6, atol=1e-6)


class TestTrueOnPolicyRMSNorm(unittest.TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_rmsnorm_block_size_env_accepts_power_of_two(self):
        from sglang.srt.batch_invariant_ops import true_on_policy_rms_norm

        x = torch.randn(2, 2048, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(2048, device="cuda", dtype=torch.float32)

        with patch.dict(
            os.environ,
            {"SGLANG_TRUE_ON_POLICY_RMSNORM_BLOCK_SIZE": "2048"},
        ):
            out = true_on_policy_rms_norm(x, weight, eps=1e-6)

        self.assertEqual(out.shape, x.shape)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_rmsnorm_block_size_env_rejects_non_power_of_two(self):
        from sglang.srt.batch_invariant_ops import true_on_policy_rms_norm

        x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(128, device="cuda", dtype=torch.float32)

        with patch.dict(
            os.environ,
            {"SGLANG_TRUE_ON_POLICY_RMSNORM_BLOCK_SIZE": "1536"},
        ):
            with self.assertRaisesRegex(ValueError, "positive power of two"):
                true_on_policy_rms_norm(x, weight, eps=1e-6)


if __name__ == "__main__":
    unittest.main()
