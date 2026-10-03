"""Real provider methods test geometry, finalization, and storage with GPU boundaries mocked."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


ROOT = Path(__file__).resolve().parents[4]
LORA_MOE = ROOT / "python/sglang/srt/lora/moe"
PROVIDER = LORA_MOE / "base_gemm_provider"


def _function(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"function {name!r} not found")


class _ContiguousRowStateStub(msgspec.Struct, kw_only=True):
    """Providers subclass the real row state, so the stub must be a Struct too."""


class TestProviderGeometry(CustomTestCase):
    def test_provider_constructs_every_builtin_capability_from_shared_constants(self):
        """Catch missing runtime imports in the attach-time registry builder."""
        from sglang.srt.lora.moe.base_gemm_provider.masked_row_domain import (
            MaskedRowDomainProvider,
            MaskedRowState,
        )
        from sglang.srt.lora.moe.quant_info import MoeLoraBf16QuantInfo

        # Only GPU launch imports are replaced; construct the actual provider
        # against its real base class and quantization metadata.
        injected = {}
        for name, entry in (
            ("activation_delta", "act_delta_masked"),
            ("dispatch_masked", "dispatch_fill_masked_bf16"),
            ("fused_act", "fused_b_act_masked"),
        ):
            module = types.ModuleType(f"sglang.srt.lora.moe.kernels.{name}")
            setattr(module, entry, mock.Mock())
            injected[module.__name__] = module
        act = injected["sglang.srt.lora.moe.kernels.fused_act"]
        with mock.patch.dict(sys.modules, injected):
            provider = MaskedRowDomainProvider(
                MoeLoraBf16QuantInfo(
                    intermediate_size=8,
                    num_local_experts=2,
                    hidden_size=4,
                    w2_weight=torch.empty((2, 4, 8)),
                    w13_weight=torch.empty((2, 16, 4)),
                )
            )

        # One callable per fused stage, bound at construction.
        self.assertIs(provider._fused_act, act.fused_b_act_masked)

        provider.contract = SimpleNamespace(lora_activation_dtype=torch.bfloat16)
        pair_to_row = torch.tensor([2, 0, -1, 3], dtype=torch.int32)
        workspace = MaskedRowState(
            hidden_permuted=torch.empty((2, 4, 4), dtype=torch.bfloat16),
            masked_m=torch.tensor([2, 2], dtype=torch.int32),
            expected_m=2,
            pair_to_row=pair_to_row,
            m_max=4,
            input_buffer_reuse=True,
        )
        activation_rows = torch.randn((2, 4, 8), dtype=torch.bfloat16)
        rows, actual_mapping = provider.mapped_down_lora_a_input(
            workspace, activation_rows
        )
        self.assertEqual(tuple(rows.shape), (8, 8))
        self.assertIs(actual_mapping, pair_to_row)
        self.assertEqual(rows.data_ptr(), activation_rows.data_ptr())


class TestSharedOuterFinalize(CustomTestCase):
    """Execute the host dispatch with real CPU buffers and mocked GPU launches."""

    def setUp(self):
        from sglang.srt.lora.moe.plan import FinalizeFamily, MoeLoraLaunchConfig, Site
        from sglang.srt.lora.workspace import LoraWorkspace

        self.family = FinalizeFamily
        self.site = Site
        spec = importlib.util.spec_from_file_location(
            "_cpu_shared_finalize_provider", PROVIDER / "base.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.provider = module.MoeBaseProvider()
        self.provider.quant_info = SimpleNamespace(num_local_experts=3, hidden_size=5)
        self.provider.contract = SimpleNamespace(lora_delta_dtype=torch.bfloat16)
        self.config = MoeLoraLaunchConfig()
        self.workspace = LoraWorkspace()
        self.workspace.begin_forward(graph_mode=True, is_prefill_graph=True)
        self.row_state = SimpleNamespace(pair_to_row=torch.arange(9))
        self.args = dict(
            down_rows=torch.empty((9, 5)),
            bridge=torch.empty((9, 2), dtype=torch.float32),
            b_down=torch.empty((2, 5, 2)),
            routing=object(),
            topk_weights=torch.empty((3, 3), dtype=torch.float32),
            routed_scaling_factor=2.5,
            output=torch.empty((3, 5)),
            launch_config=self.config,
            workspace=self.workspace,
            token_route=object(),
        )
        self.calls = mock.Mock()
        finalize = types.ModuleType("sglang.srt.lora.moe.kernels.finalize")
        finalize.invoke_shared_one_pass = self.calls.one_pass
        finalize.invoke_shared_token_delta_reduce = self.calls.reduce
        finalize.invoke_shared_token_delta_tail = self.calls.tail
        lora_b = types.ModuleType("sglang.srt.lora.moe.kernels.lora_b")
        lora_b.grouped_lora_b = self.calls.b
        patcher = mock.patch.dict(
            sys.modules, {finalize.__name__: finalize, lora_b.__name__: lora_b}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def invoke(self, family, **changes):
        self.provider.shared_outer_finalize(
            self.row_state, family=family, **(self.args | changes)
        )

    def test_one_pass_has_no_token_route_or_workspace_requirement(self):
        workspace = mock.Mock()
        self.invoke(self.family.SHARED_ONE_PASS, token_route=None, workspace=workspace)
        workspace.tensor.assert_not_called()
        self.assertEqual([c[0] for c in self.calls.mock_calls], ["one_pass"])
        kwargs = self.calls.one_pass.call_args.kwargs
        for name in (
            "down_rows",
            "bridge",
            "b_down",
            "routing",
            "topk_weights",
            "output",
        ):
            self.assertIs(kwargs[name], self.args[name])
        self.assertIs(kwargs["pair_to_row"], self.row_state.pair_to_row)
        self.assertIs(kwargs["config"], self.config.shared_one_pass)
        self.assertEqual(kwargs["num_local_experts"], 3)
        self.assertEqual(kwargs["routed_scaling_factor"], 2.5)

    def test_token_delta_preserves_buffers_configs_and_three_stage_order(self):
        with mock.patch.object(
            self.workspace, "tensor", wraps=self.workspace.tensor
        ) as tensor:
            self.invoke(self.family.SHARED_TOKEN_DELTA)
        self.assertEqual([c[0] for c in self.calls.mock_calls], ["reduce", "b", "tail"])
        self.assertEqual(
            tensor.call_args_list,
            [
                mock.call(
                    "finalize:shared_token_rank",
                    (3, 2),
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                ),
                mock.call(
                    "finalize:shared_token_delta",
                    (3, 5),
                    dtype=torch.bfloat16,
                    device=torch.device("cpu"),
                ),
            ],
        )
        reduce = self.calls.reduce.call_args.kwargs
        b = self.calls.b.call_args
        tail = self.calls.tail.call_args.kwargs
        for name in ("bridge", "routing", "topk_weights"):
            self.assertIs(reduce[name], self.args[name])
        self.assertIs(reduce["config"], self.config.shared_token_delta["reduce"])
        self.assertIs(b.args[0], reduce["token_rank"])
        self.assertIs(b.args[1], self.args["b_down"])
        self.assertIs(b.args[2], tail["token_delta"])
        self.assertIs(b.args[3], self.args["token_route"])
        self.assertEqual(b.kwargs["destination_offsets"], (0,))
        self.assertIs(b.kwargs["pair_bridge"], True)
        self.assertIs(b.kwargs["config"], self.config.for_b(self.site.DOWN))
        for name in ("down_rows", "routing", "topk_weights", "output"):
            self.assertIs(tail[name], self.args[name])
        self.assertIs(tail["pair_to_row"], self.row_state.pair_to_row)
        self.assertIs(tail["config"], self.config.shared_token_delta["tail"])
        self.assertEqual(tail["routed_scaling_factor"], 2.5)
        self.assertNotIn("routed_scaling_factor", reduce)
        self.assertNotIn("routed_scaling_factor", b.kwargs)

    def test_staged_workspace_reuses_warmed_storage(self):
        self.invoke(self.family.SHARED_TOKEN_DELTA)
        rank = self.calls.reduce.call_args.kwargs["token_rank"]
        delta = self.calls.tail.call_args.kwargs["token_delta"]
        self.calls.reset_mock()
        self.invoke(self.family.SHARED_TOKEN_DELTA)
        self.assertEqual(
            rank.data_ptr(), self.calls.reduce.call_args.kwargs["token_rank"].data_ptr()
        )
        self.assertEqual(
            delta.data_ptr(), self.calls.tail.call_args.kwargs["token_delta"].data_ptr()
        )

    def test_non_fp32_weights_fail_before_allocation_or_launch(self):
        for family in (self.family.SHARED_ONE_PASS, self.family.SHARED_TOKEN_DELTA):
            with self.subTest(family=family):
                workspace = mock.Mock()
                with self.assertRaisesRegex(TypeError, "topk_weights must stay FP32"):
                    self.invoke(
                        family,
                        workspace=workspace,
                        topk_weights=self.args["topk_weights"].to(torch.bfloat16),
                    )
                workspace.tensor.assert_not_called()
                self.assertEqual(self.calls.mock_calls, [])

    def test_missing_staged_route_fails_before_allocation_or_launch(self):
        workspace = mock.Mock()
        with self.assertRaisesRegex(ValueError, "shared token route"):
            self.invoke(
                self.family.SHARED_TOKEN_DELTA, token_route=None, workspace=workspace
            )
        workspace.tensor.assert_not_called()
        self.assertEqual(self.calls.mock_calls, [])

    def test_unexpected_family_is_not_treated_as_token_delta(self):
        for family in (self.family.MATERIALIZED, "unknown"):
            with self.subTest(family=family):
                workspace = mock.Mock()
                with self.assertRaisesRegex(ValueError, "not a shared-outer"):
                    self.invoke(family, workspace=workspace)
                workspace.tensor.assert_not_called()
                self.assertEqual(self.calls.mock_calls, [])

    def test_runner_delegates_both_shared_families_and_preserves_materialized(self):
        # Compile the real method without importing the runner's GPU dependencies.
        source = _function((LORA_MOE / "runner.py").read_text(), "_run_finalize")
        namespace = {"FinalizeFamily": self.family, "torch": torch}
        exec("from __future__ import annotations\n" + source, namespace)
        run = namespace["_run_finalize"]
        owner = SimpleNamespace(
            workspace=self.workspace,
            top_k=3,
            hidden_size=5,
            routed_scaling_factor=2.5,
        )
        batch = SimpleNamespace(down_lora_b=torch.empty((2, 1, 5, 2)))
        topk = SimpleNamespace(
            topk_ids=torch.zeros((3, 3), dtype=torch.int32),
            topk_weights=self.args["topk_weights"],
        )
        delta = torch.empty((9, 5))
        for family, into_base in (
            (self.family.SHARED_ONE_PASS, False),
            (self.family.SHARED_TOKEN_DELTA, False),
            (self.family.MATERIALIZED, False),
            (self.family.MATERIALIZED, True),
        ):
            with self.subTest(family=family, into_base=into_base):
                provider = mock.Mock()
                routes = SimpleNamespace(
                    raw=mock.Mock(return_value=self.args["routing"]),
                    shared_token=self.args["token_route"],
                )
                plan = SimpleNamespace(
                    finalize=SimpleNamespace(family=family, is_shared_outer=True),
                    down_b_into_base=into_base,
                )
                result = run(
                    owner,
                    plan,
                    self.config,
                    provider,
                    routes,
                    self.row_state,
                    self.args["output"],
                    self.args["down_rows"],
                    self.args["bridge"],
                    delta,
                    topk,
                    batch,
                    3,
                )
                self.assertIs(result, self.args["output"])
                if family is self.family.MATERIALIZED:
                    provider.finalize.assert_called_once()
                    provider.shared_outer_finalize.assert_not_called()
                    routes.raw.assert_not_called()
                    actual_delta = provider.finalize.call_args.kwargs["lora_delta"]
                    if into_base:
                        self.assertIsNone(actual_delta)
                    else:
                        self.assertEqual(actual_delta.shape, (3, 3, 5))
                        self.assertEqual(actual_delta.data_ptr(), delta.data_ptr())
                else:
                    provider.finalize.assert_not_called()
                    provider.shared_outer_finalize.assert_called_once()
                    routes.raw.assert_called_once_with(True)
                    call = provider.shared_outer_finalize.call_args
                    self.assertIs(call.args[0], self.row_state)
                    for name in (
                        "down_rows",
                        "bridge",
                        "routing",
                        "topk_weights",
                        "output",
                        "workspace",
                        "launch_config",
                        "token_route",
                    ):
                        self.assertIs(call.kwargs[name], self.args[name])
                    self.assertIs(call.kwargs["family"], family)
                    self.assertEqual(call.kwargs["b_down"].shape, (2, 5, 2))
                    self.assertEqual(
                        call.kwargs["b_down"].data_ptr(), batch.down_lora_b.data_ptr()
                    )


if __name__ == "__main__":
    unittest.main()
