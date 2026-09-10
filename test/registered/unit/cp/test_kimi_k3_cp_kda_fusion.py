"""K3 CP keeps KDA's full-TP deferred reduction and MLA's native subgroup."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.layers.layer_boundary import BatchVariant, SumGroup
from sglang.srt.models import kimi_k3
from sglang.srt.models.kimi_k3 import KimiK3DeltaAttention
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.boundary_fixtures import build_stages, parallel_of, planning
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
_K3 = "sglang.srt.models.kimi_k3."


class TestKimiK3CPKDAFusion(CustomTestCase):
    def _fusion_enabled(self, *, cp_size, linear_cp, kda, attn_res_size=4, fusion=True):
        with (
            get_context().override_server_args(enable_linear_attn_cp=linear_cp),
            get_parallel().override(
                tp_size=8, tp_rank=0, moe_tp_size=8, attn_tp_size=8 // cp_size,
                attn_cp_size=cp_size, attn_cp_rank=0, attn_tp_rank=0,
            ),
            patch(_K3 + "k3_ar_fusion.enabled", return_value=fusion),
        ):
            return kimi_k3._fuses_attn_all_reduce(
                SimpleNamespace(attn_res_block_size=attn_res_size), is_kda=kda
            )

    def test_only_kda_keeps_full_tp_fusion_under_cp(self):
        for cp_size in (1, 2, 4, 8):
            for kda in (False, True):
                with self.subTest(cp_size=cp_size, kda=kda):
                    self.assertEqual(
                        self._fusion_enabled(cp_size=cp_size, linear_cp=cp_size > 1, kda=kda),
                        kda or cp_size == 1,
                    )
        for kwargs in (
            dict(cp_size=2, linear_cp=False, kda=True),
            dict(cp_size=2, linear_cp=True, kda=True, attn_res_size=None),
            dict(cp_size=2, linear_cp=True, kda=True, fusion=False),
        ):
            self.assertFalse(self._fusion_enabled(**kwargs))

    def test_cp_stage_plans_keep_residual_rows_and_matching_head_sums(self):
        for cp_size in (2, 4, 8):
            for kda in (False, True):
                with self.subTest(cp_size=cp_size, kda=kda):
                    config = SimpleNamespace(
                        is_moe=False, attn_res_block_size=None, is_kda_layer=lambda _: kda
                    )
                    with (
                        get_context().override_server_args(enable_linear_attn_cp=True),
                        get_parallel().override(
                            tp_size=8, tp_rank=0, moe_tp_size=8, attn_tp_size=8 // cp_size,
                            attn_cp_size=cp_size, attn_cp_rank=0, attn_tp_rank=0,
                        ),
                        planning(parallel_of(attn_dp=1, attn_tp=8 // cp_size, attn_cp=cp_size, enable_prefill_cp=True), dsa_cp=True),
                    ):
                        attn, ffn = kimi_k3._stage_facts(config, 0, sp_collective=False)
                        attention, feed_forward = build_stages((attn, nn.Identity()), (ffn, nn.Identity()))
                    cp_attn = attention.plan.paths[BatchVariant.CONTEXT_PARALLEL].entry
                    cp_ffn = feed_forward.plan.paths[BatchVariant.CONTEXT_PARALLEL].entry
                    self.assertEqual(cp_attn.input_rows.sharded, frozenset())
                    self.assertEqual(cp_ffn.input_rows.sharded, frozenset())
                    self.assertIsNone(cp_attn.input_move)
                    expected = SumGroup.TP if kda else SumGroup.ATTN_TP
                    if cp_size == 8:
                        expected = SumGroup.ATTN_CP if kda else None
                    self.assertIs(cp_ffn.declared_sum, expected)

    def test_kda_projection_defers_full_tp_sum_only_with_caller_output_support(self):
        config = KimiLinearConfig(
            v_head_dim=8,
            linear_attn_config=dict(
                num_heads=16,
                head_dim=8,
                short_conv_kernel_size=4,
                kda_layers=[1, 2],
                full_attn_layers=[3, 4],
            ),
        )
        for supports_output in (False, True):

            def linear(*args, **kwargs):
                quant_method = SimpleNamespace()
                if supports_output:
                    quant_method.apply_into = Mock()
                return SimpleNamespace(
                    weight=nn.Parameter(torch.empty(8, 4)),
                    bias=None,
                    quant_method=quant_method,
                    reduce_results=kwargs.get("reduce_results", True),
                    use_dp_attention_reduce=kwargs.get(
                        "use_dp_attention_reduce", False
                    ),
                )

            with self.subTest(supports_output=supports_output), ExitStack() as stack:
                stack.enter_context(
                    get_context().override_server_args(enable_linear_attn_cp=True)
                )
                stack.enter_context(
                    get_parallel().override(
                        tp_size=8,
                        moe_tp_size=8,
                        tp_rank=6,
                        attn_tp_size=2,
                        attn_tp_rank=0,
                        attn_cp_size=4,
                        attn_cp_rank=3,
                    )
                )
                for name in (
                    "QKVParallelLinear",
                    "ReplicatedLinear",
                    "ColumnParallelLinear",
                    "MergedColumnParallelLinear",
                    "RowParallelLinear",
                ):
                    stack.enter_context(patch(_K3 + name, side_effect=linear))
                stack.enter_context(patch(_K3 + "FusedRMSNormGated"))
                stack.enter_context(patch(_K3 + "RadixLinearAttention"))
                layer = KimiK3DeltaAttention(
                    0, 64, config, quant_config=Mock(), all_reduce_fusion=True
                )
                self.assertEqual(layer.attn_tp_size, 8)
                self.assertEqual(layer.attn_tp_rank, 6)
                self.assertEqual(layer.all_reduce_fusion, supports_output)
                self.assertFalse(layer.o_proj.reduce_results)
                self.assertFalse(layer.o_proj.use_dp_attention_reduce)

    def test_deferred_sum_adds_full_sequence_residual_once(self):
        hidden = torch.arange(56, dtype=torch.float32).reshape(7, 8)
        partial = hidden / 8
        residual = torch.full_like(hidden, 3)
        with (
            patch(_K3 + "k3_ar_fusion.find_mc_ptr", return_value=1),
            patch(_K3 + "k3_ar_fusion.all_reduce", side_effect=lambda x, r: x * 8 + r) as reduce,
        ):
            output = kimi_k3._k3_all_reduce_add(partial, residual, SimpleNamespace())
        reduce.assert_called_once()
        self.assertIs(reduce.call_args.args[0], partial)
        self.assertIs(reduce.call_args.args[1], residual)
        torch.testing.assert_close(output, hidden + residual)
        with patch(_K3 + "k3_ar_fusion.find_mc_ptr", return_value=None):
            self.assertIsNone(kimi_k3._k3_all_reduce_add(partial, residual, SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
