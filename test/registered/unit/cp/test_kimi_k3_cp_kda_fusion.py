"""K3 CP keeps KDA's full-TP deferred reduction and MLA's native subgroup."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.configs.kimi_linear import KimiLinearConfig
from sglang.srt.models.kimi_k3 import KimiK3DecoderLayer, KimiK3DeltaAttention
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
_K3 = "sglang.srt.models.kimi_k3."


class _Attention(nn.Module):
    def __init__(self, *, all_reduce_fusion=False, **kwargs):
        super().__init__()
        self.all_reduce_fusion = all_reduce_fusion


class _MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, *, prefix_sum=None, **kwargs):
        return x if prefix_sum is None else x + prefix_sum


class TestKimiK3CPKDAFusion(CustomTestCase):
    def _decoder(self, *, cp_size, linear_cp, kda, attn_res_size=4, fusion=True):
        config = SimpleNamespace(
            hidden_size=8,
            is_moe=False,
            is_kda_layer=lambda _: kda,
            attn_res_block_size=attn_res_size,
            intermediate_size=16,
            hidden_act="silu",
            activation_situ_beta=None,
            activation_situ_linear_beta=None,
            rms_norm_eps=1e-5,
        )
        a2a = SimpleNamespace(
            **{
                f"is_{name}": lambda: False
                for name in ("megamoe", "deepep", "mooncake", "ascend_fuseep", "mori")
            }
        )
        with ExitStack() as stack:
            stack.enter_context(
                get_context().override_server_args(
                    enable_linear_attn_cp=linear_cp,
                )
            )
            stack.enter_context(
                get_parallel().override(
                    tp_size=8,
                    attn_tp_size=8 // cp_size,
                    attn_cp_size=cp_size,
                    attn_cp_rank=0,
                    attn_tp_rank=0,
                )
            )
            stack.enter_context(patch(_K3 + "get_moe_a2a_backend", return_value=a2a))
            stack.enter_context(
                patch(_K3 + "is_dp_attention_enabled", return_value=False)
            )
            stack.enter_context(patch(_K3 + "require_mlp_sync", return_value=False))
            stack.enter_context(
                patch(_K3 + "k3_ar_fusion.enabled", return_value=fusion)
            )
            for name in ("KimiK3DeltaAttention", "KimiK3MLAAttention"):
                stack.enter_context(patch(_K3 + name, _Attention))
            stack.enter_context(patch(_K3 + "KimiK3MLP", _MLP))
            for name in ("RMSNorm", "ReplicatedLinear"):
                stack.enter_context(
                    patch(_K3 + name, side_effect=lambda *a, **kw: nn.Identity())
                )
            return KimiK3DecoderLayer(config, layer_idx=1)

    def test_only_kda_keeps_full_tp_fusion_under_cp(self):
        for cp_size in (1, 2, 4, 8):
            for kda in (False, True):
                with self.subTest(cp_size=cp_size, kda=kda):
                    layer = self._decoder(
                        cp_size=cp_size, linear_cp=cp_size > 1, kda=kda
                    )
                    self.assertEqual(layer.all_reduce_fusion, kda or cp_size == 1)
                    self.assertEqual(
                        layer.self_attn.all_reduce_fusion, layer.all_reduce_fusion
                    )
        # An attention subgroup without the full-TP KDA ownership flag, no
        # attention residuals, or unavailable native kernels retains fallback.
        for kwargs in (
            dict(cp_size=2, linear_cp=False, kda=True),
            dict(cp_size=2, linear_cp=True, kda=True, attn_res_size=None),
            dict(cp_size=2, linear_cp=True, kda=True, fusion=False),
        ):
            self.assertFalse(self._decoder(**kwargs).all_reduce_fusion)

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
                self.assertEqual(layer.o_proj.reduce_results, not supports_output)
                self.assertFalse(layer.o_proj.use_dp_attention_reduce)

    def test_deferred_sum_adds_full_sequence_residual_once(self):
        layer = self._decoder(cp_size=4, linear_cp=True, kda=True)
        hidden = torch.arange(56, dtype=torch.float32).reshape(7, 8)
        partial = hidden / 8
        residual = torch.full_like(hidden, 3)
        reduced = hidden + residual
        attn_res = Mock()
        attn_res.forward.side_effect = [(hidden, residual), (reduced, None)]
        with (
            patch.object(layer, "_run_self_attn", return_value=partial),
            patch(
                _K3 + "k3_ar_fusion.all_reduce", side_effect=lambda x, r: x * 8 + r
            ) as reduce,
        ):
            output, prefix, sharded = layer._forward_attn_residual(
                torch.arange(7),
                hidden,
                residual,
                attn_res,
                SimpleNamespace(),
                None,
                False,
                False,
            )
        reduce.assert_called_once()
        self.assertIs(reduce.call_args.args[0], partial)
        self.assertIs(reduce.call_args.args[1], residual)
        torch.testing.assert_close(attn_res.forward.call_args_list[1].args[0], reduced)
        self.assertIsNone(attn_res.forward.call_args_list[1].args[1])
        torch.testing.assert_close(output, reduced)
        self.assertIsNone(prefix)
        self.assertFalse(sharded)


if __name__ == "__main__":
    unittest.main()
