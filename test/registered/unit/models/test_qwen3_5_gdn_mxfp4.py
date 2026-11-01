"""CPU-only routing tests for the Qwen3.5 GDN MXFP4 handoff and prefill views."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    QWEN38_GDN_HEAD_DIM,
    QWEN38_GDN_NUM_QK_HEADS,
    QWEN38_GDN_NUM_V_HEADS,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3_5
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _OutProj:
    def __init__(self, scheme):
        self.scheme = scheme
        self.last_input = None

    def __call__(self, value):
        self.last_input = value
        return value, None


def _make_norm(head_dim: int):
    return SimpleNamespace(
        group_size=None,
        norm_before_gate=True,
        bias=None,
        weight=torch.empty(head_dim, dtype=torch.bfloat16),
        eps=1e-6,
        activation="swish",
    )


class TestQwen35GDNMXFP4Routing(unittest.TestCase):
    def test_prequantized_tuple_bypasses_dense_norm_output(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z_storage = torch.empty(
            (num_tokens, num_heads, head_dim * 2), dtype=torch.bfloat16
        )
        z = z_storage[..., head_dim:]
        packed = torch.empty((num_tokens, num_heads * head_dim // 2), dtype=torch.uint8)
        scales = torch.empty((256, num_heads * head_dim // 32), dtype=torch.uint8)
        prepare_input = Mock(return_value=(packed, scales))
        out_proj = _OutProj(
            SimpleNamespace(prepare_fused_rmsnorm_gated_input=prepare_input)
        )
        norm = _make_norm(head_dim)
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        actual = Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        self.assertIs(actual[0], packed)
        self.assertIs(actual[1], scales)
        self.assertIs(out_proj.last_input[0], packed)
        self.assertIs(out_proj.last_input[1], scales)
        prepare_input.assert_called_once_with(
            out_proj,
            x,
            z,
            norm.weight,
            norm.eps,
            num_heads=num_heads,
            activation="swish",
        )

    def test_unquantized_scheme_preserves_dense_fallback(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
        dense = torch.empty_like(x)
        norm = Mock(return_value=dense)
        norm.group_size = None
        norm.norm_before_gate = True
        norm.bias = None
        norm.weight = torch.empty(head_dim, dtype=torch.bfloat16)
        norm.eps = 1e-6
        norm.activation = "swish"
        out_proj = _OutProj(scheme=None)
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        actual = Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        norm.assert_called_once_with(x, z)
        self.assertEqual(actual.shape, (num_tokens, num_heads * head_dim))
        self.assertIs(actual, out_proj.last_input)

    def test_declined_fusion_preserves_dense_fallback(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
        dense = torch.empty_like(x)
        prepare_input = Mock(return_value=None)
        out_proj = _OutProj(
            SimpleNamespace(prepare_fused_rmsnorm_gated_input=prepare_input)
        )
        norm = Mock(return_value=dense)
        norm.group_size = None
        norm.norm_before_gate = True
        norm.bias = None
        norm.weight = torch.empty(head_dim, dtype=torch.bfloat16)
        norm.eps = 1e-6
        norm.activation = "swish"
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        actual = Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        prepare_input.assert_called_once()
        norm.assert_called_once_with(x, z)
        self.assertIs(actual, out_proj.last_input)

    def test_incompatible_norm_policy_does_not_call_quant_scheme(self):
        num_tokens, num_heads, head_dim = 2, 3, 128
        x = torch.empty((num_tokens * num_heads, head_dim), dtype=torch.bfloat16)
        z = torch.empty((num_tokens, num_heads, head_dim), dtype=torch.bfloat16)
        dense = torch.empty_like(x)
        prepare_input = Mock()
        out_proj = _OutProj(
            SimpleNamespace(prepare_fused_rmsnorm_gated_input=prepare_input)
        )
        norm = Mock(return_value=dense)
        norm.group_size = None
        norm.norm_before_gate = False
        norm.bias = None
        norm.weight = torch.empty(head_dim, dtype=torch.bfloat16)
        norm.eps = 1e-6
        norm.activation = "swish"
        owner = SimpleNamespace(
            norm=norm,
            out_proj=out_proj,
            num_v_heads=num_heads,
            attn_tp_size=1,
        )

        Qwen3_5GatedDeltaNet._norm_and_out_proj(owner, x, z, z.shape)

        prepare_input.assert_not_called()
        norm.assert_called_once_with(x, z)


class TestQwen35GDNOutProjInputEpilogue(unittest.TestCase):
    """With the pipelined GDN chunk scan, the out_proj input is written by the
    linear attention's epilogue on finished rows; otherwise the norm runs after
    attention as before."""

    NUM_TOKENS, NUM_HEADS, HEAD_DIM = 4, 3, 128

    def _owner(self, scheme):
        return SimpleNamespace(
            norm=_make_norm(self.HEAD_DIM),
            out_proj=_OutProj(scheme),
            num_v_heads=self.NUM_HEADS,
            attn_tp_size=1,
            _out_proj_input_rows_fn=None,
        )

    def _buffers(self, owner, *, pipeline=True, epilogue_backend=True):
        z = torch.empty(
            (self.NUM_TOKENS, self.NUM_HEADS, self.HEAD_DIM), dtype=torch.bfloat16
        )
        backend = SimpleNamespace(
            linear_attn_backend=SimpleNamespace(
                supports_extend_epilogue=epilogue_backend
            )
        )
        with (
            qwen3_5.envs.SGLANG_GDN_CHUNK_PIPELINE.override(pipeline),
            patch(
                "sglang.srt.model_executor.forward_context.get_attn_backend",
                return_value=backend,
            ),
        ):
            return z, Qwen3_5GatedDeltaNet._out_proj_input_buffers(owner, z)

    def test_buffers_and_rows_function_come_from_the_scheme(self):
        prepared_input = (
            torch.empty(1, dtype=torch.uint8),
            torch.empty(1, dtype=torch.uint8),
        )
        rows_fn = Mock()
        prepare_rows = Mock(return_value=(prepared_input, rows_fn))
        owner = self._owner(
            SimpleNamespace(prepare_fused_rmsnorm_gated_rows=prepare_rows)
        )
        z, actual = self._buffers(owner)
        self.assertIs(actual, prepared_input)
        self.assertIs(owner._out_proj_input_rows_fn, rows_fn)
        prepare_rows.assert_called_once_with(
            owner.out_proj,
            self.NUM_TOKENS,
            z,
            owner.norm.weight,
            num_heads=self.NUM_HEADS,
            activation="swish",
        )

        o = torch.zeros(
            (1, self.NUM_TOKENS, self.NUM_HEADS, self.HEAD_DIM), dtype=torch.bfloat16
        )
        x_q, x_scales = prepared_input
        Qwen3_5GatedDeltaNet._write_out_proj_input_rows(
            owner, z, x_q, x_scales, o, 1, 3
        )
        args = rows_fn.call_args.args
        self.assertEqual(
            args[0].shape, (self.NUM_TOKENS * self.NUM_HEADS, self.HEAD_DIM)
        )
        self.assertEqual(args[0].data_ptr(), o.data_ptr())
        self.assertEqual(
            args[1:], (z, owner.norm.weight, owner.norm.eps, x_q, x_scales, 1, 3)
        )

    def test_norm_after_attention_without_pipeline_backend_or_scheme(self):
        cases = {
            "pipeline off": dict(pipeline=False),
            "backend without epilogue": dict(epilogue_backend=False),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                prepare_rows = Mock()
                owner = self._owner(
                    SimpleNamespace(prepare_fused_rmsnorm_gated_rows=prepare_rows)
                )
                self.assertIsNone(self._buffers(owner, **kwargs)[1])
                prepare_rows.assert_not_called()
        with self.subTest("scheme without rows"):
            self.assertIsNone(self._buffers(self._owner(SimpleNamespace()))[1])
        with self.subTest("grouped norm"):
            prepare_rows = Mock()
            owner = self._owner(
                SimpleNamespace(prepare_fused_rmsnorm_gated_rows=prepare_rows)
            )
            owner.norm.group_size = 64
            self.assertIsNone(self._buffers(owner)[1])
            prepare_rows.assert_not_called()
        with self.subTest("scheme declines"):
            owner = self._owner(
                SimpleNamespace(
                    prepare_fused_rmsnorm_gated_rows=Mock(return_value=None)
                )
            )
            self.assertIsNone(self._buffers(owner)[1])


class TestQwen38PrefillProjectionViews(CustomTestCase):
    """ROCm prefill reads the in_proj outputs as strided views only where the
    fused Qwen3.8 GDN prefill prologue consumes them; every other case keeps
    the contiguous unpack."""

    NUM_ROWS = 8

    def _decide(
        self,
        *,
        prologue: bool = True,
        deterministic: bool = False,
        page_major: bool = False,
        layer: dict | None = None,
        batch: dict | None = None,
    ) -> bool:
        layer_fields = dict(
            attn_tp_size=1,
            num_k_heads=QWEN38_GDN_NUM_QK_HEADS,
            num_v_heads=QWEN38_GDN_NUM_V_HEADS,
            head_k_dim=QWEN38_GDN_HEAD_DIM,
            head_v_dim=QWEN38_GDN_HEAD_DIM,
        )
        layer_fields.update(layer or {})
        batch_fields = dict(
            forward_mode=ForwardMode.EXTEND,
            multi_item_delimiter_indices=None,
            mamba_track_mask=None,
            global_num_token_non_padded_cpu=self.NUM_ROWS,
        )
        batch_fields.update(batch or {})
        exec_config = SimpleNamespace(
            deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
        )
        memory_config = SimpleNamespace(enable_page_major_kv_layout=page_major)
        with (
            patch.object(qwen3_5, "_enable_qwen38_gdn_prefill_prologue", prologue),
            patch.object(qwen3_5, "get_exec", return_value=exec_config),
            patch.object(qwen3_5, "get_memory", return_value=memory_config),
        ):
            return Qwen3_5GatedDeltaNet._uses_qwen38_prefill_views(
                SimpleNamespace(**layer_fields),
                SimpleNamespace(**batch_fields),
                self.NUM_ROWS,
            )

    def test_tp1_extend_with_prologue_reads_views(self):
        self.assertTrue(self._decide())
        # No DP metadata (None) is not DP padding.
        self.assertTrue(self._decide(batch=dict(global_num_token_non_padded_cpu=None)))

    def test_cases_outside_the_prologue_keep_the_contiguous_unpack(self):
        half_dim = QWEN38_GDN_HEAD_DIM // 2
        cases = {
            "prologue off": dict(prologue=False),
            "deterministic inference": dict(deterministic=True),
            "page-major KV layout": dict(page_major=True),
            "TP2 head shard": dict(layer=dict(attn_tp_size=2)),
            "other k-head count": dict(layer=dict(num_k_heads=8)),
            "other v-head count": dict(layer=dict(num_v_heads=32)),
            "other k-head dim": dict(layer=dict(head_k_dim=half_dim)),
            "other v-head dim": dict(layer=dict(head_v_dim=half_dim)),
            "decode": dict(batch=dict(forward_mode=ForwardMode.DECODE)),
            "target verify": dict(batch=dict(forward_mode=ForwardMode.TARGET_VERIFY)),
            "multi-item scoring": dict(
                batch=dict(multi_item_delimiter_indices=[torch.zeros(1)])
            ),
            "mamba state tracking": dict(
                batch=dict(mamba_track_mask=torch.ones(1, dtype=torch.bool))
            ),
            "DP padding": dict(
                batch=dict(global_num_token_non_padded_cpu=self.NUM_ROWS - 1)
            ),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertFalse(self._decide(**kwargs))


if __name__ == "__main__":
    unittest.main()
