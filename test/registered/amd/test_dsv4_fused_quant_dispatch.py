"""Q-LoRA producer selection with the optional AITER path disabled."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-a-test-1-gpu-small-amd")


class TestFusedQuantDispatch(unittest.TestCase):
    def test_prepare_producers(self):
        from sglang.kernels.ops.attention import fused_qk_norm_rope_store as store
        from sglang.srt.layers.attention.dsv4 import unified_kv_gate
        from sglang.srt.models import deepseek_v4 as model

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]])
        norm = torch.nn.RMSNorm(4, eps=1e-6)
        norm.variance_epsilon = 1e-6
        expected = norm(x)
        quantized = (expected, torch.ones(2, 1))

        for gfx1250 in (False, True):
            for multi_stream in (False, True):
                for aiter, e8m0 in ((False, False), (True, False), (False, True)):
                    with (
                        self.subTest(
                            gfx1250=gfx1250,
                            multi_stream=multi_stream,
                            aiter=aiter,
                            e8m0=e8m0,
                        ),
                        ExitStack() as stack,
                    ):
                        fused = stack.enter_context(
                            mock.patch.object(
                                model,
                                "_fused_rmsnorm_fp8_quant",
                                return_value=(quantized, expected),
                            )
                        )
                        if not (aiter or e8m0):
                            fused.side_effect = AssertionError(
                                "AITER producer disabled"
                            )
                        for name, value in (
                            ("_use_aiter", aiter),
                            ("_is_gfx95_supported", not gfx1250),
                            ("_is_gfx1250_supported", gfx1250),
                            ("_is_npu", False),
                        ):
                            stack.enter_context(mock.patch.object(model, name, value))
                        stack.enter_context(
                            mock.patch.object(model, "is_cp_active", return_value=False)
                        )
                        stack.enter_context(
                            mock.patch.object(
                                model,
                                "get_token_to_kv_pool",
                                return_value=mock.Mock(swa_page_size=1),
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                unified_kv_gate,
                                "is_unified_kv_triton",
                                return_value=False,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                unified_kv_gate, "is_unified_kv_fp8", return_value=False
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                model.envs.SGLANG_OPT_FUSED_QK_NORM_ROPE_VERIFY,
                                "get",
                                return_value=False,
                            )
                        )
                        stack.enter_context(
                            mock.patch.object(
                                torch.cuda, "current_stream", return_value=mock.Mock()
                            )
                        )
                        fused_store = stack.enter_context(
                            mock.patch.object(
                                store,
                                "fused_qk_norm_rope_swa_store",
                                side_effect=lambda **kw: kw["q"],
                            )
                        )
                        projection = mock.Mock(
                            side_effect=lambda value: (
                                value[0] if isinstance(value, tuple) else value,
                                None,
                            )
                        )
                        layer = SimpleNamespace(
                            alt_streams=[mock.Mock()],
                            compressor=None,
                            indexer=mock.Mock(),
                            fuse_wqa_wkv=False,
                            wq_a=lambda value: (value, None),
                            wq_b=projection,
                            wkv=lambda value: (value, None),
                            q_norm=norm,
                            kv_norm=norm,
                            eps=1e-6,
                            qk_rope_head_dim=0,
                            cos_cache=None,
                            sin_cache=None,
                            layer_id=0,
                            compress_ratio=4,
                            use_fused_qk_norm_rope=True,
                            _q_lora_quant_e8m0=mock.Mock(return_value=e8m0),
                            _normalize_q_lora=lambda value: (norm(value), norm(value)),
                        )
                        batch = mock.Mock()
                        batch.forward_mode.is_decode_or_idle.return_value = True
                        batch.forward_mode.is_target_verify.return_value = False
                        method = (
                            model.MQALayer._forward_prepare_multi_stream_hip
                            if multi_stream
                            else model.MQALayer._forward_prepare
                        )
                        result = method(layer, x, torch.arange(2), batch, mock.Mock())
                        q = result if multi_stream else result[0]
                        torch.testing.assert_close(q, expected)
                        actual = projection.call_args.args[0]
                        if aiter or e8m0:
                            self.assertIs(actual, quantized)
                            fused.assert_called_once_with(
                                x, norm.weight, 1e-6, e8m0=e8m0
                            )
                        else:
                            torch.testing.assert_close(actual, expected)
                            fused.assert_not_called()
                        layer._q_lora_quant_e8m0.assert_called_once_with(
                            with_indexer=not multi_stream
                        )
                        torch.testing.assert_close(
                            layer.indexer.call_args.kwargs["q_lora"], expected
                        )
                        if not multi_stream:
                            self.assertIs(
                                layer.indexer.call_args.kwargs["q_lora_quant"],
                                quantized if aiter or e8m0 else None,
                            )
                        fused_store.assert_called_once()


if __name__ == "__main__":
    unittest.main()
