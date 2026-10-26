"""Hyper-connection stage construction and completion across pipeline ranks."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common import moe_transport
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPipelineCompletion(unittest.TestCase):
    def test_real_stages_write_mhc_once_before_pipeline_handoff(self):
        from sglang.srt.layers.layer_boundary import append_stages, layer_stack
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.models import deepseek_v4
        from sglang.srt.models.deepseek_v2 import MoEOutput
        from sglang.test import boundary_fixtures as fixture

        for lagged in (False, True):
            for fused in (False, True):
                with self.subTest(lagged=lagged, fused=fused):
                    config = SimpleNamespace(
                        hc_pre_from_prev_sublayer=lagged,
                        model_type="deepseek_v41" if lagged else "deepseek_v4",
                    )
                    layer = SimpleNamespace(
                        config=config,
                        boundary_pre_from_prev=lagged,
                        boundary_mhc_fusions=fused,
                        use_fused_mhc_post_pre=False,
                        _can_fuse_ffn_mhc=fused,
                        _can_fuse_attn_mhc=False,
                        hc_cfg=SimpleNamespace(hidden=4),
                        mlp=SimpleNamespace(tp_size=2),
                        ffn_hc=object(),
                        attn_hc=object(),
                        self_attn=SimpleNamespace(
                            attn_tp_size=2, accepts_mxfp8_swizzled_input=lambda: False
                        ),
                        engram=None,
                        input_layernorm=fixture.Norm(),
                        post_attention_layernorm=fixture.Norm(),
                        hc_post=Mock(
                            side_effect=lambda value, streams, post, comb: (
                                streams + value[:, None]
                            )
                        ),
                    )
                    layer._bind_attention_transport = lambda *args: (
                        deepseek_v4.DeepseekV4CudaDecoderLayer._bind_attention_transport(
                            layer, *args
                        )
                    )
                    layer._bind_moe_transport = lambda *args: (
                        deepseek_v4.DeepseekV4CudaDecoderLayer._bind_moe_transport(
                            layer, *args
                        )
                    )
                    parallel = fixture.parallel_of(attn_dp=1, attn_tp=2, pp_size=2)
                    source = torch.arange(24.0).reshape(3, 2, 4)
                    batch = SimpleNamespace(
                        input_ids=torch.arange(3),
                        forward_mode=ForwardMode.EXTEND,
                        attn_cp_metadata=None,
                        residual_stream=ResidualStream(source),
                    )
                    call = boundary.LayerCall()
                    sums = Mock(side_effect=lambda value: value * 2)
                    attn_sums = Mock(side_effect=lambda value: value * 2)
                    attn_group = SimpleNamespace(all_reduce=attn_sums)
                    post = Mock(
                        side_effect=lambda hc, value, streams, **kw: (
                            streams + value[:, None]
                        )
                    )
                    with (
                        fixture.planning(parallel),
                        patch(
                            "sglang.srt.models.deepseek_v4_boundary.get_moe_a2a_backend",
                            return_value=SimpleNamespace(is_none=lambda: True),
                        ),
                        patch.object(
                            moe_transport,
                            "get_moe_a2a_backend",
                            return_value=SimpleNamespace(is_none=lambda: True),
                        ),
                        patch.object(boundary, "get_parallel", return_value=parallel),
                        patch.object(
                            moe_transport, "get_parallel", return_value=parallel
                        ),
                        patch("sglang.srt.layers.moe.post_experts_all_reduce", sums),
                        patch.object(mhc, "run_moe_post", post),
                        patch.object(mhc, "run_attn_post", post),
                        patch.object(mhc, "make_boundary", return_value=None),
                        patch.object(
                            boundary,
                            "_sum_group",
                            return_value=attn_group,
                        ),
                        patch(
                            "sglang.srt.layers.layer_boundary.row_transport._sum_group",
                            return_value=attn_group,
                        ),
                        patch(
                            "sglang.srt.models.deepseek_v4_boundary.get_cp_decode_attn_tp_ctx",
                            return_value=SimpleNamespace(use_decode_attn_tp=False),
                        ),
                        patch.object(
                            deepseek_v4,
                            "is_cross_layer_mhc_fusion_enabled",
                            return_value=fused,
                        ),
                        patch.object(
                            deepseek_v4,
                            "_is_fused_mhc_post_pre_enabled_xpu",
                            return_value=False,
                        ),
                        patch(
                            "sglang.srt.models.deepseek_v4_boundary.can_fuse_all_reduce",
                            return_value=True,
                        ),
                        patch.object(
                            boundary.HcReadCall,
                            "read",
                            side_effect=lambda value, norm, quant_format="", **kw: (
                                value.mean(1),
                                boundary.ReadState(value, call=call),
                            ),
                        ),
                    ):
                        with layer_stack(
                            next_layers=[lambda: boundary.stages(config, 1)]
                        ):
                            declarations = boundary.stages(config, 0, layer=layer)
                            attn, ffn = append_stages(
                                *((d, fixture.Norm()) for d in declarations)
                            )
                        operand = attn.prepare(source, batch, read_context=call)
                        attended = attn.finish(
                            mhc.AttnOutput(torch.ones_like(operand)), batch
                        )
                        operand = ffn.prepare(attended, batch, read_context=call)
                        with ffn.exit(batch) as exit:
                            pieces = MoEOutput(
                                torch.ones_like(operand),
                                None,
                                SimpleNamespace(quant_method=None),
                                1.0,
                                False,
                            )
                            output = exit.finish(pieces)
                        self.assertIsNone(batch.residual_stream.pending)
                        self.assertIsNone(batch.residual_stream.transport)
                        wire = residual_batch.to_pp(output, batch)
                        with layer_stack(
                            previous_layers=[lambda: boundary.stages(config, 0)],
                            final_read=boundary.FinalRead(),
                        ):
                            receiver, _ = append_stages(
                                *(
                                    (d, fixture.Norm())
                                    for d in boundary.stages(config, 1, layer=layer)
                                )
                            )
                        received_batch = SimpleNamespace(**vars(batch))
                        received = receiver.from_pp(wire, received_batch)
                        read = receiver.prepare(
                            received, received_batch, read_context=call
                        )
                        torch.testing.assert_close(read, (source + 4).mean(1))
                        # Receiving an already-completed output must not reduce it again.
                        self.assertEqual(attn_sums.call_count, 1)
                        residual_batch.clear(received_batch)
                    torch.testing.assert_close(output, source + 4)
                    torch.testing.assert_close(wire["hidden_states"], source + 4)
                    sums.assert_called_once()
                    self.assertEqual((post if lagged else layer.hc_post).call_count, 2)
                    self.assertIsNone(batch.residual_stream)


if __name__ == "__main__":
    unittest.main()
