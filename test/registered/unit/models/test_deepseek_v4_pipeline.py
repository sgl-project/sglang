"""Hyper-connection stage construction and completion across pipeline ranks."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models import deepseek_v4_boundary as boundary
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.models.deepseek_common import moe_transport
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestVPPConstruction(CustomTestCase):
    """Ownership, loader names and endpoint modules must agree across stages."""

    @contextmanager
    def construction(self, rank, size):
        from sglang.srt.layers.layer_boundary import factories
        from sglang.srt.layers.utils import layer_list
        from sglang.srt.models import deepseek_v4

        parallel = SimpleNamespace(
            pp_rank=rank,
            pp_size=size,
            tp_size=1,
            enable_dp_lm_head=False,
            pp_group=SimpleNamespace(
                is_first_rank=rank == 0,
                is_last_rank=rank == size - 1,
                world_size=size,
            ),
        )
        config = SimpleNamespace(
            num_hidden_layers=12,
            hidden_size=4,
            vocab_size=8,
            rms_norm_eps=1e-6,
            hc_eps=1e-6,
            hc_mult=2,
            hc_pre_from_prev_sublayer=True,
            model_type="deepseek_v4",
            tie_word_embeddings=False,
            q_lora_rank=2,
        )

        class Decoder(torch.nn.Module):
            def __init__(self, *, layer_id, prefix, **options):
                super().__init__()
                self.layer_id = layer_id
                self.prefix = prefix
                self.construction_stack = factories._require_stack()
                self.weight = torch.nn.Parameter(torch.full((1,), float(layer_id)))

        with ExitStack() as stack:
            for name, value in (
                ("_is_cuda", False),
                ("_is_hip", False),
                ("_is_npu", False),
                ("_is_xpu", False),
                ("DeepseekV4DecoderLayer", Decoder),
            ):
                stack.enter_context(patch.object(deepseek_v4, name, value))
            for name, value in (
                ("get_parallel", parallel),
                (
                    "get_exec",
                    SimpleNamespace(
                        features=SimpleNamespace(
                            enable_decoder_swa_bounded_replay=False
                        )
                    ),
                ),
                ("is_dp_attention_enabled", False),
                ("is_cross_layer_mhc_fusion_enabled", False),
                ("_is_fused_mhc_post_pre_enabled_xpu", False),
                ("wo_a_fp8_gemm_enabled", False),
                ("use_npu_arch35_mxfp8_wo_a", False),
                ("get_num_fused_shared_experts", 0),
                ("get_attn_tp_context", SimpleNamespace(init_context=Mock())),
            ):
                stack.enter_context(patch.object(deepseek_v4, name, return_value=value))
            stack.enter_context(
                patch.object(layer_list, "get_parallel", return_value=parallel)
            )
            stack.enter_context(
                patch("torch.get_device_module", return_value=SimpleNamespace())
            )
            stack.enter_context(
                patch.object(deepseek_v4.EngramLayout, "from_config", return_value=None)
            )
            stack.enter_context(
                patch.object(
                    deepseek_v4.DecoderReplayTail, "maybe_create", return_value=None
                )
            )
            stack.enter_context(
                patch(
                    "sglang.srt.layers.offloader.get_offloader",
                    return_value=SimpleNamespace(
                        wrap_modules=lambda modules, **kw: list(modules)
                    ),
                )
            )
            for name in (
                "VocabParallelEmbedding",
                "ParallelLMHead",
                "RMSNorm",
                "LogitsProcessor",
            ):
                stack.enter_context(
                    patch.object(
                        deepseek_v4,
                        name,
                        side_effect=lambda *a, **kw: torch.nn.Identity(),
                    )
                )
            yield deepseek_v4, config

    def test_interleaved_ownership_preserves_global_weight_names_and_endpoints(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout
        from sglang.srt.layers.utils import PPMissingLayer

        # Unequal stages expose both gaps and accidental local renumbering.
        layout = VPPLayout(12, 2, partition=(1, 2, 4, 5))
        expected = ((0, 3, 4, 5, 6), (1, 2, 7, 8, 9, 10, 11))
        all_names = set()
        for rank in range(2):
            with (
                self.subTest(rank=rank),
                self.construction(rank, 2) as (module, config),
            ):
                model = module.DeepseekV4ForCausalLM(
                    config, prefix="root", vpp_layout=layout
                )
                self.assertEqual(model.layer_ids, expected[rank])
                self.assertIs(model.vpp_layout, layout)
                self.assertEqual(
                    tuple(s.stage_id for s in model.vpp_stages), (rank, rank + 2)
                )
                stacks = []
                for stage in model.vpp_stages:
                    layers = [model.model.layers[i] for i in stage.layer_ids]
                    stacks.append(layers[0].construction_stack)
                    for layer in layers:
                        self.assertIs(layer.construction_stack, stacks[-1])
                self.assertIsNot(stacks[0], stacks[1])
                owned = tuple(
                    i
                    for i, layer in enumerate(model.model.layers)
                    if not isinstance(layer, PPMissingLayer)
                )
                self.assertEqual(owned, expected[rank])
                names = set(model.model.layers.state_dict())
                self.assertEqual(names, {f"{i}.weight" for i in expected[rank]})
                self.assertFalse(all_names & names)
                all_names.update(names)
                for i in owned:
                    self.assertEqual(
                        model.model.layers[i].prefix, f"root.model.layers.{i}"
                    )
                self.assertEqual(
                    isinstance(model.model.embed_tokens, PPMissingLayer), rank != 0
                )
                self.assertEqual(
                    isinstance(model.model.norm, PPMissingLayer), rank != 1
                )
                self.assertEqual(isinstance(model.lm_head, PPMissingLayer), rank != 1)
                with self.assertRaisesRegex(NotImplementedError, "logical-stage"):
                    model.model._forward_impl(None, None, None, None)
        self.assertEqual(all_names, {f"{i}.weight" for i in range(12)})

    def test_absent_layout_keeps_contiguous_pp_ownership(self):
        from sglang.srt.layers.utils import PPMissingLayer

        for size in (1, 2, 3):
            for rank in range(size):
                with (
                    self.subTest(size=size, rank=rank),
                    self.construction(rank, size) as (module, config),
                ):
                    model = module.DeepseekV4ForCausalLM(config)
                    expected = tuple(
                        range(rank * (12 // size), (rank + 1) * (12 // size))
                    )
                    self.assertEqual(model.layer_ids, expected)
                    self.assertIsNone(model.vpp_layout)
                    self.assertEqual(model.vpp_stages, ())
                    self.assertEqual(
                        tuple(
                            i
                            for i, layer in enumerate(model.model.layers)
                            if not isinstance(layer, PPMissingLayer)
                        ),
                        expected,
                    )

    def test_incompatible_layout_is_rejected_before_layer_allocation(self):
        from sglang.srt.distributed.vpp_layout import VPPLayout

        with (
            self.construction(0, 2) as (module, config),
            patch.object(module, "DeepseekV4DecoderLayer") as decoder,
        ):
            for layout, message in (
                (VPPLayout(8, 2), "layer count"),
                (VPPLayout(12, 3), "physical size"),
            ):
                with (
                    self.subTest(layout=layout),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    module.DeepseekV4ForCausalLM(config, vpp_layout=layout)
            decoder.assert_not_called()


class TestPipelineCompletion(unittest.TestCase):
    def test_real_stages_write_mhc_once_before_pipeline_handoff(self):
        from sglang.srt.layers.layer_boundary import append_stages, layer_stack
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
        from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
        from sglang.srt.model_executor.forward_mode import ForwardMode
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
