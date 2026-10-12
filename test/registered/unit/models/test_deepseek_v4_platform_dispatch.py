"""Unmigrated platforms keep decoder, draft and operation contracts."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models import deepseek_v4
from sglang.srt.models import deepseek_v4_dspark as dspark
from sglang.srt.models import deepseek_v4_nextn as nextn
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPlatformDispatch(unittest.TestCase):
    def test_unmigrated_decoders_keep_complete_producers(self):
        for hip in (False, True):
            for pre_from_prev in (False, True):
                for is_nextn in (False, True):
                    with self.subTest(
                        hip=hip, pre_from_prev=pre_from_prev, nextn=is_nextn
                    ):
                        config = SimpleNamespace(
                            hidden_size=8,
                            hc_pre_from_prev_sublayer=pre_from_prev,
                            model_type="deepseek_v41",
                            vision_n_layers=0,
                            hc_mult=4,
                            hc_sinkhorn_iters=20,
                            hc_eps=1e-6,
                            rms_norm_eps=1e-6,
                        )
                        attention = SimpleNamespace(
                            wo_b=SimpleNamespace(reduce_results=True)
                        )
                        moe = SimpleNamespace(reduce_results=True)
                        hip_entry = SimpleNamespace(
                            init_decoder_layer=lambda layer, quant: setattr(
                                layer, "hc_boundary_fused", pre_from_prev
                            )
                        )
                        with (
                            patch.object(deepseek_v4, "_is_cuda", False),
                            patch.object(deepseek_v4, "_is_hip", hip),
                            patch.object(deepseek_v4, "_hip", hip_entry, create=True),
                            patch.object(
                                deepseek_v4,
                                "is_dsa_enable_prefill_cp",
                                return_value=False,
                            ),
                            patch.object(
                                deepseek_v4.DeepseekV4DecoderLayer,
                                "_build_self_attn",
                                return_value=attention,
                            ),
                            patch.object(
                                deepseek_v4.deepseek_v2,
                                "DeepseekV2MoE",
                                return_value=moe,
                            ),
                            patch.object(
                                deepseek_v4,
                                "append_stages",
                                side_effect=AssertionError("unmigrated platform"),
                            ),
                        ):
                            layer = deepseek_v4.DeepseekV4DecoderLayer(
                                config, 0, is_nextn=is_nextn
                            )
                        self.assertFalse(layer._uses_stage_boundaries)
                        self.assertTrue(attention.wo_b.reduce_results)
                        self.assertTrue(moe.reduce_results)
                        self.assertFalse(hasattr(layer, "attn_boundary"))

    def test_original_operation_state_contract_is_selected(self):
        layer = deepseek_v4.DeepseekV4DecoderLayer.__new__(
            deepseek_v4.DeepseekV4DecoderLayer
        )
        torch.nn.Module.__init__(layer)
        layer._uses_stage_boundaries = False
        state = object()
        for name in (
            "op_mhc_post_attn_pre_mlp",
            "op_mhc_postprocess",
            "op_gather_a",
            "op_gather_b",
            "op_moe",
            "op_combine_a",
            "op_combine_b",
        ):
            with self.subTest(operation=name):
                worker = Mock(return_value=object())
                setattr(layer, "_compat_" + name, worker)
                result = getattr(layer, name)(state)
                worker.assert_called_once_with(state)
                self.assertIs(result, worker.return_value)

    def test_tbo_exception_releases_both_child_transports(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        model = deepseek_v4.DeepseekV4Model.__new__(deepseek_v4.DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.layers = [object()]
        model.start_layer, model.end_layer = 0, 1
        children = [SimpleNamespace(residual_stream=None) for _ in range(2)]
        transports = [SimpleNamespace(release=Mock()) for _ in children]

        def fail(**kwargs):
            for child, transport in zip(children, transports):
                residual_batch.start(child)
                child.residual_stream.transport = (object(), transport)
            raise RuntimeError("overlap compute failed")

        strategy = SimpleNamespace(operations=[], tbo_delta_stages=0)
        batch = SimpleNamespace(tbo_children=children, global_forward_mode=object())
        with (
            patch.object(deepseek_v4, "_is_cuda", True),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch.object(
                deepseek_v4,
                "get_moe_a2a_backend",
                return_value=SimpleNamespace(is_none=lambda: True),
            ),
            patch(
                "sglang.srt.batch_overlap.operations_strategy.OperationsStrategy.init_new_tbo",
                return_value=strategy,
            ),
            patch(
                "sglang.srt.batch_overlap.two_batch_overlap._model_forward_filter_inputs",
                return_value={},
            ),
            patch(
                "sglang.srt.batch_overlap.operations.execute_overlapped_operations",
                side_effect=fail,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "overlap compute failed"):
                model._forward_layers_tbo(torch.arange(2), torch.ones(2, 4), batch)
        for child, transport in zip(children, transports):
            self.assertIsNone(child.residual_stream)
            transport.release.assert_called_once()

    def test_gfx1250_attention_keeps_plain_norm_gate(self):
        class StopAttention(Exception):
            pass

        layer = deepseek_v4.DeepseekV4DecoderLayer.__new__(
            deepseek_v4.DeepseekV4DecoderLayer
        )
        torch.nn.Module.__init__(layer)
        layer._uses_stage_boundaries = False
        layer.use_fused_mhc_post_pre = False
        for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "input_layernorm"):
            setattr(layer, name, object())
        value = object()
        layer.hc_pre = Mock(return_value=(value, object(), object(), False))
        layer._input_norm = Mock(return_value=(object(), None))
        layer.self_attn = SimpleNamespace(
            maybe_use_decode_attn_tp=Mock(side_effect=StopAttention)
        )
        with patch.object(deepseek_v4, "_is_gfx95_supported", False):
            with self.assertRaises(StopAttention):
                layer(object(), object(), object(), object(), object())
        layer._input_norm.assert_called_once_with(value, allow_aiter_quant=False)

    def test_main_and_dspark_use_original_forward_on_unmigrated_platforms(self):
        for module, cls in (
            (deepseek_v4, deepseek_v4.DeepseekV4Model),
            (dspark, dspark.DeepseekV4ForCausalLMDSpark),
        ):
            with self.subTest(model=cls.__name__):
                model = cls.__new__(cls)
                torch.nn.Module.__init__(model)
                model._forward_compat = Mock(return_value=object())
                with patch.object(module, "_is_cuda", False):
                    result = model(object(), object(), object(), input_embeds=None)
                self.assertIs(result, model._forward_compat.return_value)
                model._forward_compat.assert_called_once()

    def test_nextn_completes_original_tensor_tail(self):
        model = nextn.DeepseekV4ModelNextN.__new__(nextn.DeepseekV4ModelNextN)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=2)
        model.hc_mult = 2
        model.enorm = model.hnorm = torch.nn.Identity()
        model.e_proj = model.h_proj = lambda x: (x, None)
        streams = torch.arange(4.0).reshape(1, 2, 2)
        model.decoder = Mock(return_value=(streams, streams + 1, object(), object()))
        completed = streams + 5
        model.decoder.hc_post.return_value = completed
        model.hc_head = lambda value, *args: value.mean(1)
        model.hc_head_fn = model.hc_head_scale = model.hc_head_base = None
        model.shared_head = SimpleNamespace(norm=torch.nn.Identity())
        ids = torch.tensor([1])
        batch = SimpleNamespace(spec_info=SimpleNamespace(hidden_states=streams))
        with (
            patch.object(nextn, "_is_cuda", False),
            patch.object(nextn, "_is_npu", False),
            patch.object(
                nextn, "get_parallel", return_value=SimpleNamespace(attn_dp_size=1)
            ),
            patch.object(nextn, "dsa_use_prefill_cp", return_value=False),
        ):
            result, pre_head = model(ids, ids, batch, input_embeds=torch.ones(1, 2))
        model.decoder.hc_post.assert_called_once()
        torch.testing.assert_close(pre_head, completed.flatten(1))
        torch.testing.assert_close(result, completed.mean(1))

    def test_legacy_dspark_base_keeps_plain_input_norm(self):
        stage = dspark.DSparkV4Stage.__new__(dspark.DSparkV4Stage)
        torch.nn.Module.__init__(stage)
        stage._uses_stage_boundaries = False
        stage.hc_pre_from_prev_sublayer = False
        for name in (
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_attn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "hc_ffn_base",
        ):
            setattr(stage, name, object())
        stage.input_layernorm = stage.post_attention_layernorm = torch.nn.Identity()
        stage.hc_pre = lambda value, *args: (value + 0.5, object(), object(), False)
        stage.hc_post = lambda value, residual, *args: value + residual
        stage.self_attn = Mock(side_effect=lambda positions, value, batch: value * 2)
        stage.self_attn.maybe_use_decode_attn_tp.return_value = nullcontext()
        stage._run_ffn = lambda value, batch: value + 7
        stage._input_norm = Mock(side_effect=AssertionError("changed norm route"))
        value = torch.ones(1, 2, 2)
        result, pre = stage(torch.tensor([0]), value, object())
        first = (value + 0.5) * 2 + value
        torch.testing.assert_close(result, (first + 0.5 + 7) + first)
        self.assertIsNone(pre)


if __name__ == "__main__":
    unittest.main()
