"""Non-CUDA platforms keep decoder, draft and operation contracts."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models import deepseek_v4
from sglang.srt.models import deepseek_v4_dspark as dspark
from sglang.srt.models import deepseek_v4_nextn as nextn
from sglang.srt.models import deepseek_v4_platform_stage as platform_stage
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPlatformDispatch(unittest.TestCase):
    def test_cuda_main_loop_materializes_and_clears_output(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        model = deepseek_v4.DeepseekV4Model.__new__(deepseek_v4.DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.hc_mult = 2
        model.hc_pre_from_prev_sublayer = False
        model.dspark_layers_to_capture = None
        model.start_layer, model.end_layer = 0, 1
        model._can_run_tbo = lambda batch: False
        model.hc_head = lambda value, *args: value.mean(1)
        model.hc_head_fn = model.hc_head_scale = model.hc_head_base = None
        model.norm = torch.nn.Identity()

        def layer(**kwargs):
            return residual_batch.set_written(
                kwargs["hidden_states"] + 3, kwargs["forward_batch"]
            )

        model.layers = [layer]
        ids = torch.tensor([1])
        batch = SimpleNamespace(input_ids=ids, attn_cp_metadata=None)
        with (
            patch.object(deepseek_v4, "_is_cuda", True),
            patch.object(deepseek_v4, "_is_npu", False),
            patch.object(
                deepseek_v4,
                "get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch.object(deepseek_v4, "get_attn_backend", return_value=object()),
            patch.object(deepseek_v4, "check_cuda_graph_backend", return_value=False),
            patch.object(
                deepseek_v4,
                "get_global_expert_distribution_recorder",
                return_value=SimpleNamespace(
                    with_current_layer=lambda i: nullcontext()
                ),
            ),
            patch(
                "sglang.kernels.ops.layernorm.mhc.hc_broadcast",
                side_effect=lambda value, n: value[:, None].repeat(1, n, 1),
            ),
        ):
            result = model(ids, ids, batch, torch.ones(1, 2))
        torch.testing.assert_close(result[0], torch.full((1, 2), 4.0))
        self.assertIsNone(batch.residual_stream)

    def test_nextn_exception_releases_invocation_transport(self):
        from sglang.srt.layers.layer_boundary.residual import batch as residual_batch

        model = nextn.DeepseekV4ModelNextN.__new__(nextn.DeepseekV4ModelNextN)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(hidden_size=2)
        model.hc_mult = 2
        model.enorm = model.hnorm = torch.nn.Identity()
        model.e_proj = model.h_proj = lambda x: (x, None)
        transport = SimpleNamespace(release=Mock())

        def fail(**kwargs):
            batch = kwargs["forward_batch"]
            residual_batch.stream_of(batch).transport = (object(), transport)
            raise RuntimeError("decoder failed")

        model.decoder = fail
        model.hc_head = lambda value, *args: value.mean(1)
        model.hc_head_fn = model.hc_head_scale = model.hc_head_base = None
        model.shared_head = SimpleNamespace(norm=torch.nn.Identity())
        ids = torch.tensor([1])
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(hidden_states=torch.ones(1, 2, 2))
        )
        with (
            patch.object(nextn, "_is_cuda", True),
            patch.object(nextn, "_is_npu", False),
            patch.object(
                nextn, "get_parallel", return_value=SimpleNamespace(attn_dp_size=1)
            ),
            patch.object(nextn, "dsa_use_prefill_cp", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "decoder failed"):
                model(ids, ids, batch, input_embeds=torch.ones(1, 2))
            self.assertIsNone(batch.residual_stream)
            transport.release.assert_called_once()
            model.decoder = lambda **kwargs: residual_batch.set_written(
                kwargs["hidden_states"], kwargs["forward_batch"]
            )
            result, _ = model(ids, ids, batch, input_embeds=torch.ones(1, 2))
            torch.testing.assert_close(result, torch.full((1, 2), 2.0))
        self.assertIsNone(batch.residual_stream)

    def test_non_cuda_nextn_completes_tensor_tail(self):
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

    def test_non_cuda_dspark_uses_plain_input_norm(self):
        stage = dspark.DSparkV4PlatformStage.__new__(dspark.DSparkV4PlatformStage)
        torch.nn.Module.__init__(stage)
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


class TestStageLifecycle(unittest.TestCase):
    def make_layer(self, cuda, *, lagged=True):
        cls = (
            deepseek_v4.DeepseekV4CudaDecoderLayer
            if cuda
            else deepseek_v4.DeepseekV4PlatformDecoderLayer
        )
        config = SimpleNamespace(
            hidden_size=8,
            hc_pre_from_prev_sublayer=lagged,
            model_type="deepseek_v41",
            vision_n_layers=0,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            rms_norm_eps=1e-6,
        )
        attention = SimpleNamespace(
            wo_b=SimpleNamespace(reduce_results=True),
            attn_tp_size=4,
        )
        moe = SimpleNamespace(reduce_results=True)
        with (
            patch.object(deepseek_v4, "_is_cuda", cuda),
            patch.object(deepseek_v4, "_is_hip", False),
            patch.object(deepseek_v4, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(
                cls,
                "_build_self_attn",
                return_value=attention,
            ),
            patch.object(deepseek_v4.deepseek_v2, "DeepseekV2MoE", return_value=moe),
            patch.object(
                deepseek_v4.hc_boundary, "stages", return_value=(object(), object())
            ),
            patch.object(
                deepseek_v4,
                "append_stages",
                return_value=(object(), object()),
            ),
        ):
            return cls(config, 0)

    def test_weight_refresh_invalidates_only_owned_boundary_caches(self):
        for cuda in (False, True):
            with self.subTest(cuda=cuda):
                layer = self.make_layer(cuda)
                old_hc = layer.attn_hc
                layer.__dict__["_can_fuse_attn_mhc"] = True
                layer.__dict__["_can_fuse_ffn_mhc"] = True
                if not cuda:
                    layer.local_boundary = layer.next_boundary = object()
                with patch.object(
                    deepseek_v4,
                    "get_platform",
                    return_value=SimpleNamespace(is_sm100=False, is_sm90=False),
                ):
                    layer.refresh_mhc_norm_weight_cache()
                self.assertNotIn("_can_fuse_attn_mhc", layer.__dict__)
                self.assertNotIn("_can_fuse_ffn_mhc", layer.__dict__)
                self.assertIsNot(layer.attn_hc, old_hc)
                self.assertEqual(
                    layer._input_layernorm_weight_bf16.dtype, torch.bfloat16
                )
                if not cuda:
                    self.assertIsNone(layer.local_boundary)
                    self.assertIsNone(layer.next_boundary)

    def test_non_cuda_partial_attention_cannot_fuse_mhc(self):
        layer = self.make_layer(False)
        layer.self_attn.wo_b.reduce_results = False
        layer.local_boundary = SimpleNamespace(norm_fusable=True)
        with (
            patch.object(
                platform_stage, "get_parallel", return_value=SimpleNamespace(tp_size=4)
            ),
            patch.object(platform_stage.mhc, "can_fuse_post", return_value=True),
        ):
            self.assertFalse(layer._can_fuse_attn_mhc)


if __name__ == "__main__":
    unittest.main()
