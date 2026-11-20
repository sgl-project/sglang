"""Regression tests for Quark block-FP8 weights in the DeepSeek loader."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.model_runner_components import weight_updater
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
    _load_fused_indexer_wk,
    _resolve_layer_weight_block_size,
    _resolve_quark_fp8_block_scale_name,
)
from sglang.test.test_utils import CustomTestCase


class _FakeParam:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, loaded_weight):
        self.loaded = loaded_weight


class _FakeDeepseekLoader(DeepseekV2WeightLoaderMixin, torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.config = SimpleNamespace(
            n_routed_experts=0,
            q_lora_rank=2048,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
        )
        self.model = SimpleNamespace()
        self.quant_config = SimpleNamespace(get_name=lambda: "quark")
        self.num_fused_shared_experts = 0
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self._params = params
        self.post_load_called = False

    def named_parameters(self):
        return iter(self._params.items())

    def _maybe_quant_weights_to_fp8_ue8m0(
        self, weights, attn_quant_modules, nextn_conf
    ):
        return weights

    def post_load_weights(self, is_nextn=False, weight_names=None):
        self.post_load_called = True


def _weight_updater(model):
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=SimpleNamespace(dtype=torch.float32),
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=MagicMock(),
        recapture_cuda_graph=None,
        get_model_runner=lambda: None,
    )


_FUSED_QKV_A_SCALE = (
    "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"
)
_FUSED_INDEXER = "model.layers.0.self_attn.indexer.wk_weights_proj.weight"
_Q_A_SCALE = "model.layers.0.self_attn.q_a_proj.weight_scale"
_KV_A_SCALE = "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale"


class TestQuarkBlockFp8WeightLoading(CustomTestCase):
    def test_scale_name_maps_only_when_scale_inv_is_registered(self):
        scale_inv_name = (
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"
        )
        params_dict = {scale_inv_name: torch.empty(1)}
        checkpoint_name = scale_inv_name.removesuffix("_inv")

        self.assertEqual(
            _resolve_quark_fp8_block_scale_name(checkpoint_name, params_dict),
            scale_inv_name,
        )

        mxfp4_name = "model.layers.0.mlp.experts.w13_weight_scale"
        params_dict = {mxfp4_name: torch.empty(1)}
        self.assertEqual(
            _resolve_quark_fp8_block_scale_name(mxfp4_name, params_dict),
            mxfp4_name,
        )

    def test_layer_quant_method_precedes_config_level_summary(self):
        layer = SimpleNamespace(
            quant_method=SimpleNamespace(weight_block_size=[64, 128])
        )
        quant_config = SimpleNamespace(
            linear_fp8_config=SimpleNamespace(weight_block_size=[128, 128]),
            weight_block_size=None,
        )

        self.assertEqual(
            _resolve_layer_weight_block_size(layer, quant_config),
            [64, 128],
        )

        layer.quant_method.weight_block_size = None
        self.assertIsNone(_resolve_layer_weight_block_size(layer, quant_config))

    def test_fused_qkv_a_quark_scales_load_into_scale_inv(self):
        target_name = (
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"
        )
        target = _FakeParam()
        loader = _FakeDeepseekLoader({target_name: target})
        q_scale = torch.ones((2, 3), dtype=torch.float32)
        kv_scale = torch.full((1, 3), 2.0, dtype=torch.float32)

        loader.do_load_weights(
            [
                (
                    "model.layers.0.self_attn.q_a_proj.weight_scale",
                    q_scale,
                ),
                (
                    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale",
                    kv_scale,
                ),
            ]
        )

        torch.testing.assert_close(
            target.loaded,
            torch.cat([q_scale, kv_scale], dim=0),
        )
        self.assertTrue(loader.post_load_called)

    def test_fused_qkv_a_halves_pair_across_load_weights_calls(self):
        """A chunked online update can send q_a_proj and kv_a_proj_with_mqa in
        different calls, and may reuse its buffer once a call returns."""
        target_name = (
            "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa.weight_scale_inv"
        )
        target = _FakeParam()
        loader = _FakeDeepseekLoader({target_name: target})
        q_scale = torch.ones((2, 3), dtype=torch.float32)
        kv_scale = torch.full((1, 3), 2.0, dtype=torch.float32)

        loader.do_load_weights(
            [("model.layers.0.self_attn.q_a_proj.weight_scale", q_scale)]
        )
        self.assertIsNone(target.loaded)
        q_scale.fill_(9.0)
        loader.do_load_weights(
            [("model.layers.0.self_attn.kv_a_proj_with_mqa.weight_scale", kv_scale)]
        )

        torch.testing.assert_close(
            target.loaded,
            torch.cat([torch.ones((2, 3)), kv_scale], dim=0),
        )
        self.assertEqual(loader.discard_pending_weight_halves(), [])

    def test_begin_weight_update_drops_a_half_left_by_an_earlier_load(self):
        """A half startup left must not pair with the session's other half."""
        target = _FakeParam()
        loader = _FakeDeepseekLoader({_FUSED_QKV_A_SCALE: target})
        loader.do_load_weights([(_Q_A_SCALE, torch.ones((2, 3)))])

        with patch.object(DefaultModelLoader, "restore_weights_before_loading"):
            _weight_updater(loader).begin_weight_update()
        loader.do_load_weights([(_KV_A_SCALE, torch.ones((1, 3)))])

        self.assertIsNone(target.loaded)
        self.assertEqual(loader.discard_pending_weight_halves(), [_KV_A_SCALE])

    def test_rollback_after_a_failed_disk_load_drops_its_halves(self):
        """A failed load's half must not pair with the rollback reload's other half."""
        target = _FakeParam()
        loader = _FakeDeepseekLoader({_FUSED_QKV_A_SCALE: target})
        disk_loads = [
            [(_Q_A_SCALE, torch.ones((2, 3)))],
            [(_KV_A_SCALE, torch.ones((1, 3)))],
        ]

        def load_weights_and_postprocess(model, weights, target_device):
            model.do_load_weights(disk_loads.pop(0))
            if disk_loads:
                raise RuntimeError("truncated checkpoint")

        model_loader = MagicMock(spec=DefaultModelLoader)
        model_loader.load_weights_and_postprocess.side_effect = (
            load_weights_and_postprocess
        )
        module = weight_updater.__name__
        with (
            patch(
                f"{module}.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            patch(f"{module}.LoadConfig"),
            patch(f"{module}.get_pp_stage_load_group"),
            patch(f"{module}.get_available_gpu_memory", return_value=0.0),
            patch(f"{module}.get_model_loader", return_value=model_loader),
            patch.object(DefaultModelLoader.Source, "init_new"),
            patch.object(DefaultModelLoader, "restore_weights_before_loading"),
        ):
            success, _ = _weight_updater(loader).update_weights_from_disk(
                "unused", "auto"
            )

        self.assertFalse(success)
        self.assertIsNone(target.loaded)

    def test_ipc_update_drops_held_halves_before_and_after_it(self):
        """An IPC update loads in several calls with no session around it."""
        target = _FakeParam()
        loader = _FakeDeepseekLoader({_FUSED_QKV_A_SCALE: target})
        loader.do_load_weights([(_Q_A_SCALE, torch.ones((2, 3)))])

        class _Worker:
            def __init__(self, model_runner):
                pass

            def update_weights_from_ipc(self, zmq_handles):
                loader.do_load_weights([(_KV_A_SCALE, torch.ones((1, 3)))])

        updater = _weight_updater(loader)
        worker_module = SimpleNamespace(
            SGLangCheckpointEngineWorkerExtensionImpl=_Worker
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.checkpoint_engine.checkpoint_engine_worker": worker_module
                },
            ),
            patch.object(
                weight_updater,
                "get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
        ):
            success, message = updater.update_weights_from_ipc(
                SimpleNamespace(zmq_handles={})
            )

        self.assertTrue(success, message)
        self.assertIsNone(target.loaded)
        self.assertEqual(loader.discard_pending_weight_halves(), [])

    def test_fp8_indexer_wk_scale_and_weight_pair_across_load_weights_calls(self):
        """The indexer wk pair is held apart from the a_proj pair, and is cloned too."""
        fused_param = torch.nn.Parameter(
            torch.zeros((4, 4), dtype=torch.bfloat16), requires_grad=False
        )
        loader = _FakeDeepseekLoader({_FUSED_INDEXER: fused_param})
        loader.quant_config.weight_block_size = [128, 128]
        wk_scale = torch.ones((1, 1), dtype=torch.float32)
        wk_weight = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
        dequantized = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch(
            "sglang.srt.models.deepseek_common.deepseek_weight_loader."
            "block_quant_dequant",
            return_value=dequantized,
        ) as block_quant_dequant:
            loader.do_load_weights(
                [("model.layers.0.self_attn.indexer.wk.weight_scale", wk_scale)]
            )
            block_quant_dequant.assert_not_called()
            wk_scale.fill_(9.0)
            loader.do_load_weights(
                [("model.layers.0.self_attn.indexer.wk.weight", wk_weight)]
            )

        torch.testing.assert_close(
            block_quant_dequant.call_args.args[1], torch.ones((1, 1))
        )
        torch.testing.assert_close(fused_param[:2], dequantized)
        self.assertEqual(loader.discard_pending_weight_halves(), [])

        # weights_proj, bf16 in the checkpoint, fills the bottom rows
        weights_proj = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
        loader.do_load_weights(
            [("model.layers.0.self_attn.indexer.weights_proj.weight", weights_proj)]
        )
        torch.testing.assert_close(fused_param[-2:], weights_proj)
        torch.testing.assert_close(fused_param[:2], dequantized)

    def test_end_weight_update_reports_halves_held_by_a_wrapped_model(self):
        """A VLM holds the DeepSeek model as a submodule; its halves still fail the update."""
        loader = _FakeDeepseekLoader(
            {_FUSED_INDEXER: torch.zeros((4, 4), dtype=torch.bfloat16)}
        )
        wrapper = torch.nn.Module()
        wrapper.language_model = loader
        names = [f"model.layers.{i}.self_attn.q_a_proj.weight" for i in range(9)]
        wk_scale_name = "model.layers.0.self_attn.indexer.wk.weight_scale_inv"
        loader.do_load_weights(
            [(name, torch.ones((2, 3))) for name in names]
            + [(wk_scale_name, torch.ones((1, 1)))]
        )
        updater = _weight_updater(wrapper)

        message = updater.end_weight_update(run_post_load=False)

        self.assertIn("one half of 10 fused weights", message)
        self.assertIn(names[0], message)
        self.assertIn(wk_scale_name, message)
        self.assertIsNone(updater.end_weight_update(run_post_load=False))

    def test_unfused_quark_scale_loads_into_scale_inv(self):
        target_name = "model.layers.0.self_attn.o_proj.weight_scale_inv"
        target = _FakeParam()
        loader = _FakeDeepseekLoader({target_name: target})
        loaded_scale = torch.ones((2, 3), dtype=torch.float32)

        loader.do_load_weights(
            [
                (
                    "model.layers.0.self_attn.o_proj.weight_scale",
                    loaded_scale,
                ),
            ]
        )

        self.assertIs(target.loaded, loaded_scale)
        self.assertTrue(loader.post_load_called)

    def test_quark_weight_scale_is_not_misclassified_as_indexer_weight(self):
        fused_name = "model.layers.0.self_attn.indexer.wk_weights_proj.weight"
        fused_param = torch.nn.Parameter(
            torch.zeros((4, 4), dtype=torch.bfloat16),
            requires_grad=False,
        )
        params_dict = {fused_name: fused_param}
        pending = {}
        loaded_weight = torch.ones((2, 4), dtype=torch.float8_e4m3fn)
        loaded_scale = torch.ones((1, 1), dtype=torch.float32)
        dequantized = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch(
            "sglang.srt.models.deepseek_common.deepseek_weight_loader."
            "block_quant_dequant",
            return_value=dequantized,
        ) as block_quant_dequant:
            self.assertTrue(
                _load_fused_indexer_wk(
                    "model.layers.0.self_attn.indexer.wk.weight",
                    loaded_weight,
                    params_dict,
                    pending,
                    None,
                )
            )
            self.assertTrue(
                _load_fused_indexer_wk(
                    "model.layers.0.self_attn.indexer.wk.weight_scale",
                    loaded_scale,
                    params_dict,
                    pending,
                    None,
                )
            )

        block_quant_dequant.assert_called_once()
        weight, scale, *rest = block_quant_dequant.call_args.args
        self.assertTrue(torch.equal(weight.float(), loaded_weight.float()))
        self.assertTrue(torch.equal(scale, loaded_scale))
        self.assertEqual(rest, [[128, 128], torch.bfloat16])
        torch.testing.assert_close(fused_param[:2], dequantized)
        self.assertEqual(pending, {})


if __name__ == "__main__":
    unittest.main()
