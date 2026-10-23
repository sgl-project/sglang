"""A tensor weight update loads the entry of the process's deployment TP rank."""

import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.hyperconnection import HyperConnectionConfig
from sglang.srt.layers.hyperconnection_packed import (
    PackedGatedResidual,
    PackedHCModelMixin,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, post_load_weights
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTensorUpdateRank(unittest.TestCase):
    def test_a_narrowed_updater_reads_the_deployment_rank_entry(self):
        model = torch.nn.Linear(2, 2, bias=False)
        # An attention-owning draft records its attention-TP rank at init.
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        per_rank = [torch.full((2, 2), float(rank)) for rank in range(2)]
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in per_rank]
        )
        with (
            # This test checks CPU payload rank selection, not CUDA UUID
            # serialization. Do not leave a process-global CUDA reducer patch
            # installed for unrelated CPU IPC tests in the same pytest process.
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater."
                "monkey_patch_torch_reductions"
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=1),
        ):
            success, message = updater.update_weights_from_tensor(
                [("weight", payload)], load_format="direct"
            )
        self.assertTrue(success, message)
        self.assertTrue(torch.equal(model.weight.detach(), per_rank[1]))


class _HCModel(PackedHCModelMixin, torch.nn.Module):
    pass


class TestHCWeightUpdateCompletion(CustomTestCase):
    def test_checkpoint_load_rejects_completely_missing_hc_weights(self):
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
        from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

        for model_type in (
            Qwen4ExpForConditionalGeneration,
            Qwen4ExpForCausalLMMTP,
        ):
            with self.subTest(model=model_type.__name__):
                model = model_type.__new__(model_type)
                torch.nn.Module.__init__(model)
                model.config = SimpleNamespace()
                model.hc = PackedGatedResidual(
                    HyperConnectionConfig(
                        hidden_size=512,
                        hc_lowrank=8,
                        hc_per_branch_norm=True,
                        params_dtype=torch.bfloat16,
                    )
                )
                with self.assertRaisesRegex(ValueError, "have not been loaded"):
                    model.load_weights([])
                self.assertFalse(model.hc._hc_weights_ready)

    def test_checkpoint_engine_buckets_finalize_once_and_recover(self):
        model = _HCModel()
        model.hc = PackedGatedResidual(
            HyperConnectionConfig(
                hidden_size=512,
                hc_lowrank=8,
                hc_per_branch_norm=True,
                params_dtype=torch.bfloat16,
            )
        )
        payload = {
            "input_mix_weight_down.weight": torch.ones(8, 2048, dtype=torch.bfloat16),
            "block_inject_weight.weight": torch.full(
                (4, 2048), 2, dtype=torch.bfloat16
            ),
            "input_mix_weight_up.weight": torch.full(
                (2048, 8), 3, dtype=torch.bfloat16
            ),
            "hc_norm.weight": torch.full((2048,), 0.5, dtype=torch.bfloat16),
        }

        def load_bucket(items):
            parameters = dict(model.hc.named_parameters())
            for name, value in items:
                name = name.replace(
                    "block_inject_weight.weight", "input_mix_weight_down.weight"
                )
                parameter = parameters[name]
                parameter.weight_loader(parameter, value)
            self.assertFalse(model.hc.prepare_sum_state_weights())

        model.load_weights = load_bucket
        should_fail = False

        class Worker:
            def __init__(self, runner):
                self.runner = runner

            def update_weights_from_ipc(self, handles):
                for index, item in enumerate(payload.items()):
                    self.runner.model.load_weights([item])
                    if should_fail and index == 0:
                        raise RuntimeError("interrupted synthetic IPC update")

        fake = ModuleType("sglang.srt.checkpoint_engine.checkpoint_engine_worker")
        fake.SGLangCheckpointEngineWorkerExtensionImpl = Worker
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: SimpleNamespace(model=model),
        )
        with (
            patch.dict(sys.modules, {fake.__name__: fake}),
            patch.object(WeightUpdater, "_assert_weight_cache_inactive"),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater._unsupported_derived_weight_cache_error",
                return_value=None,
            ),
        ):
            request = SimpleNamespace(zmq_handles={})
            success, message = updater.update_weights_from_ipc(request)
            self.assertTrue(success, message)
            torch.testing.assert_close(
                model.hc._hc_packed_down[:, :8],
                torch.full((4, 8, 512), 1.5, dtype=torch.bfloat16),
            )
            self.assertFalse(model.hc._hc_update_open)
            should_fail = True
            success, _ = updater.update_weights_from_ipc(request)
            self.assertFalse(success)
            self.assertFalse(model.hc._hc_weights_ready)
            self.assertFalse(model.hc._hc_update_open)
            with self.assertRaisesRegex(ValueError, "complete raw or native reload"):
                post_load_weights(model)
            self.assertFalse(model.hc._hc_weights_ready)
            should_fail = False
            updater.begin_weight_update()
            success, message = updater.update_weights_from_ipc(request)
            self.assertTrue(success, message)
            self.assertTrue(model.hc._hc_update_open)  # Outer session still owns end.
            updater.end_weight_update(run_post_load=False)
            self.assertTrue(model.hc._hc_weights_ready)
            self.assertFalse(model.hc._hc_update_open)
            torch.testing.assert_close(
                model.hc._hc_packed_down[:, :8],
                torch.full((4, 8, 512), 1.5, dtype=torch.bfloat16),
            )

    def test_pending_hc_rejects_completion_even_when_post_load_is_skipped(self):
        """A successful bucket is not a complete HC update or a valid initial load."""
        hc = PackedGatedResidual.__new__(PackedGatedResidual)
        torch.nn.Module.__init__(hc)
        hc._packed_sum_weights = True
        hc._hc_loaded_weights = {"down"}
        hc._sum_weights_dirty = True
        hc._sum_operators = (object(), object())
        hc._hc_load_format = "checkpoint"
        hc._hc_weights_ready = True
        hc._hc_needs_full_reload = False
        hc._hc_update_open = True
        hc._hc_refresh_up = False
        model = _HCModel()
        model.add_module("hc", hc)
        model.load_weights = lambda weights: None
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater."
                "DefaultModelLoader.postprocess_weights",
            ) as postprocess,
            patch("sglang.srt.model_loader.loader.is_cuda_alike", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "hc:.*Down and Inject"):
                updater.end_weight_update(run_post_load=False)
            with self.assertRaisesRegex(ValueError, "hc:.*Down and Inject"):
                post_load_weights(model)
            # Initial/default disk loading has no begin/end update session.
            with self.assertRaisesRegex(ValueError, "hc:.*Down and Inject"):
                DefaultModelLoader.load_weights_and_postprocess(
                    model, [], torch.device("cpu")
                )
            postprocess.assert_not_called()

            hc._hc_loaded_weights.clear()
            hc._hc_load_format = None
            hc._hc_refresh_up = False
            hc._sum_weights_dirty = False
            updater.end_weight_update(run_post_load=False)
            postprocess.assert_called_once()


if __name__ == "__main__":
    unittest.main()
