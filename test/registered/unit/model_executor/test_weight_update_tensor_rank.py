"""A tensor weight update loads the entry of the process's deployment TP rank."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.layers.hyperconnection_packed import (
    PackedGatedResidual,
    create_gated_residual,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
    WeightUpdater,
)
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


class TestHCInitialLoadAndUpdateGuard(CustomTestCase):
    @torch.no_grad()
    def test_checkpoint_inject_names_follow_hc_storage(self):
        """Inject aliases must select packed storage without renaming legacy HC."""
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
        from sglang.srt.models.qwen4_exp_mtp import Qwen4ExpForCausalLMMTP

        config = HyperConnectionConfig(
            hidden_size=512,
            hc_lowrank=64,
            hc_per_branch_norm=True,
            params_dtype=torch.bfloat16,
        )
        checkpoint = {
            "input_mix_weight_down.weight": torch.ones(64, 2048),
            "block_inject_weight.weight": torch.full((4, 2048), 2.0),
            "hc_norm.weight": torch.zeros(2048),
            "input_mix_weight_up.weight": torch.full((2048, 64), 3.0),
        }
        for model_type, prefix in (
            (Qwen4ExpForConditionalGeneration, "model.language_model"),
            (Qwen4ExpForCausalLMMTP, "mtp"),
        ):
            for packed in (False, True):
                with self.subTest(model=model_type.__name__, packed=packed):
                    model = model_type.__new__(model_type)
                    torch.nn.Module.__init__(model)
                    model.config = SimpleNamespace(tie_word_embeddings=False)
                    model.pp_group = SimpleNamespace(is_last_rank=True)
                    model.model = torch.nn.Module()
                    model.model.start_layer, model.model.end_layer = 0, 1
                    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
                    layer = model.model.layers[0]
                    with patch("torch.cuda.is_available", return_value=False):
                        hc_type = PackedGatedResidual if packed else GatedResidual
                        layer.hc = hc_type(config)
                        if prefix != "mtp":
                            layer.legacy = GatedResidual(config)
                            layer.alias = layer.hc
                    weights = [
                        (f"{prefix}.layers.0.self_attn.hc.{name}", value)
                        for name, value in checkpoint.items()
                    ]
                    if prefix != "mtp":
                        weights.extend(
                            (
                                (
                                    "model.layers.0.legacy.block_inject_weight.weight",
                                    checkpoint["block_inject_weight.weight"],
                                ),
                                (
                                    "model.layers.0.alias.block_inject_weight.weight",
                                    checkpoint["block_inject_weight.weight"],
                                ),
                            )
                        )
                    loaded = model.load_weights(weights)
                    if packed:
                        down = layer.hc.input_mix_weight_down.weight
                        torch.testing.assert_close(
                            down[:, :64], torch.ones_like(down[:, :64])
                        )
                        torch.testing.assert_close(
                            down[:, 64:68], torch.full_like(down[:, 64:68], 2)
                        )
                    else:
                        torch.testing.assert_close(
                            layer.hc.block_inject_weight.weight,
                            checkpoint["block_inject_weight.weight"].to(
                                config.params_dtype
                            ),
                        )
                    if prefix != "mtp":
                        alias_weight = (
                            "input_mix_weight_down" if packed else "block_inject_weight"
                        )
                        self.assertIn(
                            f"model.layers.0.alias.{alias_weight}.weight", loaded
                        )
                        torch.testing.assert_close(
                            layer.legacy.block_inject_weight.weight,
                            checkpoint["block_inject_weight.weight"].to(
                                config.params_dtype
                            ),
                        )

    def test_packed_lowrank_selection(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            for lowrank, packed in (
                (8, False),
                (288, False),
                (64, True),
                (128, False),
                (192, True),
                (320, True),
            ):
                with self.subTest(lowrank=lowrank):
                    config = HyperConnectionConfig(
                        hidden_size=512,
                        hc_lowrank=lowrank,
                        hc_per_branch_norm=True,
                        params_dtype=torch.bfloat16,
                    )
                    layer = create_gated_residual(config)
                    if packed:
                        self.assertIsInstance(layer, PackedGatedResidual)
                    else:
                        self.assertIs(type(layer), GatedResidual)
                    if lowrank % 64:
                        with self.assertRaisesRegex(ValueError, "aligned to 64"):
                            PackedGatedResidual(config)

    def _updater(self, model):
        return WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda **kwargs: self.fail("model metadata changed"),
            recapture_cuda_graph=lambda: self.fail("Graph was recaptured"),
            get_model_runner=lambda: self.fail("checkpoint worker was constructed"),
        )

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
                        hc_lowrank=64,
                        hc_per_branch_norm=True,
                        params_dtype=torch.bfloat16,
                    )
                )
                with self.assertRaisesRegex(ValueError, "have not been loaded"):
                    model.load_weights([])
                self.assertFalse(model.hc._hc_weights_ready)
                model.hc._hc_weights_ready = True
                with self.assertRaisesRegex(ValueError, "Online weight updates"):
                    model.load_weights([])

    def test_packed_hc_rejects_update_entrypoints_before_mutation(self):
        model = torch.nn.Module()
        model.hc = PackedGatedResidual(
            HyperConnectionConfig(
                hidden_size=512, hc_lowrank=64, hc_per_branch_norm=True
            )
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        model.load_weights = lambda weights: self.fail("an update reached the loader")
        updater = self._updater(model)
        calls = (
            lambda: updater.update_weights_from_disk("unused", "auto"),
            lambda: updater.load_weights_from_distributed([]),
            lambda: updater.update_weights_from_tensor([], load_format="direct"),
            lambda: updater.update_weights_from_tensor(
                [], load_format="flattened_bucket"
            ),
            lambda: updater.update_weights_from_ipc(SimpleNamespace(zmq_handles={})),
        )
        with patch.object(WeightUpdater, "_assert_weight_cache_inactive"):
            for index, call in enumerate(calls):
                with self.subTest(entrypoint=index):
                    success, message = call()
                    self.assertFalse(success)
                    self.assertIn("Online weight updates are not supported", message)
            for call in (
                updater.begin_weight_update,
                lambda: updater.end_weight_update(run_post_load=True),
            ):
                with self.assertRaisesRegex(ValueError, "Online weight updates"):
                    call()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_other_derived_caches_keep_existing_native_sessions(self):
        """A non-HC cache marker alone must not disable native-memory sessions."""
        model = torch.nn.Linear(2, 2, bias=False)
        model._derived_weight_cache_error = "existing non-HC derived cache"
        updater = self._updater(model)
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater."
                "DefaultModelLoader.restore_weights_before_loading"
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater."
                "DefaultModelLoader.postprocess_weights"
            ),
        ):
            updater.begin_weight_update()
            with torch.no_grad():
                model.weight.fill_(2)
            updater.end_weight_update(run_post_load=True)
        torch.testing.assert_close(model.weight, torch.full((2, 2), 2.0))

    def test_model_state_dict_guards_run_before_any_parameter_is_written(self):
        """Both actual model constructors must install the root reload guard."""
        from sglang.srt.models import qwen4_exp, qwen4_exp_mtp

        config = SimpleNamespace(hidden_size=512, hc_count=4, vocab_size=8)

        def init_parent(instance, config, *_):
            torch.nn.Module.__init__(instance)
            instance.config = config
            instance.language_model_only = True
            instance.visual = None

        with patch.object(
            qwen4_exp.Qwen3VLForConditionalGeneration, "__init__", init_parent
        ):
            models = [qwen4_exp.Qwen4ExpForConditionalGeneration(config)]
        with (
            patch.object(
                qwen4_exp_mtp,
                "get_parallel",
                return_value=SimpleNamespace(pp_group=None, enable_dp_lm_head=False),
            ),
            patch.object(
                qwen4_exp_mtp.Qwen4ExpForCausalLMMTP,
                "_init_mtp_input_fusion",
                return_value=None,
            ),
            patch.object(
                qwen4_exp_mtp, "Qwen4ExpModel", return_value=torch.nn.Module()
            ),
            patch.object(
                qwen4_exp_mtp, "ParallelLMHead", return_value=torch.nn.Identity()
            ),
            patch.object(
                qwen4_exp_mtp, "LogitsProcessor", return_value=torch.nn.Identity()
            ),
        ):
            models.append(qwen4_exp_mtp.Qwen4ExpForCausalLMMTP(config))
        for model in models:
            with self.subTest(model=type(model).__name__):
                model.before_hc = torch.nn.Parameter(
                    torch.zeros(1), requires_grad=False
                )
                model.hc = PackedGatedResidual(
                    HyperConnectionConfig(
                        hidden_size=512, hc_lowrank=64, hc_per_branch_norm=True
                    )
                )
                model.hc._hc_weights_ready = True
                with self.assertRaisesRegex(ValueError, "Online weight updates"):
                    model.load_state_dict({"before_hc": torch.ones(1)}, strict=False)
                torch.testing.assert_close(
                    model.before_hc, torch.zeros(1), rtol=0, atol=0
                )


if __name__ == "__main__":
    unittest.main()
