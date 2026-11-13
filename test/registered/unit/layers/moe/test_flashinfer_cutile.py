"""CPU contracts for the FlashInfer cuTile NVFP4 adapter."""

import gc
import sys
import unittest
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.moe.moe_runner import flashinfer_cutile
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import (
    PackedTopKOutput,
    StandardTopKOutput,
    StandardTopKOutputPacked,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.minimax_m3_nvfp4 import MiniMaxM3Nvfp4MoE
from sglang.srt.model_executor.model_runner_components import weight_exporter
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.runtime_context import (
    get_context,
    get_exec,
    get_flags,
    get_parallel,
    get_schedule,
    override_platform,
    reset_context,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _ModelWithExperts(torch.nn.Module):
    """Keep one ordinary parameter beside experts to detect partial mutations."""

    def __init__(self, experts):
        super().__init__()
        self.dense = torch.nn.Linear(2, 2, bias=False)
        self.experts = experts

    def load_weights(self, named_tensors):
        # A real write exposes a misplaced update guard; counting calls cannot.
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name, value in named_tensors:
                parameters[name].copy_(value)

    def get_weights_by_name(self, name, truncate_size, tp_size):
        return self.get_parameter(name)[:truncate_size]


class TestFlashInferCuTile(CustomTestCase):
    """Exercise real SGLang contracts with only the CUDA/FlashInfer boundary stubbed."""

    def setUp(self):
        super().setUp()
        reset_context()
        # Cleanup runs even when setup or a test assertion fails. Restore all
        # overrides before dropping the runtime resources created by this test.
        self.addCleanup(reset_context)
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(_runtime_scope())
        _stub_flashinfer_boundary(self._stack)

    def test_explicit_marlin_selects_portable_experts_on_sm12x(self):
        for device_sm in (120, 121):
            with (
                self.subTest(sm=device_sm),
                override_platform(device_sm=device_sm, is_blackwell=True),
                get_flags().moe.override(runner_backend=MoeRunnerBackend.MARLIN),
            ):
                scheme = MiniMaxM3Nvfp4MoE()
                self.assertTrue(scheme.use_marlin)
                self.assertFalse(scheme.use_flashinfer_cutile)
                self.assertFalse(scheme.use_flashinfer_trtllm)
                self.assertFalse(scheme.requantizes_experts)

    def test_marlin_opt_in_preserves_existing_expert_selection(self):
        cases = (
            (120, MoeRunnerBackend.FLASHINFER_CUTILE, MoeA2ABackend.NONE, False),
            (121, MoeRunnerBackend.FLASHINFER_CUTILE, MoeA2ABackend.NONE, False),
            (90, MoeRunnerBackend.MARLIN, MoeA2ABackend.NONE, True),
            (100, MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.MEGAMOE, False),
            (100, MoeRunnerBackend.MARLIN, MoeA2ABackend.MEGAMOE, False),
            (120, MoeRunnerBackend.DEEP_GEMM, MoeA2ABackend.MEGAMOE, False),
        )
        for device_sm, backend, a2a, use_marlin in cases:
            with (
                self.subTest(sm=device_sm, backend=backend),
                override_platform(device_sm=device_sm, is_blackwell=device_sm >= 100),
                get_flags().moe.override(runner_backend=backend, a2a_backend=a2a),
            ):
                scheme = MiniMaxM3Nvfp4MoE()
                self.assertEqual(scheme.use_marlin, use_marlin)
                self.assertEqual(
                    scheme.use_flashinfer_cutile,
                    backend == MoeRunnerBackend.FLASHINFER_CUTILE,
                )

    def test_experimental_marlin_does_not_enable_the_sm12x_opt_in(self):
        with (
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN
            ),
            self.assertRaisesRegex(ValueError, "experts require"),
        ):
            MiniMaxM3Nvfp4MoE()

    def test_weight_preparation_preserves_branch_scale_pairing(self):
        layer = _layer()
        original = {
            name: parameter.detach().clone()
            for name, parameter in layer.named_parameters()
        }
        prepared = flashinfer_cutile.prepare_flashinfer_cutile_nvfp4_weights(
            layer, _config()
        )
        view = prepared.weight_pack.views["cutile_nvfp4"]

        self.assertTrue(
            torch.equal(view.w1_fp4[:, :64], original["w13_weight_packed"][:, 64:])
        )
        self.assertTrue(
            torch.equal(view.w1_fp4[:, 64:], original["w13_weight_packed"][:, :64])
        )
        torch.testing.assert_close(
            view.w1_block_scale[:, :64].float(),
            original["w13_weight_scale"][:, 64:].float(),
        )
        torch.testing.assert_close(
            view.w1_block_scale[:, 64:].float(),
            original["w13_weight_scale"][:, :64].float(),
        )
        torch.testing.assert_close(
            view.w1_global_scale, torch.tensor([[4.0, 2.0], [6.0, 3.0]])
        )
        torch.testing.assert_close(view.w2_global_scale, torch.tensor([5.0, 7.0]))
        self.assertTrue(torch.equal(view.w2_fp4, original["w2_weight_packed"]))
        self.assertEqual(view.source_format, "compressed_tensors")
        self.assertEqual(
            (view.activation.alpha, view.activation.beta, view.activation.limit),
            (
                1.702,
                1.0,
                7.0,
            ),
        )
        # Source globals are encode scales; the adapter must not invert or mutate them.
        for name, original_value in original.items():
            if getattr(layer, name, None) is not None:
                torch.testing.assert_close(
                    getattr(layer, name).float(), original_value.float()
                )

    def test_prepared_weights_are_owned_by_each_layer(self):
        first, second = _layer(), _layer()
        second.w13_weight_packed.fill_(68)
        one = flashinfer_cutile.prepare_flashinfer_cutile_nvfp4_weights(
            first, _config()
        )
        two = flashinfer_cutile.prepare_flashinfer_cutile_nvfp4_weights(
            second, _config()
        )
        self.assertIsNot(one.weight_pack, two.weight_pack)
        self.assertFalse(
            torch.equal(
                one.weight_pack.views["cutile_nvfp4"].w1_fp4,
                two.weight_pack.views["cutile_nvfp4"].w1_fp4,
            )
        )

    def test_nonunit_activation_global_scale_is_rejected(self):
        for name in ["w13_input_global_scale", "w2_input_global_scale"]:
            with self.subTest(name=name):
                layer = _layer()
                getattr(layer, name).fill_(2.0)
                with self.assertRaisesRegex(ValueError, "scale"):
                    flashinfer_cutile.prepare_flashinfer_cutile_nvfp4_weights(
                        layer, _config()
                    )

    def test_unsupported_runner_config_is_rejected(self):
        for overrides in [
            {"params_dtype": torch.float16},
            {"activation": "relu"},
            {"is_gated": False},
            {"no_combine": True},
            {"apply_router_weight_on_input": True},
            {"num_fused_shared_experts": 1},
            {"silu_mul_keep_fp32": True},
            {"num_local_experts": 1},
            {"hidden_size": 65},
            {"intermediate_size_per_partition": 65},
        ]:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    flashinfer_cutile.validate_flashinfer_cutile_config(
                        _config(**overrides)
                    )

    def test_routing_values_and_output_scaling(self):
        for factor in [None, 2.5]:
            for topk_type in [StandardTopKOutput, StandardTopKOutputPacked]:
                with self.subTest(factor=factor, topk_type=topk_type):
                    config = _config(routed_scaling_factor=factor)
                    runner = flashinfer_cutile.FlashInferCuTileRunnerCore(config)
                    dispatch = _dispatch(topk_type)
                    original_weights = dispatch.topk_output.topk_weights.clone()
                    original_input = dispatch.hidden_states.clone()

                    payload = flashinfer_cutile.FlashInferCuTileMoeQuantInfo(
                        weight_pack=object(), device=torch.device("cpu")
                    )
                    result = runner.run_from_dispatch(
                        dispatch, payload, config
                    ).hidden_states
                    execution = runner._get_execution(torch.device("cpu"), 2)
                    activations = execution.layer.activations[-1]
                    torch.testing.assert_close(
                        activations.topk_weights, original_weights
                    )
                    self.assertEqual(activations.topk_ids.dtype, torch.int32)
                    expected = torch.tensor([8.0, 6.0])[:, None].expand(2, 64)
                    if factor is not None:
                        expected = expected * factor
                    torch.testing.assert_close(result.float(), expected)
                    torch.testing.assert_close(
                        dispatch.topk_output.topk_weights, original_weights
                    )
                    torch.testing.assert_close(dispatch.hidden_states, original_input)

    def test_empty_batch_does_not_create_execution(self):
        config = _config()
        runner = flashinfer_cutile.FlashInferCuTileRunnerCore(config)
        create = Mock(
            side_effect=AssertionError("empty batch must not create workspace")
        )
        self._stack.enter_context(
            patch.object(sys.modules["flashinfer.fused_moe"], "MoELayer", create)
        )
        dispatch = _dispatch(empty=True)
        payload = flashinfer_cutile.FlashInferCuTileMoeQuantInfo(
            weight_pack=object(), device=torch.device("cpu")
        )
        result = runner.run_from_dispatch(dispatch, payload, config).hidden_states
        self.assertEqual(result.shape, (0, 64))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_packed_only_routing_is_rejected(self):
        config = _config()
        runner = flashinfer_cutile.FlashInferCuTileRunnerCore(config)
        dispatch = _dispatch()
        dispatch = dispatch._replace(
            topk_output=PackedTopKOutput(
                packed_topk_ids=torch.zeros(2, 2, dtype=torch.int32),
                router_logits=torch.zeros(2, 2),
            )
        )
        payload = flashinfer_cutile.FlashInferCuTileMoeQuantInfo(
            weight_pack=object(), device=torch.device("cpu")
        )
        with self.assertRaisesRegex(
            (TypeError, ValueError), "[Ss]tandard|routing|top-?k"
        ):
            runner.run_from_dispatch(dispatch, payload, config)

    def test_invalid_weight_global_scale_is_rejected(self):
        for value in [0.0, -1.0, float("nan"), float("inf")]:
            with self.subTest(value=value):
                layer = _layer()
                layer.w2_weight_global_scale.fill_(value)
                with self.assertRaisesRegex(ValueError, "finite positive"):
                    flashinfer_cutile.prepare_flashinfer_cutile_nvfp4_weights(
                        layer, _config()
                    )

    def test_invalid_global_scale_columns_are_rejected(self):
        layer = _layer()
        layer.w13_weight_global_scale = torch.nn.Parameter(
            torch.ones(2, 3), requires_grad=False
        )
        with self.assertRaisesRegex(ValueError, "shape|global"):
            flashinfer_cutile.prepare_flashinfer_cutile_nvfp4_weights(layer, _config())

    def test_negative_padding_does_not_route_to_an_expert(self):
        config = _config()
        runner = flashinfer_cutile.FlashInferCuTileRunnerCore(config)
        dispatch = _dispatch()
        dispatch.topk_output.topk_ids[0, 1] = -1
        original_ids = dispatch.topk_output.topk_ids.clone()
        original_weights = dispatch.topk_output.topk_weights.clone()

        payload = flashinfer_cutile.FlashInferCuTileMoeQuantInfo(
            weight_pack=object(), device=torch.device("cpu")
        )
        result = runner.run_from_dispatch(dispatch, payload, config).hidden_states
        activations = runner._get_execution(torch.device("cpu"), 2).layer.activations[
            -1
        ]
        self.assertEqual(activations.topk_ids[0, 1].item(), 0)
        self.assertEqual(activations.topk_weights[0, 1].item(), 0)
        torch.testing.assert_close(
            result.float(), torch.tensor([2.0, 6.0])[:, None].expand(2, 64)
        )
        self.assertTrue(torch.equal(dispatch.topk_output.topk_ids, original_ids))
        torch.testing.assert_close(dispatch.topk_output.topk_weights, original_weights)

    def test_execution_pool_shares_only_matching_stream_and_geometry(self):
        stream = [10]
        self._stack.enter_context(
            patch.object(
                torch.cuda,
                "current_stream",
                lambda device: SimpleNamespace(cuda_stream=stream[0]),
            )
        )
        with get_schedule().override(max_prefill_tokens=8):
            first = flashinfer_cutile.FlashInferCuTileRunnerCore(_config())
            second = flashinfer_cutile.FlashInferCuTileRunnerCore(_config())
            device = torch.device("cpu")
            shared = first._get_execution(device, 2)
            self.assertIs(second._get_execution(device, 2), shared)

            stream[0] = 11
            self.assertIsNot(first._get_execution(device, 2), shared)
            stream[0] = 10
            other = flashinfer_cutile.FlashInferCuTileRunnerCore(
                _config(gemm1_beta=2.0)
            )
            self.assertIsNot(other._get_execution(device, 2), shared)
            with self.assertRaisesRegex(ValueError, "capacity"):
                first._get_execution(device, 9)

    def test_context_reset_releases_execution_while_runner_is_alive(self):
        """Old runner references must not keep a reset runtime's workspaces alive."""
        runner = flashinfer_cutile.FlashInferCuTileRunnerCore(_config())
        execution = runner._get_execution(torch.device("cpu"), 2)
        reference = weakref.ref(execution)
        del execution
        gc.collect()
        # Resources own the workspace, even when no forward holds it.
        self.assertIsNot(reference(), None)
        reset_context()
        gc.collect()
        self.assertIs(reference(), None)

        with _runtime_scope():
            self._stack.enter_context(
                patch.object(flashinfer_cutile, "_is_capturing", lambda device: True)
            )
            with self.assertRaisesRegex(RuntimeError, "warm"):
                runner._get_execution(torch.device("cpu"), 2)
            self._stack.enter_context(
                patch.object(flashinfer_cutile, "_is_capturing", lambda device: False)
            )
            self.assertIsNot(runner._get_execution(torch.device("cpu"), 2), None)

    def test_context_reset_does_not_reuse_old_execution_or_warmup(self):
        """A still-referenced old execution cannot leak warmup into a new context."""
        config = _config()
        runner = flashinfer_cutile.FlashInferCuTileRunnerCore(config)
        dispatch = _dispatch()
        payload = flashinfer_cutile.FlashInferCuTileMoeQuantInfo(
            weight_pack=object(), device=torch.device("cpu")
        )
        runner.run_from_dispatch(dispatch, payload, config)
        old = runner._get_execution(torch.device("cpu"), 2)
        self.assertIn(2, old.warmed_token_counts)
        reset_context()
        with _runtime_scope():
            new = runner._get_execution(torch.device("cpu"), 2)
            self.assertIsNot(new, old)
            self._stack.enter_context(
                patch.object(flashinfer_cutile, "_is_capturing", lambda device: True)
            )
            with self.assertRaisesRegex(RuntimeError, "warm"):
                runner.run_from_dispatch(dispatch, payload, config)

    def test_cold_execution_cannot_be_created_during_capture(self):
        self._stack.enter_context(
            patch.object(flashinfer_cutile, "_is_capturing", lambda device: True)
        )
        create = Mock(side_effect=AssertionError("must not allocate during capture"))
        self._stack.enter_context(
            patch.object(sys.modules["flashinfer.fused_moe"], "MoELayer", create)
        )
        runner = flashinfer_cutile.FlashInferCuTileRunnerCore(_config())
        with self.assertRaisesRegex(RuntimeError, "warm"):
            runner._get_execution(torch.device("cpu"), 2)

    def test_capture_requires_a_warmed_token_count(self):
        capture = [True]
        self._stack.enter_context(
            patch.object(flashinfer_cutile, "_is_capturing", lambda device: capture[0])
        )
        config = _config()
        execution = flashinfer_cutile._CuTileExecution(
            config=config, device=torch.device("cpu"), max_num_tokens=8
        )
        act = SimpleNamespace(
            hidden_states_q=torch.ones(2, 64),
            topk_ids=torch.zeros(2, 2, dtype=torch.int32),
            topk_weights=torch.ones(2, 2),
        )
        with self.assertRaisesRegex(RuntimeError, "warm"):
            execution(act, object())
        capture[0] = False
        expected = execution(act, object())
        capture[0] = True
        torch.testing.assert_close(execution(act, object()), expected)
        act.hidden_states_q = torch.ones(3, 64)
        with self.assertRaisesRegex(RuntimeError, "warm"):
            execution(act, object())

    def test_dynamic_checkpoint_omitted_input_scales_default_to_one(self):
        scheme = _minimax_scheme(use_cutile=True)
        layer = _allocate_scheme_layer(scheme)
        torch.testing.assert_close(layer.w13_input_global_scale, torch.ones(2, 2))
        torch.testing.assert_close(layer.w2_input_global_scale, torch.ones(2))
        torch.testing.assert_close(
            layer.w13_weight_global_scale, torch.full((2, 2), 13.0)
        )
        scheme.moe_runner_config = _config(layer=layer)
        # The actual load-completion path must accept a checkpoint with no input scales.
        scheme.process_weights_after_loading(layer)

    def test_checkpoint_input_scale_override_is_preserved_and_rejected(self):
        for name in ["w13_input_global_scale", "w2_input_global_scale"]:
            with self.subTest(name=name):
                scheme = _minimax_scheme(use_cutile=True)
                layer = _allocate_scheme_layer(scheme)
                # Simulate the loader overwriting the newly allocated default with a real value.
                getattr(layer, name).fill_(2.0)
                scheme.moe_runner_config = _config(layer=layer)
                with self.assertRaisesRegex(ValueError, "scale"):
                    scheme.process_weights_after_loading(layer)
                torch.testing.assert_close(
                    getattr(layer, name), torch.full_like(getattr(layer, name), 2.0)
                )

    def test_non_cutile_weight_allocation_keeps_existing_behavior(self):
        scheme = _minimax_scheme(use_cutile=False)
        layer = _allocate_scheme_layer(scheme)
        self.assertFalse(hasattr(layer, "_weight_export_error"))
        torch.testing.assert_close(
            layer.w13_input_global_scale, torch.full((2, 2), 13.0)
        )
        torch.testing.assert_close(layer.w2_input_global_scale, torch.full((2,), 13.0))

    def test_prepared_weights_reject_reload_without_discarding_current_pack(self):
        """A reload must fail explicitly before replacing the live prepared weights."""
        scheme = _minimax_scheme(use_cutile=True)
        layer = _allocate_scheme_layer(scheme)
        # A cold loader may invoke restore before the first preparation.
        scheme.restore_weights_before_loading(layer)
        scheme.moe_runner_config = _config(layer=layer)
        scheme.process_weights_after_loading(layer)
        prepared = scheme.cutile_quant_info
        view = prepared.weight_pack.views["cutile_nvfp4"]
        execution_weights = {
            name: value.detach().clone()
            for name, value in vars(view).items()
            if isinstance(value, torch.Tensor)
        }
        remaining = {
            name: value.detach().clone() for name, value in layer.named_parameters()
        }
        for operation in (
            scheme.restore_weights_before_loading,
            scheme.process_weights_after_loading,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Online weight updates.*flashinfer_cutile"
            ):
                operation(layer)
            self.assertIs(scheme.cutile_quant_info, prepared)
            for name, value in execution_weights.items():
                torch.testing.assert_close(getattr(view, name).float(), value.float())
            self.assertEqual(set(dict(layer.named_parameters())), set(remaining))
            for name, value in layer.named_parameters():
                torch.testing.assert_close(value.float(), remaining[name].float())

    def test_online_updates_reject_before_mutating_any_model_weight(self):
        """The real updater must see the nested expert marker before any loader runs."""
        for prepared in [False, True]:
            with self.subTest(prepared=prepared):
                scheme = _minimax_scheme(use_cutile=True)
                layer = _allocate_scheme_layer(scheme)
                self.assertIn("flashinfer_cutile", layer._derived_weight_cache_error)
                scheme.moe_runner_config = _config(layer=layer)
                if prepared:
                    scheme.process_weights_after_loading(layer)

                model = _ModelWithExperts(layer)
                before = {
                    name: value.detach().clone()
                    for name, value in model.named_parameters()
                }
                updater = WeightUpdater(
                    tp_rank=0,
                    device="cpu",
                    gpu_id=0,
                    model_config=None,
                    custom_weight_loaders={},
                    get_model=lambda: model,
                    update_model_fields=lambda **kwargs: None,
                    recapture_cuda_graph=lambda: None,
                    get_model_runner=lambda: None,
                )
                tensors = [("dense.weight", torch.zeros_like(model.dense.weight))]
                reason = layer._derived_weight_cache_error
                self.assertEqual(updater.load_rejection("plugin_weight_update"), reason)
                for result in (
                    updater.update_weights_from_disk("/unused/checkpoint", "auto"),
                    updater.update_weights_from_tensor(tensors),
                    updater.update_weights_from_tensor(tensors, load_format="direct"),
                    updater.load_weights_from_distributed(tensors),
                    updater.update_weights_from_ipc(None),
                ):
                    self.assertEqual(result, (False, reason))
                self.assertEqual(set(dict(model.named_parameters())), set(before))
                for name, value in model.named_parameters():
                    torch.testing.assert_close(value.float(), before[name].float())

    def test_exports_reject_before_writing_or_sending_incomplete_weights(self):
        """Unregistered expert weights must not produce a successful partial export."""
        for prepared in [False, True]:
            with self.subTest(prepared=prepared), ExitStack() as export_scope:
                from sglang.srt.model_loader.loader import (
                    RemoteModelLoader,
                    ShardedStateLoader,
                )

                scheme = _minimax_scheme(use_cutile=True)
                layer = _allocate_scheme_layer(scheme)
                scheme.moe_runner_config = _config(layer=layer)
                if prepared:
                    scheme.process_weights_after_loading(layer)
                reason = layer._weight_export_error
                self.assertTrue(
                    "Weight export" in reason and "flashinfer_cutile" in reason
                )

                model = _ModelWithExperts(layer)
                parameters_before = {
                    name: value.detach().clone()
                    for name, value in model.named_parameters()
                }
                if prepared:
                    quant_info = scheme.cutile_quant_info
                    view = quant_info.weight_pack.views["cutile_nvfp4"]
                    execution_weights = {
                        name: value.detach().clone()
                        for name, value in vars(view).items()
                        if isinstance(value, torch.Tensor)
                    }

                def unexpected_side_effect(*args, **kwargs):
                    self.fail(
                        "A rejected cuTile export must not start storage or communication"
                    )

                # Keep WeightExporter real. Only replace storage/communication boundaries
                # with tripwires, so even a misplaced guard cannot write or send test data.
                for target, attribute in (
                    (ShardedStateLoader, "save_model"),
                    (RemoteModelLoader, "save_model"),
                    (weight_exporter, "init_custom_process_group"),
                    (torch.distributed, "is_initialized"),
                    (torch.distributed, "broadcast"),
                    (torch.distributed, "barrier"),
                ):
                    export_scope.enter_context(
                        patch.object(target, attribute, unexpected_side_effect)
                    )
                exporter = weight_exporter.WeightExporter(
                    tp_rank=0,
                    tp_size=1,
                    gpu_id=0,
                    get_model_path=unexpected_side_effect,
                    get_model=lambda: model,
                )
                tmp_path = Path(export_scope.enter_context(TemporaryDirectory()))
                save_path = tmp_path / "sharded"
                with self.assertRaisesRegex(
                    RuntimeError, "Weight export.*flashinfer_cutile"
                ):
                    exporter.save_sharded_model(str(save_path))
                with self.assertRaisesRegex(
                    RuntimeError, "Weight export.*flashinfer_cutile"
                ):
                    exporter.save_remote_model("redis://unused/checkpoint")
                self.assertEqual(
                    exporter.init_weights_send_group_for_remote_instance(
                        master_address="127.0.0.1",
                        ports="29500",
                        group_rank=0,
                        world_size=1,
                        group_name="cutile_export_test",
                    ),
                    (False, reason),
                )
                self.assertEqual(
                    exporter.send_weights_to_remote_instance(
                        master_address="127.0.0.1",
                        ports="29500",
                        group_name="cutile_export_test",
                    ),
                    (False, reason),
                )
                self.assertIsNone(exporter.get_weights_by_name("dense.weight"))
                self.assertFalse(save_path.exists())
                self.assertEqual(list(tmp_path.iterdir()), [])
                self.assertEqual(exporter._weights_send_group, {})
                self.assertEqual(
                    set(dict(model.named_parameters())), set(parameters_before)
                )
                for name, value in model.named_parameters():
                    torch.testing.assert_close(
                        value.float(), parameters_before[name].float()
                    )
                if prepared:
                    self.assertIs(scheme.cutile_quant_info, quant_info)
                    for name, value in execution_weights.items():
                        torch.testing.assert_close(
                            getattr(view, name).float(), value.float()
                        )


@contextmanager
def _runtime_scope(max_num_tokens=512):
    """Publish real config; exercise the same accessors as model execution."""
    with (
        get_context().override_server_args(
            max_prefill_tokens=max_num_tokens, disable_cuda_graph=True
        ),
        get_exec().graph.override(cuda_graph_config=None),
        get_parallel().override(moe_ep_size=1),
        get_flags().moe.override(
            a2a_backend=MoeA2ABackend.NONE,
            runner_backend=MoeRunnerBackend.FLASHINFER_CUTILE,
        ),
        override_platform(is_blackwell=True, is_hip=False, device_sm=120),
    ):
        yield


def _stub_flashinfer_boundary(stack):
    """Keep SGLang interfaces real; replace only FlashInfer and CUDA boundaries."""
    names = ("flashinfer", "flashinfer.fused_moe", "flashinfer.fused_moe.core")
    modules = {name: ModuleType(name) for name in names}

    class WeightPack:
        def __init__(self):
            self.views = {}

        def prepare_for(self, name, view):
            self.views[name] = view

    class CuTileConfig:
        @staticmethod
        def prepare_weights(**kwargs):
            return SimpleNamespace(**kwargs)

    class MoELayer:
        def __init__(self, *, config, device):
            self.config = config
            self.device = device
            self.activations = []

        def __call__(self, activations, weights):
            self.activations.append(activations)
            # A deterministic external-kernel substitute makes routing and
            # output-scale changes observable, without pretending to run FP4.
            totals = (activations.topk_weights * (activations.topk_ids + 1)).sum(
                dim=-1, keepdim=True
            )
            return (activations.hidden_states_q.float() * totals).bfloat16()

    provider = modules["flashinfer.fused_moe"]
    provider.CuTileNvfp4Config = CuTileConfig
    provider.MoEWeightPack = WeightPack
    provider.MoELayer = MoELayer
    for name in (
        "SwiGLU",
        "MoEActivationPack",
        "MoEConfig",
        "RoutingConfig",
        "QuantConfig",
        "ExpertConfig",
        "BackendOptions",
        "ExecutionConfig",
    ):
        setattr(provider, name, SimpleNamespace)
    provider.QuantFormat = SimpleNamespace(NVFP4="nvfp4")
    modules["flashinfer.fused_moe.core"].RoutingInputMode = SimpleNamespace(
        PackedPrecomputed="packed_precomputed"
    )
    stack.enter_context(patch.dict(sys.modules, modules))

    # The adapter uses CPU tensors here; actual hardware eligibility and driver
    # behavior remain covered by the separate manual SM120/SM121 tests.
    stack.enter_context(
        patch.object(flashinfer_cutile, "_check_device_support", lambda device: None)
    )
    stack.enter_context(
        patch.object(flashinfer_cutile, "_is_capturing", lambda device: False)
    )
    stack.enter_context(
        patch.object(
            torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=10)
        )
    )


def _config(**overrides):
    layer = torch.nn.Module()
    layer.moe_ep_size = layer.moe_tp_size = 1
    layer.moe_ep_rank = layer.moe_tp_rank = 0
    values = dict(
        num_experts=2,
        num_local_experts=2,
        hidden_size=64,
        intermediate_size_per_partition=64,
        top_k=2,
        params_dtype=torch.bfloat16,
        num_fused_shared_experts=0,
        gemm1_alpha=1.702,
        gemm1_beta=1.0,
        gemm1_clamp_limit=7.0,
        gate_up_interleaved=False,
        layer=layer,
    )
    values.update(overrides)
    return MoeRunnerConfig(**values)


def _layer():
    # Distinct gate/up bytes and scale values expose mismatched permutations.
    # These sentinels exercise loading contracts, not FP4 numerical accuracy.
    layer = torch.nn.Module()
    values = {
        "w13_weight_packed": torch.cat(
            [torch.full((2, 64, 32), 17), torch.full((2, 64, 32), 34)], dim=1
        ).to(torch.uint8),
        "w13_weight_scale": torch.cat(
            [torch.full((2, 64, 4), 2.0), torch.full((2, 64, 4), 3.0)], dim=1
        ).to(torch.float8_e4m3fn),
        "w13_weight_global_scale": torch.tensor([[2.0, 4.0], [3.0, 6.0]]),
        "w13_input_global_scale": torch.ones(2, 2),
        "w2_weight_packed": torch.full((2, 64, 32), 51, dtype=torch.uint8),
        "w2_weight_scale": torch.full((2, 64, 4), 4.0).to(torch.float8_e4m3fn),
        "w2_weight_global_scale": torch.tensor([5.0, 7.0]),
        "w2_input_global_scale": torch.ones(2),
    }
    for name, value in values.items():
        layer.register_parameter(name, torch.nn.Parameter(value, requires_grad=False))
    layer.num_experts = layer.num_local_experts = 2
    layer.hidden_size = layer.intermediate_size_per_partition = 64
    layer.params_dtype = torch.bfloat16
    layer.moe_ep_size = layer.moe_tp_size = 1
    layer.moe_ep_rank = layer.moe_tp_rank = 0
    return layer


def _dispatch(topk_type=StandardTopKOutput, *, empty=False):
    count = 0 if empty else 2
    values = dict(
        topk_ids=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)[:count],
        topk_weights=torch.tensor([[2.0, 3.0], [1.0, 4.0]])[:count],
        router_logits=torch.zeros(count, 2),
    )
    if topk_type is StandardTopKOutputPacked:
        values["packed_topk_ids"] = torch.zeros(count, 2, dtype=torch.int32)
    return StandardDispatchOutput(
        hidden_states=torch.ones(count, 64, dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=topk_type(**values),
    )


def _minimax_scheme(*, use_cutile):
    backend = (
        MoeRunnerBackend.FLASHINFER_CUTILE if use_cutile else MoeRunnerBackend.MARLIN
    )
    with (
        get_flags().moe.override(runner_backend=backend),
        override_platform(is_blackwell=use_cutile, is_hip=False),
    ):
        return MiniMaxM3Nvfp4MoE()


def _allocate_scheme_layer(scheme):
    # Give uninitialized float32 storage a deterministic marker. The parent
    # allocator and MiniMax override still execute their real implementations.
    empty = torch.empty

    def marked_empty(*args, **kwargs):
        value = empty(*args, **kwargs)
        if value.dtype == torch.float32:
            value.fill_(13.0)
        elif value.dtype in (torch.uint8, torch.float8_e4m3fn):
            value.fill_(1)
        return value

    layer = torch.nn.Module()
    layer.moe_ep_size = layer.moe_tp_size = 1
    layer.moe_ep_rank = layer.moe_tp_rank = 0
    with patch.object(torch, "empty", marked_empty):
        scheme.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )
    return layer


if __name__ == "__main__":
    unittest.main()
