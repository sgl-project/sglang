"""CPU unit tests for how ModelOptNvFp4FusedMoEMethod resolves ``auto``.

The flags that pick the weight layout (create_weights,
process_weights_after_loading) and the runner create_moe_runner builds must
come from one resolution: a layer prepared for one runner and dispatched to
another fails on its first forward. The device and parallel state are stubbed
so the tests stay on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import contextlib
import itertools
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.utils import (
    MoeRunnerBackend,
    speculative_moe_backend_context,
)
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.srt.runtime_context import get_flags
from sglang.test.test_utils import CustomTestCase


@contextmanager
def _device(
    capability: tuple,
    live: MoeRunnerBackend,
    tp_ep: tuple = (1, 1),
    device_0_capability: tuple | None = None,
    explicitly_set: bool | None = None,
):
    """``_worker_gpu`` with ``live`` as the process-wide backend;
    ``explicitly_set`` is whether the user passed it, by default whether
    ``live`` is anything but auto."""
    if explicitly_set is None:
        explicitly_set = live is not MoeRunnerBackend.AUTO
    with (
        _worker_gpu(capability, tp_ep, device_0_capability),
        mock.patch.object(modelopt_quant, "get_moe_runner_backend", return_value=live),
        mock.patch.object(
            modelopt_quant,
            "is_moe_runner_backend_explicit",
            return_value=explicitly_set,
        ),
    ):
        yield


@contextmanager
def _worker_gpu(
    capability: tuple,
    tp_ep: tuple = (1, 1),
    device_0_capability: tuple | None = None,
):
    """The worker's CUDA device 1 of ``capability`` (device 0 may differ), with
    ``tp_ep`` as the tensor and expert parallel sizes."""
    # The platform flags describe device 0.
    is_blackwell = (device_0_capability or capability) >= (10, 0)
    platform = SimpleNamespace(is_blackwell=is_blackwell)
    parallel = SimpleNamespace(tp_size=tp_ep[0], moe_ep_size=tp_ep[1])
    capabilities = {0: device_0_capability or capability, 1: capability}
    with (
        mock.patch.object(modelopt_quant, "get_platform", return_value=platform),
        mock.patch.object(modelopt_quant, "get_parallel", return_value=parallel),
        mock.patch.object(modelopt_quant, "is_cuda", return_value=True),
        mock.patch.object(modelopt_quant, "get_cuda_version", return_value=(12, 8)),
        mock.patch.object(modelopt_quant.torch.cuda, "current_device", return_value=1),
        mock.patch.object(
            modelopt_quant,
            "get_device_capability",
            side_effect=lambda device_id=0: capabilities[device_id],
        ),
    ):
        yield


def _method(
    is_w4a16: bool = False, per_token: bool = False
) -> ModelOptNvFp4FusedMoEMethod:
    return ModelOptNvFp4FusedMoEMethod(
        SimpleNamespace(use_per_token_activation=per_token, is_w4a16=is_w4a16)
    )


def _create_runner(method: ModelOptNvFp4FusedMoEMethod) -> MoeRunnerBackend:
    """Run create_moe_runner and return the backend it built MoeRunner with."""
    with mock.patch.object(modelopt_quant, "MoeRunner") as runner:
        method.create_moe_runner(
            SimpleNamespace(),
            MoeRunnerConfig(
                num_experts=8,
                num_local_experts=8,
                hidden_size=64,
                intermediate_size_per_partition=32,
                activation="silu",
                is_gated=True,
            ),
        )
    runner.assert_called_once()
    return runner.call_args.args[0]


def _a2a(name: str) -> SimpleNamespace:
    return SimpleNamespace(
        is_none=lambda: name == "none",
        is_deepep=lambda: name == "deepep",
        is_megamoe=lambda: False,
    )


def _create_layer(
    live: MoeRunnerBackend | None,
    intermediate_size_per_partition: int,
    is_gated: bool = True,
    explicitly_set: bool | None = None,
) -> tuple:
    """Run the real create_weights on SM120 with ``live`` as the process-wide
    backend (None: the real MoE flags) and return the method and the layer it
    built."""
    with (
        (
            _worker_gpu((12, 0))
            if live is None
            else _device((12, 0), live, explicitly_set=explicitly_set)
        ),
        mock.patch.object(
            modelopt_quant, "get_offloader", return_value=modelopt_quant.NoopOffloader()
        ),
    ):
        method = ModelOptNvFp4FusedMoEMethod(
            modelopt_quant.ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True, group_size=16
            )
        )
        layer = torch.nn.Module()
        layer.moe_runner_config = SimpleNamespace(is_gated=is_gated)
        layer.num_experts = layer.num_local_experts = 4
        method.create_weights(
            layer=layer,
            num_experts=4,
            hidden_size=64,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=torch.bfloat16,
        )
    return method, layer


class _ReachedWeightPrep(Exception):
    pass


def _process_until_weight_prep(
    method: ModelOptNvFp4FusedMoEMethod, layer: torch.nn.Module
) -> None:
    """Run process_weights_after_loading up to its backend weight prep, which
    needs FlashInfer, and stop there with _ReachedWeightPrep."""
    with (
        mock.patch.object(
            modelopt_quant, "get_moe_a2a_backend", return_value=_a2a("none")
        ),
        mock.patch.object(
            modelopt_quant, "_use_nvfp4_dispatch", side_effect=_ReachedWeightPrep
        ),
    ):
        method.process_weights_after_loading(layer)


class TestNvFp4MoeAutoBackend(CustomTestCase):
    def test_auto_on_blackwell_prepares_trtllm_weights_for_the_trtllm_runner(self):
        """W4A4 and W4A16 layers alike resolve to TRT-LLM on SM100-family GPUs."""
        for capability, is_w4a16 in itertools.product(
            ((10, 0), (10, 3)), (False, True)
        ):
            with self.subTest(capability=capability, w4a16=is_w4a16):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = _method(is_w4a16)
                    runner = _create_runner(method)
                # The weight prep flags and the runner must agree.
                self.assertTrue(method.enable_flashinfer_trtllm_moe)
                self.assertTrue(method.use_flashinfer_trtllm_weight_layout)
                self.assertTrue(runner.is_flashinfer_trtllm())

    def test_auto_on_hopper_falls_back_to_marlin(self):
        for is_w4a16 in (False, True):
            with self.subTest(w4a16=is_w4a16):
                with _device((9, 0), MoeRunnerBackend.AUTO):
                    method = _method(is_w4a16)
                    runner = _create_runner(method)
                self.assertFalse(method.enable_flashinfer_trtllm_moe)
                self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
                self.assertTrue(runner.is_marlin())

    def test_explicit_backend_is_kept(self):
        for live, is_trtllm in (
            (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, True),
            (MoeRunnerBackend.FLASHINFER_TRTLLM, True),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, False),
        ):
            with self.subTest(live=live.value):
                with _device((10, 0), live):
                    method = _method()
                    runner = _create_runner(method)
                self.assertIs(runner, live)
                self.assertEqual(method.enable_flashinfer_trtllm_moe, is_trtllm)

    def test_w4a16_layers_warn_on_fp4_runners(self):
        """A backend that quantizes activations to FP4 runs W4A16 layers with a
        warning; the runners that keep their 16-bit activations (CuTe DSL only
        off its DeepEP path) run them without one. Every explicit backend is
        kept, and W4A4 layers never warn."""
        env = envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16
        for capability in ((10, 0), (12, 0)):
            for live, cutedsl_w4a16, a2a_name, keeps in (
                (MoeRunnerBackend.FLASHINFER_CUTLASS, True, "none", False),
                (MoeRunnerBackend.FLASHINFER_TRTLLM, True, "none", False),
                (MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED, False, "none", False),
                (MoeRunnerBackend.FLASHINFER_CUTEDSL, False, "none", False),
                (MoeRunnerBackend.FLASHINFER_MEGAMOE, False, "none", False),
                (MoeRunnerBackend.FLASHINFER_CUTEDSL, True, "none", True),
                (MoeRunnerBackend.FLASHINFER_CUTEDSL, True, "one_sided", True),
                (MoeRunnerBackend.FLASHINFER_CUTEDSL, True, "deepep", False),
                (MoeRunnerBackend.FLASHINFER_MEGAMOE, True, "none", True),
                (MoeRunnerBackend.MARLIN, False, "none", True),
                (MoeRunnerBackend.LORA_MARLIN, False, "none", True),
            ):
                with (
                    self.subTest(
                        capability=capability,
                        live=live.value,
                        env=cutedsl_w4a16,
                        a2a=a2a_name,
                    ),
                    _device(capability, live),
                    env.override(cutedsl_w4a16),
                    mock.patch.object(
                        modelopt_quant,
                        "get_moe_a2a_backend",
                        return_value=_a2a(a2a_name),
                    ),
                    mock.patch.object(modelopt_quant.logger, "warning_once") as warn,
                ):
                    self.assertIs(_method().get_moe_runner_backend(None), live)
                    warn.assert_not_called()
                    self.assertIs(
                        _method(is_w4a16=True).get_moe_runner_backend(None), live
                    )
                    if keeps:
                        warn.assert_not_called()
                    else:
                        warn.assert_called_once()
                        self.assertIn(
                            f"The {live.value} MoE runner quantizes the "
                            "activations of W4A16_NVFP4",
                            warn.call_args.args[0],
                        )

    def test_auto_outside_sm100_avoids_trtllm_gen(self):
        """trtllm-gen MoE kernels are SM100-only: on SM110 and SM120 W4A4 and
        W4A16 layers alike run on FlashInfer CUTLASS, the W4A16 ones with the
        warning, and the weight prep follows the same resolution."""
        cutlass = MoeRunnerBackend.FLASHINFER_CUTLASS
        for capability, is_w4a16, tp_ep in (
            ((12, 0), False, (1, 1)),
            ((12, 1), False, (4, 4)),
            ((12, 0), False, (4, 1)),
            ((12, 1), True, (1, 1)),
            ((12, 0), True, (4, 4)),
            ((11, 0), False, (1, 1)),
            ((11, 0), True, (1, 1)),
        ):
            with self.subTest(capability=capability, w4a16=is_w4a16, tp_ep=tp_ep):
                with (
                    _device(capability, MoeRunnerBackend.AUTO, tp_ep),
                    mock.patch.object(modelopt_quant.logger, "warning_once") as warn,
                ):
                    method = _method(is_w4a16)
                    runner = _create_runner(method)
                self.assertIs(runner, cutlass)
                self.assertIs(method.get_moe_runner_backend(None), cutlass)
                self.assertFalse(method.enable_flashinfer_trtllm_moe)
                self.assertFalse(method.use_flashinfer_trtllm_weight_layout)
                self.assertTrue(method.enable_flashinfer_cutlass_moe)
                self.assertTrue(method.load_up_proj_weight_first)
                self.assertEqual(warn.called, is_w4a16)

    def test_auto_stops_at_startup_where_no_runner_fits(self):
        """At an expert parallel size flashinfer_cutlass rejects, auto has no
        runner on SM110/SM120 (Marlin runs the layers only when asked), so the
        method stops with the sizes instead of picking one silently."""
        for capability, is_w4a16 in itertools.product(
            ((12, 0), (11, 0)), (False, True)
        ):
            with self.subTest(capability=capability, w4a16=is_w4a16):
                with _device(capability, MoeRunnerBackend.AUTO, tp_ep=(4, 2)):
                    with self.assertRaisesRegex(
                        ValueError, r"1 or 4 \(the tensor parallel size\), not 2"
                    ):
                        _method(is_w4a16)

    def test_auto_reads_the_workers_gpu(self):
        """With --base-gpu-id, the worker's GPU is not device 0, for the
        support check as for the resolution."""
        with _device((12, 0), MoeRunnerBackend.AUTO, device_0_capability=(9, 0)):
            method = _method()
        self.assertIs(
            method.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_CUTLASS
        )
        with _device((12, 0), MoeRunnerBackend.AUTO, device_0_capability=(10, 0)):
            method = _method()
        self.assertIs(
            method.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_CUTLASS
        )
        with _device((10, 0), MoeRunnerBackend.AUTO, device_0_capability=(12, 0)):
            method = _method()
        self.assertIs(
            method.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_TRTLLM
        )

    def test_per_token_activation_refuses_flashinfer_cutlass(self):
        """Weight prep replaces the static activation scales with ones for
        per-token scaling, which flashinfer_cutlass never computes, so its
        layers stop before the weights are allocated."""
        layer = SimpleNamespace(moe_runner_config=SimpleNamespace(is_gated=True))
        for capability, live in (
            ((12, 0), MoeRunnerBackend.AUTO),
            ((10, 0), MoeRunnerBackend.FLASHINFER_CUTLASS),
        ):
            with self.subTest(capability=capability, live=live.value):
                with _device(capability, live):
                    method = _method(per_token=True)
                with self.assertRaisesRegex(ValueError, "per-token NVFP4"):
                    method.create_weights(
                        layer=layer,
                        num_experts=8,
                        hidden_size=64,
                        intermediate_size_per_partition=128,
                        params_dtype=None,
                    )
        with _device((10, 0), MoeRunnerBackend.AUTO):
            method = _method(per_token=True)
        self.assertIs(
            method.get_moe_runner_backend(None), MoeRunnerBackend.FLASHINFER_TRTLLM
        )
        self.assertFalse(method.enable_flashinfer_cutlass_moe)

    def test_resolver_gives_w4a16_the_w4a4_runner_with_an_a2a_backend(self):
        """Under auto the resolver gives W4A16 layers the runner W4A4 layers
        get, also with an A2A backend; whether that runner supports the A2A
        backend is MoeRunner's check, as for W4A4 layers."""
        for capability, expected in (
            ((10, 0), MoeRunnerBackend.FLASHINFER_TRTLLM),
            ((12, 0), MoeRunnerBackend.FLASHINFER_CUTLASS),
        ):
            with self.subTest(capability=capability):
                with (
                    _device(capability, MoeRunnerBackend.AUTO),
                    mock.patch.object(
                        modelopt_quant,
                        "get_moe_a2a_backend",
                        return_value=_a2a("deepep"),
                    ),
                ):
                    method = _method(is_w4a16=True)
                self.assertIs(method.get_moe_runner_backend(None), expected)

    def test_swizzled_scales_are_allocated_only_for_an_offloader(self):
        """Weight prep binds the swizzled block scales for every runner that
        reads them, so create_weights allocates none (36 MiB per Qwen3-30B-A3B
        layer), under auto or an explicit flashinfer_cutlass, unless an
        offloader registers them by name before loading."""
        for live, offloader, allocated in (
            (MoeRunnerBackend.AUTO, modelopt_quant.NoopOffloader(), False),
            (
                MoeRunnerBackend.FLASHINFER_CUTLASS,
                modelopt_quant.NoopOffloader(),
                False,
            ),
            (MoeRunnerBackend.AUTO, mock.Mock(), True),
            (MoeRunnerBackend.FLASHINFER_CUTLASS, mock.Mock(), True),
        ):
            with self.subTest(live=live.value, offloader=type(offloader).__name__):
                with (
                    _device((12, 0), live),
                    mock.patch.object(
                        modelopt_quant, "get_offloader", return_value=offloader
                    ),
                ):
                    method = ModelOptNvFp4FusedMoEMethod(
                        modelopt_quant.ModelOptFp4Config(
                            is_checkpoint_nvfp4_serialized=True, group_size=16
                        )
                    )
                    layer = torch.nn.Module()
                    layer.moe_runner_config = SimpleNamespace(is_gated=True)
                    layer.num_experts = layer.num_local_experts = 2
                    method.create_weights(
                        layer=layer,
                        num_experts=2,
                        hidden_size=64,
                        intermediate_size_per_partition=64,
                        params_dtype=torch.bfloat16,
                    )
                self.assertIs(method.enable_flashinfer_cutlass_moe, True)
                self.assertIs(layer.w13_blockscale_swizzled is not None, allocated)
                self.assertIs(layer.w2_blockscale_swizzled is not None, allocated)

    def test_runner_follows_the_init_resolution(self):
        """create_moe_runner builds the runner __init__ resolved, which the
        weight layout follows, whatever the process-wide backend reads when
        it runs."""
        with _device((12, 0), MoeRunnerBackend.AUTO):
            method = _method()
        with _device((12, 0), MoeRunnerBackend.FLASHINFER_TRTLLM):
            runner = _create_runner(method)
        self.assertIs(runner, MoeRunnerBackend.FLASHINFER_CUTLASS)

    def test_cutedsl_path_follows_the_layers_backend(self):
        """The CuTe DSL v1 (DeepEP) and v2 layouts follow the layer's resolved
        backend, not the global, which speculative decoding swaps."""
        with (
            _device((10, 0), MoeRunnerBackend.FLASHINFER_CUTEDSL),
            mock.patch.object(
                modelopt_quant, "get_moe_a2a_backend", return_value=_a2a("deepep")
            ),
        ):
            method = _method()
            with mock.patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
            ):
                self.assertTrue(method._is_cutedsl_v1_deepep)
                self.assertFalse(method._is_cutedsl_v2_standard)

    def test_online_quantization_resolves_like_a_w4a4_checkpoint(self):
        """Experts quantized at load time (a BF16 checkpoint with
        --quantization modelopt_fp4, or FP8 MTP experts converted to NVFP4)
        build the method from the online NVFP4 config, which resolves auto as a
        serialized W4A4 checkpoint does."""
        from sglang.srt.layers.quantization.nvfp4_online import (
            ModelOptNvFp4OnlineFusedMoEMethod,
        )

        config = modelopt_quant.ModelOptFp4Config.for_online_weight_quantization()
        for capability, expected in (
            ((10, 0), MoeRunnerBackend.FLASHINFER_TRTLLM),
            ((12, 0), MoeRunnerBackend.FLASHINFER_CUTLASS),
        ):
            with self.subTest(capability=capability):
                with _device(capability, MoeRunnerBackend.AUTO):
                    method = ModelOptNvFp4OnlineFusedMoEMethod(
                        config, "model.layers.0.mlp.experts"
                    )
                self.assertIs(method.get_moe_runner_backend(None), expected)

    def test_mixed_precision_marks_its_w4a16_experts(self):
        """MIXED_PRECISION hands each MoE layer the config of its own
        quant_algo, so on SM120 both run on FlashInfer CUTLASS under auto and
        only the W4A16 experts warn that their activations go to FP4."""
        from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

        config = modelopt_quant.ModelOptMixedPrecisionConfig.from_config(
            {
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "model.layers.0.mlp.experts": {
                            "quant_algo": "W4A16_NVFP4",
                            "group_size": 16,
                        },
                        "model.layers.1.mlp.experts": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        },
                    },
                }
            }
        )
        layer = object.__new__(FusedMoE)
        cutlass = MoeRunnerBackend.FLASHINFER_CUTLASS
        with (
            _device((12, 0), MoeRunnerBackend.AUTO),
            mock.patch.object(modelopt_quant.logger, "warning_once") as warn,
        ):
            nvfp4 = config.get_quant_method(layer, "model.layers.1.mlp.experts")
            warn.assert_not_called()
            w4a16 = config.get_quant_method(layer, "model.layers.0.mlp.experts")
            warn.assert_called_once()
        self.assertTrue(w4a16.quant_config.is_w4a16)
        self.assertFalse(nvfp4.quant_config.is_w4a16)
        self.assertIs(w4a16.get_moe_runner_backend(None), cutlass)
        self.assertIs(nvfp4.get_moe_runner_backend(None), cutlass)

    def test_unrequested_cutlass_refuses_gated_shards_that_need_padding(self):
        """CUTLASS weight prep pads gated rows only when 2 x I is not a
        multiple of 128 (Qwen3-30B-A3B at TP8: I=96), which it cannot do for a
        gated layer, so a CUTLASS backend the user did not pass stops in
        create_weights, whether the method resolved auto or the server args
        already carry flashinfer_cutlass (the modelopt_fp4 default on SM120).
        Other sizes, non-gated layers and an explicit flashinfer_cutlass are
        left alone."""
        cutlass = MoeRunnerBackend.FLASHINFER_CUTLASS
        for live, explicitly_set in (
            (MoeRunnerBackend.AUTO, False),
            (cutlass, False),
        ):
            with (
                self.subTest(live=live.value, explicitly_set=explicitly_set),
                self.assertRaisesRegex(
                    ValueError, r"multiple of 64, not 96\..*pending fix"
                ),
            ):
                _create_layer(
                    live,
                    intermediate_size_per_partition=96,
                    explicitly_set=explicitly_set,
                )
        for live, explicitly_set, size, is_gated in (
            (MoeRunnerBackend.AUTO, False, 128, True),
            (MoeRunnerBackend.AUTO, False, 768, True),
            (MoeRunnerBackend.AUTO, False, 96, False),
            (cutlass, False, 128, True),
            (cutlass, True, 96, True),
        ):
            with self.subTest(
                live=live.value,
                explicitly_set=explicitly_set,
                size=size,
                is_gated=is_gated,
            ):
                _, layer = _create_layer(
                    live,
                    intermediate_size_per_partition=size,
                    is_gated=is_gated,
                    explicitly_set=explicitly_set,
                )
                self.assertEqual(layer.w2_weight.shape[2], size // 2)

    def test_unrequested_cutlass_refuses_unequal_gate_up_global_scales(self):
        """flashinfer_cutlass applies one global weight scale per expert to
        both halves of w13, so on a CUTLASS backend the user did not pass
        (auto, or flashinfer_cutlass the server args resolved) a layer whose
        w1 and w3 scales differ stops before any weight is touched; equal
        scales and an explicit flashinfer_cutlass go on to weight prep."""
        cutlass = MoeRunnerBackend.FLASHINFER_CUTLASS
        for live, explicitly_set, unequal, raises in (
            (MoeRunnerBackend.AUTO, False, True, True),
            (cutlass, False, True, True),
            (MoeRunnerBackend.AUTO, False, False, False),
            (cutlass, False, False, False),
            (cutlass, True, True, False),
        ):
            with self.subTest(
                live=live.value, explicitly_set=explicitly_set, unequal=unequal
            ):
                method, layer = _create_layer(
                    live,
                    intermediate_size_per_partition=128,
                    explicitly_set=explicitly_set,
                )
                for param in layer.parameters():
                    param.data.copy_(torch.arange(param.numel()).view(param.shape))
                layer.w13_weight_scale_2.data.fill_(0.5)
                if unequal:
                    layer.w13_weight_scale_2.data[2, 1] = 0.25
                before = {
                    name: param.detach().clone()
                    for name, param in layer.named_parameters()
                }
                if raises:
                    with self.assertRaisesRegex(
                        ValueError, r"^1 of 4 local experts have different w1 and w3"
                    ):
                        _process_until_weight_prep(method, layer)
                    after = dict(layer.named_parameters())
                    self.assertEqual(after.keys(), before.keys())
                    for name, value in before.items():
                        self.assertTrue(torch.equal(after[name], value), name)
                else:
                    with self.assertRaises(_ReachedWeightPrep):
                        _process_until_weight_prep(method, layer)

    def test_draft_context_carries_the_drafts_explicitness(self):
        """The draft is built under speculative_moe_backend_context, so a CUTLASS
        draft the user did not pass (auto) is checked even when the target's
        flashinfer_cutlass was explicit, an explicit CUTLASS draft is not when
        the target's was auto, and layers built after the context get the
        target's answer back."""
        moe = get_flags().moe
        saved = msgspec.structs.asdict(moe)
        self.addCleanup(lambda: [setattr(moe, k, v) for k, v in saved.items()])
        auto, cutlass = MoeRunnerBackend.AUTO, MoeRunnerBackend.FLASHINFER_CUTLASS
        for target, target_explicit, draft, draft_explicit in (
            (cutlass, True, auto, False),
            (auto, False, cutlass, True),
        ):
            with self.subTest(target=target.value, draft=draft.value):
                moe.runner_backend = target
                moe.runner_backend_explicit = target_explicit
                moe.speculative_runner_backend = draft
                moe.speculative_runner_backend_explicit = draft_explicit
                for in_draft, checked in (
                    (True, not draft_explicit),
                    (False, not target_explicit),
                ):
                    scope = (
                        speculative_moe_backend_context()
                        if in_draft
                        else contextlib.nullcontext()
                    )
                    with self.subTest(in_draft=in_draft), scope:
                        if checked:
                            with self.assertRaisesRegex(ValueError, "not 96"):
                                _create_layer(None, intermediate_size_per_partition=96)
                        else:
                            _create_layer(None, intermediate_size_per_partition=96)
                        method, layer = _create_layer(
                            None, intermediate_size_per_partition=128
                        )
                        layer.w13_weight_scale_2.data.copy_(
                            torch.tensor([[0.5, 0.25]] * 4)
                        )
                        if checked:
                            with self.assertRaisesRegex(
                                ValueError, "4 of 4 local experts"
                            ):
                                _process_until_weight_prep(method, layer)
                        else:
                            with self.assertRaises(_ReachedWeightPrep):
                                _process_until_weight_prep(method, layer)

    def test_standalone_draft_counts_only_its_own_runner_as_explicit(self):
        """A STANDALONE draft is built with the target's runner, outside
        speculative_moe_backend_context. Its NVFP4 layers must not take the
        target's explicit flashinfer_cutlass as the draft's choice: with the draft
        on auto they get both checks, and a draft runner passed as
        flashinfer_cutlass still keeps main's behaviour. A draft runner passed as
        another backend, or inherited without being passed, does not count."""
        from sglang.srt.layers.moe import utils as moe_utils
        from sglang.srt.speculative import standalone_worker_v2

        moe = get_flags().moe
        saved = msgspec.structs.asdict(moe)
        self.addCleanup(lambda: [setattr(moe, k, v) for k, v in saved.items()])
        cutlass, auto = MoeRunnerBackend.FLASHINFER_CUTLASS, MoeRunnerBackend.AUTO
        triton = MoeRunnerBackend.TRITON

        class _DraftBuilt(Exception):
            pass

        def build_draft_model(**_):
            try:
                _create_layer(None, intermediate_size_per_partition=96)
            except ValueError as e:
                draft_layer_errors.append(str(e))
            raise _DraftBuilt

        module = standalone_worker_v2
        spec = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_steps=1,
            speculative_num_draft_tokens=2,
            speculative_algorithm="STANDALONE",
            speculative_boundary_reduction=None,
            speculative_enable_w4a4_mxfp4_megamoe=None,
        )
        exec_bag = SimpleNamespace(
            comm=SimpleNamespace(override=lambda **_: contextlib.nullcontext())
        )
        for target_explicit, draft, draft_explicit, draft_checked in (
            (True, auto, False, True),
            (False, cutlass, True, False),
            # The draft is built on the target's flashinfer_cutlass, not triton.
            (True, triton, True, True),
            # Inherited from a target default, not passed.
            (False, cutlass, False, True),
        ):
            with self.subTest(
                target_explicit=target_explicit,
                draft=draft.value,
                draft_explicit=draft_explicit,
            ):
                moe.runner_backend = cutlass
                moe.runner_backend_explicit = target_explicit
                moe.speculative_runner_backend = draft
                moe.speculative_runner_backend_explicit = draft_explicit
                draft_layer_errors = []
                with (
                    mock.patch.object(
                        module, "get_device", return_value=SimpleNamespace(device="cpu")
                    ),
                    mock.patch.object(module, "get_spec", return_value=spec),
                    mock.patch.object(moe_utils, "get_spec", return_value=spec),
                    mock.patch.object(moe_utils, "get_exec", return_value=exec_bag),
                    mock.patch.object(module, "resolve_draft_parallel_layout"),
                    mock.patch.object(
                        module, "draft_tp_context", contextlib.nullcontext
                    ),
                    mock.patch.object(
                        module, "draft_pp_context", contextlib.nullcontext
                    ),
                    mock.patch.object(
                        module.StandaloneDraftWorker, "_rebuild_topk1_chain_buffers"
                    ),
                    mock.patch.object(
                        module, "TpModelWorker", side_effect=build_draft_model
                    ),
                    self.assertRaises(_DraftBuilt),
                ):
                    module.StandaloneDraftWorker(
                        server_args=None,
                        gpu_id=0,
                        nccl_port=0,
                        target_worker=SimpleNamespace(
                            model_runner=SimpleNamespace(
                                model_config=SimpleNamespace(context_len=128)
                            )
                        ),
                    )
                self.assertEqual(len(draft_layer_errors), int(draft_checked))
                if draft_layer_errors:
                    self.assertIn("not 96", draft_layer_errors[0])
                # The target's own layers keep the target's answer.
                if target_explicit:
                    _create_layer(None, intermediate_size_per_partition=96)
                else:
                    with self.assertRaisesRegex(ValueError, "not 96"):
                        _create_layer(None, intermediate_size_per_partition=96)


if __name__ == "__main__":
    unittest.main()
