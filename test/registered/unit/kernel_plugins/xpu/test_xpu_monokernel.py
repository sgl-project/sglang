"""CPU contracts of --enable-xpu-monokernel: the flag selects the XPU backend of the
shared monokernel binding, the device checks read the Intel GPU's properties, and
the startup checks refuse what an XPU spec does not cover."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.plugin_kernels.amd import spec as registry
from sglang.kernels.plugin_kernels.xpu import spec as xpu_spec
from sglang.kernels.plugin_kernels.xpu.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
)
from sglang.srt.kernel_plugins.amd import binding
from sglang.srt.kernel_plugins.amd.binding import (
    covered_capture_sizes,
    monokernel_disable_reasons,
    selected_monokernel,
)
from sglang.srt.kernel_plugins.nvidia.binding import handle_nvidia_megakernel
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

ARCH = "FakeXpuMonoForCausalLM"
BACKEND = xpu_spec.BACKEND


class FakeBuilt:
    def forward(self, plan, input_ids, positions, forward_batch, input_embeds=None):
        return "mono"


SPEC = MonoSpec(
    name="fake_xpu",
    devices={"xe2": DeviceLimits(widths=(1, 2, 4, 8), tp_sizes=(2,), resident_blocks=32)},
    features=frozenset(),
    requires=Requirements(
        dtypes=frozenset({"bfloat16", "float16"}),
        attention_backends=frozenset({"intel_xpu"}),
        server_args={"page_size": frozenset({64, 128})},
    ),
    numerics=Numerics(kind=NumericsKind.TOLERANCE, bounds={"layer.hidden.cosine": 0.999}),
    model_disable_reasons=lambda model, hf, copies: [],
    build=lambda model, hf, ctx: FakeBuilt(),
    api_version=registry.MONOKERNEL_API_VERSION,
)


def server_args(**kw):
    defaults = {
        "model_path": "dummy",
        "tp_size": 2,
        "page_size": 64,
        "attention_backend": "intel_xpu",
        "enable_xpu_monokernel": True,
    }
    return ServerArgs(**{**defaults, **kw})


def reasons_for(cfg, isa="xe2", dtype="bfloat16"):
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[ARCH], model_type="fake"), dtype=dtype
    )
    return monokernel_disable_reasons(cfg=cfg, backend=BACKEND, model_config=model_config, isa=isa)


def xpu_props(arch, xe_cores=32):
    return SimpleNamespace(architecture=arch, gpu_subslice_count=xe_cores)


class TestXpuMonokernel(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        registry.register(backend=BACKEND, architecture=ARCH, path=f"{__name__}:SPEC")

    def test_the_flag_selects_the_xpu_backend(self):
        self.assertEqual(selected_monokernel(server_args()), (BACKEND, "--enable-xpu-monokernel"))
        self.assertIsNone(selected_monokernel(ServerArgs(model_path="dummy")))
        publish(server_args(), role="test")
        self.addCleanup(reset_context)
        self.assertEqual(binding._backend(), BACKEND)

    def test_the_deprecated_backend_choice_maps_to_the_flag(self):
        args = prepare_server_args(["--model-path", "dummy", "--monokernel-backend", "xpu"])
        self.assertTrue(args.enable_xpu_monokernel)
        self.assertFalse(args.enable_amd_monokernel)

    def test_the_isa_of_an_intel_gpu(self):
        for arch, isa in ((0x500800000, "xe2"), (0x500400400, "xe2"), (0x780000400, "xe3")):
            with self.subTest(arch=hex(arch)):
                with (
                    patch.object(binding, "_device_module", return_value=torch.xpu),
                    patch.object(torch.xpu, "current_device", return_value=0),
                    patch.object(torch.xpu, "get_device_properties", return_value=xpu_props(arch)),
                ):
                    self.assertEqual(binding.device_isa(), isa)
        self.assertEqual(binding._compute_units(xpu_props(0x500800000, 20)), 20)
        self.assertEqual(binding._compute_units(SimpleNamespace(multi_processor_count=132)), 132)

    def test_a_supported_setup_has_no_reasons(self):
        spec, reasons = reasons_for(server_args())
        self.assertIs(spec, SPEC)
        self.assertEqual(reasons, [])

    def test_every_refusal_reason_is_listed(self):
        spec, reasons = reasons_for(server_args(tp_size=4, page_size=16, pp_size=2), dtype="float32")
        self.assertIs(spec, SPEC)
        for want in ("tp_size 4", "page_size 16", "dtype 'float32'", Feature.PIPELINE_PARALLEL.value):
            self.assertTrue(any(want in r for r in reasons), (want, reasons))
        spec, reasons = reasons_for(server_args(), isa="xe3")
        self.assertIsNone(spec)
        self.assertIn("has no spec", reasons[0])

    def test_capture_sizes_align_to_the_widths(self):
        self.assertEqual(
            covered_capture_sizes(widths=(1, 2, 4, 8), tokens_per_request=1, sizes=[1, 2, 4, 8, 12, 16, 24]),
            [1, 2, 4, 8, 12, 16, 24],
        )
        self.assertEqual(
            covered_capture_sizes(widths=(1, 2, 4, 8), tokens_per_request=1, sizes=[1, 3, 6, 16]),
            [1, 2, 4, 8, 16],
        )

    def test_one_full_forward_replacement_at_a_time(self):
        cfg = server_args(enable_amd_monokernel=True)
        with patch.object(binding, "resolving_view", return_value=cfg):
            with self.assertRaisesRegex(ValueError, "--enable-amd-monokernel is also set"):
                binding.handle_monokernel(cfg)
        cfg = server_args(enable_nvidia_megakernel=True)
        with (
            patch("sglang.srt.kernel_plugins.nvidia.binding.resolving_view", return_value=cfg),
            patch("sglang.srt.kernel_plugins.nvidia.binding.model_config_of"),
            patch("sglang.srt.kernel_plugins.nvidia.binding.msa_serves", return_value=False),
            patch("sglang.srt.kernel_plugins.nvidia.binding.plugin_reasons", return_value=["x"]),
        ):
            with self.assertRaisesRegex(ValueError, "--enable-xpu-monokernel is also set"):
                handle_nvidia_megakernel(cfg)


if __name__ == "__main__":
    unittest.main()
