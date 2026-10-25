"""CPU checks of the platform seams of the megakernel binding: --enable-megakernel,
the device target, a platform-owned dispatch and a built pool_layout."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.plugin_kernels.nvidia import spec as registry
from sglang.srt.kernel_plugins.nvidia import binding
from sglang.srt.platforms.interface import SRTPlatform
from sglang.srt.runtime_context import publish, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fakes import ARCH, SPEC, FakeBuilt, Lm, batch, server_args  # noqa: E402

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

TARGET = "fake_accel_v1"


class OotPlatform:
    """An out-of-tree platform: no torch.cuda, its own target and graph runner."""

    activate_megakernel_pool_layout = SRTPlatform.activate_megakernel_pool_layout

    def __init__(self, *, owns_dispatch=False, accepts_layouts=False):
        self.owns_dispatch, self.accepts_layouts = owns_dispatch, accepts_layouts
        self.dispatched, self.layouts = [], []

    def is_cuda_alike(self):
        return False

    def is_out_of_tree(self):
        return True

    def get_megakernel_target(self):
        return TARGET

    def install_megakernel_dispatch(self, *, model, built, binding):
        self.dispatched.append(built)
        return self.owns_dispatch

    def accept(self, layout):
        self.layouts.append(layout)


class LayoutBuilt(FakeBuilt):
    pool_layout = "kernel-native"


SPEC_OOT = registry.MegakernelSpec(
    name="fake-oot",
    requires=registry.Requirements(
        isas=frozenset({TARGET}),
        tp_sizes=frozenset(),
        decode_graph_backends=frozenset(),
    ),
    features=SPEC.features,
    numerics=SPEC.numerics,
    max_rows=8,
    build=lambda model, hf, ctx: FakeBuilt(ctx.widths),
    api_version=registry.MEGAKERNEL_API_VERSION,
)
SPEC_LAYOUT = registry.MegakernelSpec(
    **{
        **{f: getattr(SPEC_OOT, f) for f in SPEC_OOT.__struct_fields__},
        "build": lambda model, hf, ctx: LayoutBuilt(ctx.widths),
    }
)


def install(platform, *, hooks=None):
    lm = Lm()
    with (
        patch.object(binding, "current_platform", platform),
        patch.object(binding, "deadline_s", return_value=30.0),
        patch.object(binding, "_gather_tp", side_effect=lambda x, group=None: [x]),
        patch.object(binding, "_tokens_per_request", return_value=1),
        patch.object(binding, "_decode_graphs_enabled", return_value=True),
    ):
        binding.install_megakernel(
            model=lm,
            hf_config=SimpleNamespace(architectures=[ARCH]),
            capture_tail_hooks=[] if hooks is None else hooks,
            is_draft=False,
        )
    return lm


class TestMegakernelPlatformHooks(CustomTestCase):
    def setUp(self):
        registry_patch = patch.dict(registry._REGISTRY, clear=True)
        registry_patch.start()
        self.addCleanup(registry_patch.stop)
        self.addCleanup(reset_context)

    def _publish(self, **flags):
        publish(server_args(**flags), role="test")

    def test_enable_megakernel_is_the_platform_agnostic_spelling(self):
        self._publish()
        self.assertFalse(binding.megakernel_enabled())
        reset_context()
        self._publish(enable_megakernel=True)
        self.assertTrue(binding.megakernel_enabled())
        reset_context()
        self._publish(enable_nvidia_megakernel=True)
        self.assertTrue(binding.megakernel_enabled())

    def test_specs_match_the_platform_target(self):
        registry.register(architecture=ARCH, path=f"{__name__}:SPEC_OOT")
        with patch.object(binding, "current_platform", OotPlatform()):
            self.assertEqual(binding.device_isa(), TARGET)
        spec, why = registry.find_spec(architectures=[ARCH], isa=TARGET)
        self.assertIs(spec, SPEC_OOT, why)
        spec, why = registry.find_spec(architectures=[ARCH], isa="sm_103")
        self.assertIsNone(spec)
        self.assertIn("not sm_103", why)

    def test_a_platform_that_owns_dispatch_keeps_the_model_forward(self):
        registry.register(architecture=ARCH, path=f"{__name__}:SPEC_OOT")
        self._publish(enable_megakernel=True)
        platform, hooks = OotPlatform(owns_dispatch=True), []
        lm = install(platform, hooks=hooks)
        built = binding.built_megakernel(lm)
        self.assertEqual(platform.dispatched, [built])
        self.assertNotIn("forward", lm.__dict__)
        self.assertEqual(hooks, [], "no CUDA-graph capture check")
        self.assertEqual(lm(batch(2).input_ids, None, batch(2)), "standard")

        # The default (False) keeps core's model.forward wrapper.
        lm = install(OotPlatform(owns_dispatch=False))
        self.assertIn("forward", lm.__dict__)

    def test_a_pool_layout_goes_to_the_platform_before_dispatch(self):
        registry.register(architecture=ARCH, path=f"{__name__}:SPEC_LAYOUT")
        self._publish(enable_megakernel=True)
        platform = OotPlatform(owns_dispatch=True)
        platform.activate_megakernel_pool_layout = platform.accept
        install(platform)
        self.assertEqual(platform.layouts, ["kernel-native"])
        # SRTPlatform's default refuses a layout its pools cannot build.
        with self.assertRaisesRegex(ValueError, "declares a pool_layout"):
            install(OotPlatform(owns_dispatch=True))


if __name__ == "__main__":
    unittest.main()
