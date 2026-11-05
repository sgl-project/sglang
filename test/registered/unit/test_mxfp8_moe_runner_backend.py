"""Unit tests for the mxfp8 moe_runner_backend constraints in
srt/arg_groups/overrides.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.model_override_base import ResolvedView
from sglang.srt.arg_groups.overrides import _moe_runner_backend_quant_constraints
from sglang.srt.runtime_context import override_platform
from sglang.test.test_utils import CustomTestCase


@contextlib.contextmanager
def _platform(*, is_hip: bool, is_gfx95: bool = False, arch: str = "none"):
    """gfx95 is read straight from utils.common, not through the platform
    probes, so it is patched where overrides.py imported it.

    Every case pins the Blackwell probes rather than letting them answer from
    the runner's machine: `override_platform` only replaces the facts it is
    given, so an unpinned case would resolve flashinfer_trtllm on a CPU runner
    and be refused on a consumer-Blackwell host.
    """
    facts = {"is_hip": is_hip, "is_npu": False, "is_cuda": not is_hip}
    facts.update(_ARCHES[arch])
    with override_platform(**facts):
        with patch(
            "sglang.srt.arg_groups.overrides.is_gfx95_supported",
            return_value=is_gfx95,
        ):
            yield


# The platform answers this pass reads. device_sm is major*10+minor, the same
# number nvidia-smi prints as a compute capability (12.0 -> 120). "none" is any
# device outside the Blackwell families, which is what ROCm answers too.
_ARCHES = {
    "none": dict(is_sm100=False, is_sm110=False, is_sm120=False, device_sm=0),
    "sm90": dict(is_sm100=False, is_sm110=False, is_sm120=False, device_sm=90),
    "sm100": dict(is_sm100=True, is_sm110=False, is_sm120=False, device_sm=100),
    "sm103": dict(is_sm100=True, is_sm110=False, is_sm120=False, device_sm=103),
    "sm107": dict(is_sm100=True, is_sm110=False, is_sm120=False, device_sm=107),
    "sm110": dict(is_sm100=False, is_sm110=True, is_sm120=False, device_sm=110),
    "sm120": dict(is_sm100=False, is_sm110=False, is_sm120=True, device_sm=120),
    "sm121": dict(is_sm100=False, is_sm110=False, is_sm120=True, device_sm=121),
}


def _resolve(backend: str, quantization: str = "mxfp8") -> str:
    """Run the pass and return the backend it resolves to. The pass declares
    only what it changes, so an empty declaration means the request survived.

    moe_a2a_backend is "none" throughout: the pass reads it to pick the default
    (flashinfer_megamoe a2a forces a matching runner), and that interaction is
    out of scope here.
    """
    view = ResolvedView(
        SimpleNamespace(
            moe_runner_backend=backend,
            quantization=quantization,
            moe_a2a_backend="none",
        )
    )
    declared = _moe_runner_backend_quant_constraints(view)
    return declared.get("moe_runner_backend", backend)


class TestMxfp8MoeRunnerBackend(CustomTestCase):
    def test_rocm_honors_explicit_triton(self):
        # triton is the only mxfp8 runner ROCm can use: every other entry in
        # MXFP8_MOE_RUNNER_BACKEND_CHOICES is CUDA-only, and flashinfer_trtllm's
        # MoE apply path imports flashinfer, which is absent on ROCm.
        for is_gfx95 in (True, False):
            with self.subTest(is_gfx95=is_gfx95):
                with _platform(is_hip=True, is_gfx95=is_gfx95):
                    self.assertEqual(_resolve("triton"), "triton")

    def test_cuda_still_rejects_triton(self):
        with _platform(is_hip=False, arch="sm100"):
            self.assertEqual(_resolve("triton"), "flashinfer_trtllm")

    def test_auto_default_is_unchanged(self):
        # The fix widens `allowed`, not the default: "auto" must resolve exactly
        # as it did before on every platform.
        cases = (
            (dict(is_hip=True, is_gfx95=True), "triton"),
            (dict(is_hip=True, is_gfx95=False), "flashinfer_trtllm"),
            (dict(is_hip=False, is_gfx95=False, arch="sm100"), "flashinfer_trtllm"),
        )
        for facts, expected in cases:
            with self.subTest(**facts):
                with _platform(**facts):
                    self.assertEqual(_resolve("auto"), expected)

    def test_unsupported_backend_is_still_overridden(self):
        # The widened `allowed` must not turn the gate into a pass-through: a
        # name outside MXFP8_MOE_RUNNER_BACKEND_CHOICES still falls to the
        # platform default.
        with _platform(is_hip=True, is_gfx95=False):
            self.assertEqual(_resolve("marlin"), "flashinfer_trtllm")

    def test_non_mxfp8_quantization_is_untouched(self):
        with _platform(is_hip=True, is_gfx95=False):
            self.assertEqual(_resolve("triton", quantization="fp8"), "triton")


class TestMxfp8TrtllmDeviceConstraint(CustomTestCase):
    """flashinfer's trtllm-gen MoE dispatches no cubin on consumer Blackwell, so on
    SM110/SM120/SM121 the mxfp8 default must be refused at startup rather than
    resolving to a backend that fails mid-forward."""

    _REFUSAL = "no config matching the device"

    def test_refused_on_devices_without_trtllm_moe_cubins(self):
        for arch in ("sm110", "sm120", "sm121"):
            for backend in ("auto", "flashinfer_trtllm", "flashinfer_trtllm_routed"):
                with self.subTest(arch=arch, backend=backend):
                    with _platform(is_hip=False, arch=arch):
                        with self.assertRaisesRegex(ValueError, self._REFUSAL):
                            _resolve(backend)

    def test_accepted_on_the_sm100_family(self):
        for arch in ("sm100", "sm103", "sm107"):
            with self.subTest(arch=arch):
                with _platform(is_hip=False, arch=arch):
                    self.assertEqual(_resolve("auto"), "flashinfer_trtllm")
                    self.assertEqual(
                        _resolve("flashinfer_trtllm_routed"),
                        "flashinfer_trtllm_routed",
                    )

    def test_names_outside_the_choices_are_refused_after_the_override(self):
        # The device check reads the backend this pass resolved to, not the name the
        # request carried, so an off-list name takes the existing device-free warning
        # first and lands in the same refusal rather than being warned into a runner
        # that has no cubins on the device. flashinfer_cutlass is the name worth
        # pinning: it is what this pass resolves modelopt_fp4 to on SM120, but it is
        # not in MXFP8_MOE_RUNNER_BACKEND_CHOICES, so for mxfp8 it is off-list too.
        for arch in ("sm110", "sm120", "sm121"):
            for backend in ("triton", "marlin", "flashinfer_cutlass"):
                with self.subTest(arch=arch, backend=backend):
                    with _platform(is_hip=False, arch=arch):
                        with self.assertRaisesRegex(ValueError, self._REFUSAL):
                            _resolve(backend)

        # Off consumer Blackwell the override keeps answering as it did.
        for backend in ("triton", "marlin", "flashinfer_cutlass"):
            with self.subTest(arch="sm90", backend=backend):
                with _platform(is_hip=False, arch="sm90"):
                    self.assertEqual(_resolve(backend), "flashinfer_trtllm")

    def test_accepted_where_the_default_is_not_trtllm(self):
        # Hopper answers the same question the way it did before: this arm has no
        # trtllm-device check, so only a request naming a trtllm backend is in
        # range, and neither is refused on a device this pass never defaults to
        # trtllm for.
        for backend in ("auto", "flashinfer_trtllm", "cutlass", "deep_gemm"):
            with self.subTest(arch="sm90", backend=backend):
                with _platform(is_hip=False, arch="sm90"):
                    _resolve(backend)

    def test_non_trtllm_backends_keep_working_on_consumer_blackwell(self):
        # The gate fires on the resolved backend, not on the device: asking for a
        # runner that is not trtllm-gen must stay untouched on SM120.
        for backend in ("cutlass", "deep_gemm"):
            with self.subTest(backend=backend):
                with _platform(is_hip=False, arch="sm120"):
                    self.assertEqual(_resolve(backend), backend)

    def test_megamoe_default_is_untouched_on_consumer_blackwell(self):
        # moe_a2a_backend == flashinfer_megamoe selects a different default
        # entirely, and that path stays accepted on SM120.
        view = ResolvedView(
            SimpleNamespace(
                moe_runner_backend="auto",
                quantization="mxfp8",
                moe_a2a_backend="flashinfer_megamoe",
            )
        )
        with _platform(is_hip=False, arch="sm120"):
            self.assertEqual(
                _moe_runner_backend_quant_constraints(view),
                {"moe_runner_backend": "flashinfer_megamoe"},
            )

    def test_rocm_is_untouched(self):
        # gfx95 resolves to triton and non-gfx95 ROCm still lands on
        # flashinfer_trtllm, which this gate must not touch: it keys on the CUDA
        # Blackwell probes, and ROCm answers neither.
        with _platform(is_hip=True, is_gfx95=True):
            self.assertEqual(_resolve("auto"), "triton")
        with _platform(is_hip=True, is_gfx95=False):
            self.assertEqual(_resolve("auto"), "flashinfer_trtllm")

    def test_other_quantizations_are_untouched(self):
        with _platform(is_hip=False, arch="sm120"):
            self.assertEqual(
                _resolve("flashinfer_trtllm", quantization="fp8"), "flashinfer_trtllm"
            )


if __name__ == "__main__":
    unittest.main()
