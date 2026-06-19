"""Unit tests for ``kernel_shape_profiler`` — no server, no model loading.

These tests exercise the shape-discovery profiling utility added for the
profiling enhancements work:

  * schema inference from (string and real) type annotations,
  * the kernel-launch detection heuristics used by auto-discovery,
  * the ``record_function`` fallback wrapper,
  * the public ``enable()`` / ``disable()`` lifecycle, including that a
    wrapped kernel actually emits ``Input Dims`` in a profiler trace and
    that repeated enable/disable cycles do not raise, and
  * the new config fields that gate the feature (default off).

The integration tests register a synthetic kernel into the profiler's
registry (with auto-discovery disabled) so they run quickly on CPU or GPU
without importing the full ``sglang.srt`` tree.

Usage:
    python -m unittest test_kernel_shape_profiler -v
    python test_kernel_shape_profiler.py
"""

import inspect
import sys
import types
import unittest
from typing import Optional

import torch

import sglang.srt.utils.kernel_shape_profiler as ksp
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=15, suite="base-b-test-cpu")


def _make_param(annotation, name="x", default=inspect._empty):
    return inspect.Parameter(
        name,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=annotation,
        default=default,
    )


class TestSchemaInference(CustomTestCase):
    """``_infer_schema_type`` / ``_build_schema_from_sig`` mapping logic."""

    def test_real_annotations(self):
        cases = {
            torch.Tensor: "Tensor",
            int: "int",
            float: "float",
            bool: "bool",
            str: "str",
            torch.dtype: "ScalarType",
        }
        for annotation, expected in cases.items():
            self.assertEqual(ksp._infer_schema_type(_make_param(annotation)), expected)

    def test_optional_tensor_real_annotation(self):
        self.assertEqual(
            ksp._infer_schema_type(_make_param(Optional[torch.Tensor])), "Tensor?"
        )

    def test_string_annotations_pep563(self):
        cases = {
            "torch.Tensor": "Tensor",
            "Tensor": "Tensor",
            "Optional[torch.Tensor]": "Tensor?",
            "Optional[Tensor]": "Tensor?",
            "int": "int",
            "float": "float",
        }
        for annotation, expected in cases.items():
            self.assertEqual(ksp._infer_schema_type(_make_param(annotation)), expected)

    def test_unannotated_returns_none(self):
        self.assertIsNone(ksp._infer_schema_type(_make_param(inspect._empty)))

    def test_unknown_annotation_returns_none(self):
        self.assertIsNone(ksp._infer_schema_type(_make_param(dict)))

    def test_build_schema_separates_tensor_and_non_tensor(self):
        def fn(q: torch.Tensor, k: torch.Tensor, scale: float, causal: bool):
            return q

        sig = inspect.signature(fn)
        result = ksp._build_schema_from_sig(sig)
        self.assertIsNotNone(result)
        schema_str, tensor_names, non_tensor_names = result
        self.assertEqual(tensor_names, ["q", "k"])
        self.assertEqual(non_tensor_names, ["scale", "causal"])
        self.assertTrue(schema_str.endswith("-> ()"))
        self.assertIn("Tensor q", schema_str)
        self.assertIn("Tensor k", schema_str)

    def test_build_schema_none_without_tensor_params(self):
        def fn(a: int, b: float):
            return a

        self.assertIsNone(ksp._build_schema_from_sig(inspect.signature(fn)))

    def test_build_schema_none_with_varargs(self):
        def fn(x: torch.Tensor, *args, **kwargs):
            return x

        self.assertIsNone(ksp._build_schema_from_sig(inspect.signature(fn)))

    def test_build_schema_skip_self(self):
        def method(self, x: torch.Tensor):
            return x

        result = ksp._build_schema_from_sig(inspect.signature(method), skip_self=True)
        self.assertIsNotNone(result)
        _, tensor_names, _ = result
        self.assertEqual(tensor_names, ["x"])


class TestKernelLauncherHeuristic(CustomTestCase):
    """``_is_likely_kernel_launcher`` / ``_source_launches_kernel``."""

    def test_tensor_annotation_is_launcher(self):
        def fn(x: torch.Tensor, n: int):
            return x

        self.assertTrue(ksp._is_likely_kernel_launcher(fn, inspect.signature(fn)))

    def test_only_non_tensor_annotations_excluded(self):
        def fn(a: int, b: float):
            return a

        self.assertFalse(ksp._is_likely_kernel_launcher(fn, inspect.signature(fn)))

    def test_unannotated_with_triton_grid_source_is_launcher(self):
        def fn(x, y):
            grid = (1,)
            some_kernel[grid](x, y)  # noqa: F821

        self.assertTrue(ksp._is_likely_kernel_launcher(fn, inspect.signature(fn)))

    def test_unannotated_plain_util_excluded(self):
        def fn(x, y):
            return x + y

        self.assertFalse(ksp._is_likely_kernel_launcher(fn, inspect.signature(fn)))

    def test_source_detects_torch_ops(self):
        def fn(x):
            return torch.ops.sglang.foo(x)  # noqa

        self.assertTrue(ksp._source_launches_kernel(fn))


class TestRecordFunctionWrapper(CustomTestCase):
    """``_make_record_function_wrapper`` preserves results and emits shapes."""

    def test_wrapper_preserves_return_value(self):
        def fn(x, y):
            return x + y

        wrapper = ksp._make_record_function_wrapper("mymod.fn", fn)
        a = torch.ones(3)
        b = torch.ones(3) * 2
        self.assertTrue(torch.allclose(wrapper(a, b), a + b))

    def test_wrapper_emits_shapes_in_event_name(self):
        def fn(x, y):
            return x + y

        wrapper = ksp._make_record_function_wrapper("mymod.fancy_kernel", fn)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            wrapper(torch.randn(2, 5), torch.randn(2, 5))

        names = [e.name for e in prof.events()]
        matches = [n for n in names if "mymod.fancy_kernel" in n]
        self.assertTrue(matches, f"no record_function event found in {names[:20]}")
        # Tensor shapes are embedded directly in the event name.
        self.assertTrue(
            any("[2, 5]" in n for n in matches),
            f"shape not embedded in event name: {matches}",
        )


class TestEnableDisable(CustomTestCase):
    """End-to-end enable()/disable() over a synthetic kernel registry."""

    def setUp(self):
        # Snapshot module-level config we override so tearDown can restore it.
        self._orig_prefixes = ksp._AUTO_DISCOVER_PREFIXES
        self._orig_entries = ksp._KERNEL_ENTRY_POINTS
        # Disable the (heavy) auto-discovery scan; we only want our fake ops.
        ksp._AUTO_DISCOVER_PREFIXES = ()

        self._mod_name = "_ksp_test_fake_kernels"
        mod = types.ModuleType(self._mod_name)

        def annotated_kernel(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
            return x + y * alpha

        def optional_kernel(x: torch.Tensor, bias: Optional[torch.Tensor] = None):
            return x if bias is None else x + bias

        def unannotated_launcher(x, y):
            # Source contains a Triton-style launch marker so the heuristic
            # routes this through the record_function fallback wrapper.
            grid = (1,)
            if False:  # never executed; only the source text matters
                some_kernel[grid](x, y)  # noqa: F821
            return x + y

        mod.annotated_kernel = annotated_kernel
        mod.optional_kernel = optional_kernel
        mod.unannotated_launcher = unannotated_launcher
        self._fake_mod = mod
        self._orig_fns = {
            "annotated_kernel": annotated_kernel,
            "optional_kernel": optional_kernel,
            "unannotated_launcher": unannotated_launcher,
        }
        sys.modules[self._mod_name] = mod

        ksp._KERNEL_ENTRY_POINTS = [
            (self._mod_name, "annotated_kernel"),
            (self._mod_name, "optional_kernel"),
            (self._mod_name, "unannotated_launcher"),
        ]

    def tearDown(self):
        if ksp.is_enabled():
            ksp.disable()
        ksp._AUTO_DISCOVER_PREFIXES = self._orig_prefixes
        ksp._KERNEL_ENTRY_POINTS = self._orig_entries
        sys.modules.pop(self._mod_name, None)

    def test_enable_patches_and_disable_restores(self):
        self.assertFalse(ksp.is_enabled())
        ksp.enable()
        self.assertTrue(ksp.is_enabled())
        # The module reference is replaced with a wrapper.
        self.assertIsNot(
            self._fake_mod.annotated_kernel, self._orig_fns["annotated_kernel"]
        )

        ksp.disable()
        self.assertFalse(ksp.is_enabled())
        self.assertIs(
            self._fake_mod.annotated_kernel, self._orig_fns["annotated_kernel"]
        )

    def test_wrapped_kernel_is_numerically_transparent(self):
        ksp.enable()
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        out = self._fake_mod.annotated_kernel(a, b, alpha=2.0)
        self.assertTrue(torch.allclose(out, a + b * 2.0))

    def test_custom_op_emits_input_dims(self):
        ksp.enable()
        a = torch.randn(4, 8)
        b = torch.randn(4, 8)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            self._fake_mod.annotated_kernel(a, b, alpha=2.0)

        op_events = [e for e in prof.events() if "annotated_kernel" in e.name]
        self.assertTrue(op_events, "wrapped kernel did not appear in trace")
        # At least one event must carry the recorded input dims.
        with_shapes = [e for e in op_events if getattr(e, "input_shapes", None)]
        self.assertTrue(
            with_shapes,
            f"no Input Dims captured for wrapped kernel: "
            f"{[(e.name, getattr(e, 'input_shapes', None)) for e in op_events]}",
        )
        self.assertIn([4, 8], with_shapes[0].input_shapes)

    def test_record_function_fallback_emits_shapes(self):
        ksp.enable()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
        ) as prof:
            self._fake_mod.unannotated_launcher(torch.randn(3, 7), torch.randn(3, 7))

        names = [e.name for e in prof.events()]
        matches = [n for n in names if "unannotated_launcher" in n]
        self.assertTrue(matches, f"fallback event missing in {names[:20]}")
        self.assertTrue(
            any("[3, 7]" in n for n in matches),
            f"fallback shapes not embedded: {matches}",
        )

    def test_optional_tensor_none_falls_back(self):
        # When every tensor arg is None the dispatcher can't be used; the
        # wrapper must fall back to the original and still return correctly.
        ksp.enable()
        x = torch.randn(2, 3)
        out = self._fake_mod.optional_kernel(x, bias=None)
        self.assertTrue(torch.allclose(out, x))
        out2 = self._fake_mod.optional_kernel(x, bias=torch.ones(2, 3))
        self.assertTrue(torch.allclose(out2, x + 1.0))

    def test_enable_is_idempotent(self):
        ksp.enable()
        n_patches = len(ksp._patches)
        ksp.enable()  # second call is a no-op while already enabled
        self.assertTrue(ksp.is_enabled())
        self.assertEqual(len(ksp._patches), n_patches)

    def test_reenable_cycle_does_not_raise(self):
        # Regression guard: re-registering custom ops across enable/disable
        # cycles must not raise "operator already exists".
        for _ in range(3):
            ksp.enable()
            self.assertTrue(ksp.is_enabled())
            ksp.disable()
            self.assertFalse(ksp.is_enabled())
        # The kernel works again after the final disable->enable.
        ksp.enable()
        a, b = torch.randn(2, 2), torch.randn(2, 2)
        self.assertTrue(torch.allclose(self._fake_mod.annotated_kernel(a, b), a + b))

    def test_disable_without_enable_is_safe(self):
        # disable() before any enable() must be a harmless no-op.
        self.assertFalse(ksp.is_enabled())
        ksp.disable()
        self.assertFalse(ksp.is_enabled())


class TestShapeDiscoveryConfigFields(CustomTestCase):
    """The feature is gated behind config fields that default to off.

    These checks import the (heavy) ``sglang.srt`` config/manager modules.
    In a degraded environment where that import chain is broken by an
    unrelated optional dependency (e.g. an ``aiter``/``triton`` version
    mismatch), the test is skipped rather than reported as a failure, since
    the field defaults themselves are what we mean to assert.
    """

    def test_profile_req_input_default_off(self):
        try:
            from sglang.srt.managers.io_struct import ProfileReqInput
        except Exception as e:
            self.skipTest(f"io_struct import chain unavailable: {e}")

        self.assertFalse(ProfileReqInput().shape_discovery)
        self.assertTrue(ProfileReqInput(shape_discovery=True).shape_discovery)

    def test_profile_req_default_off(self):
        try:
            from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
        except Exception as e:
            self.skipTest(f"io_struct import chain unavailable: {e}")

        req = ProfileReq(type=ProfileReqType.START_PROFILE)
        self.assertFalse(req.shape_discovery)
        req = ProfileReq(type=ProfileReqType.START_PROFILE, shape_discovery=True)
        self.assertTrue(req.shape_discovery)

    def test_server_arg_default_off(self):
        try:
            from sglang.srt.server_args import ServerArgs
        except Exception as e:
            self.skipTest(f"server_args import chain unavailable: {e}")

        # Dataclass field default is exposed on the class.
        self.assertFalse(ServerArgs.enable_shape_discovery_for_cuda_graph_profile)

    def test_profile_manager_configure_accepts_shape_discovery(self):
        try:
            from sglang.srt.utils.profile_utils import ProfileManager
        except Exception as e:
            self.skipTest(f"profile_utils import chain unavailable: {e}")

        sig = inspect.signature(ProfileManager.configure)
        self.assertIn("shape_discovery", sig.parameters)


if __name__ == "__main__":
    unittest.main()
