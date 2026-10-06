"""CPU-only safety checks for the metadata-validation cache."""

import importlib.util
import unittest
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_path = (
    Path(__file__).resolve().parents[3]
    / "python/sglang/srt/layers/quantization/mxfp8_dispatch_cache.py"
)
_spec = importlib.util.spec_from_file_location("mxfp8_dispatch_cache", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class TensorMetadata:
    def __init__(self, dtype="fp8", shape=(8, 128), strides=(128, 1)):
        self.dtype = dtype
        self.shape = shape
        self.strides = strides
        self.device = "cuda:0"
        self.layout = "strided"

    def stride(self):
        return self.strides


class DispatchCacheTest(unittest.TestCase):
    def setUp(self):
        self.checks = []

        def raw(a, b, a_scale, b_scale, skip_check=False, **kwargs):
            self.checks.append(skip_check)
            if not skip_check and a.dtype != "fp8":
                raise ValueError("invalid dtype")
            return object()

        self.cache = _module.Mxfp8DispatchCache(
            raw, SimpleNamespace(_cute_dsl_gemm_mxfp8_runner=lambda *args: object())
        )
        self.args = [TensorMetadata() for _ in range(4)]
        self.kwargs = dict(
            out_dtype="bf16", use_8x4_sf_layout=False, backend="cute-dsl"
        )

    def test_reuses_validation_but_not_outputs(self):
        first = self.cache(*self.args, **self.kwargs)
        second = self.cache(*[TensorMetadata() for _ in range(4)], **self.kwargs)
        self.assertIsNot(first, second)
        self.assertEqual(self.checks, [False, True])

    def test_runner_factory_is_wrapped_once_and_reuses_matching_arguments(self):
        module = SimpleNamespace(_cute_dsl_gemm_mxfp8_runner=lambda *args: object())
        first = _module.Mxfp8DispatchCache(self.cache.raw_mm, module)
        second = _module.Mxfp8DispatchCache(self.cache.raw_mm, module)
        self.assertIs(first.runner_factory, second.runner_factory)
        self.assertIs(module._cute_dsl_gemm_mxfp8_runner, first.runner_factory)
        runner = first.runner_factory(10, 0, True, "bf16")
        self.assertIs(runner, second.runner_factory(10, 0, True, "bf16"))
        self.assertIsNot(runner, second.runner_factory(10, 0, False, "bf16"))

    def test_dtype_change_is_revalidated_and_failed_check_not_cached(self):
        self.cache(*self.args, **self.kwargs)
        self.args[0].dtype = "bf16"
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "invalid dtype"):
                self.cache(*self.args, **self.kwargs)
        self.assertEqual(self.checks, [False, False, False])

    def test_shape_stride_and_device_changes_are_revalidated(self):
        self.cache(*self.args, **self.kwargs)
        for attribute, value in (
            ("shape", (16, 128)),
            ("strides", (256, 1)),
            ("device", "cuda:1"),
        ):
            setattr(self.args[0], attribute, value)
            self.cache(*self.args, **self.kwargs)
        self.assertEqual(self.checks, [False] * 4)

    def test_other_backends_keep_original_validation(self):
        self.kwargs["backend"] = "auto"
        for _ in range(2):
            self.cache(*self.args, **self.kwargs)
        self.assertEqual(self.checks, [False, False])

    def test_supported_flashinfer_api_enables_cache(self):
        for version in ("0.7.0", "0.7.0+cu130"):
            with (
                self.subTest(version=version),
                patch.multiple(
                    _module,
                    flashinfer_version=version,
                    gemm_base=SimpleNamespace(_cute_dsl_gemm_mxfp8_runner=lambda: None),
                ),
            ):
                cached = _module.maybe_cache_mxfp8_dispatch(self.cache.raw_mm)
                self.assertIsInstance(cached, _module.Mxfp8DispatchCache)

    def test_decorated_validation_api_accepts_skip_check(self):
        def public_api(a, b, a_scale, b_scale):
            pass

        @wraps(public_api)
        def decorated(*args, **kwargs):
            return self.cache.raw_mm(*args, **kwargs)

        with patch.multiple(
            _module,
            flashinfer_version="0.7.0",
            gemm_base=SimpleNamespace(_cute_dsl_gemm_mxfp8_runner=lambda: None),
        ):
            cached = _module.maybe_cache_mxfp8_dispatch(decorated)
        self.assertIsInstance(cached, _module.Mxfp8DispatchCache)
        cached(*self.args, **self.kwargs)
        cached(*self.args, **self.kwargs)
        self.assertEqual(self.checks, [False, True])

    def test_unsupported_version_does_not_modify_runner(self):
        for version in ("0.6.18", "0.7.1", "0.7.0rc1", "0.7.01", "0.7.0.post1"):
            with self.subTest(version=version):
                factory = lambda: None
                module = SimpleNamespace(_cute_dsl_gemm_mxfp8_runner=factory)
                raw = self.cache.raw_mm
                with patch.multiple(
                    _module, flashinfer_version=version, gemm_base=module
                ):
                    self.assertIs(_module.maybe_cache_mxfp8_dispatch(raw), raw)
                self.assertIs(module._cute_dsl_gemm_mxfp8_runner, factory)

    def test_missing_private_api_keeps_original_dispatch(self):
        raw = self.cache.raw_mm
        for module in (
            None,
            SimpleNamespace(),
            SimpleNamespace(_cute_dsl_gemm_mxfp8_runner=None),
        ):
            with patch.multiple(_module, flashinfer_version="0.7.0", gemm_base=module):
                self.assertIs(_module.maybe_cache_mxfp8_dispatch(raw), raw)
        unsupported_raw = lambda a, b: None
        factory = lambda: None
        module = SimpleNamespace(_cute_dsl_gemm_mxfp8_runner=factory)
        with patch.multiple(_module, flashinfer_version="0.7.0", gemm_base=module):
            self.assertIs(
                _module.maybe_cache_mxfp8_dispatch(unsupported_raw), unsupported_raw
            )
        self.assertIs(module._cute_dsl_gemm_mxfp8_runner, factory)


if __name__ == "__main__":
    unittest.main()
