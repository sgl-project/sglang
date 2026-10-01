"""CPU-only safety checks for the opt-in metadata-validation cache."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
