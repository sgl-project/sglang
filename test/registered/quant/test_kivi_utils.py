import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch


def register_cpu_ci(*_args, **_kwargs):
    return None

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

_MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "python"
    / "sglang"
    / "srt"
    / "layers"
    / "quantization"
    / "kivi_utils.py"
)
_SPEC = spec_from_file_location("kivi_utils_under_test", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Failed to load KIVI utils module from {_MODULE_PATH}.")
_KIVI_UTILS = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_KIVI_UTILS)

quant_and_pack_kcache = _KIVI_UTILS.quant_and_pack_kcache
quant_and_pack_vcache = _KIVI_UTILS.quant_and_pack_vcache
unpack_and_dequant_kcache = _KIVI_UTILS.unpack_and_dequant_kcache
unpack_and_dequant_vcache = _KIVI_UTILS.unpack_and_dequant_vcache


class TestKIVIUtils(unittest.TestCase):
    def test_unpack_and_dequant_kcache_uses_quant_param_dtype(self):
        base = torch.arange(
            1, 1 + 1 * 2 * 16 * 16, dtype=torch.float32
        ).view(1, 2, 16, 16)

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                k = base.to(dtype)
                code, scale, mn = quant_and_pack_kcache(k, group_size=4, bits=2)
                out = unpack_and_dequant_kcache(
                    code, scale, mn, group_size=4, bits=2
                )

                self.assertEqual(scale.dtype, dtype)
                self.assertEqual(mn.dtype, dtype)
                self.assertEqual(out.dtype, dtype)
                self.assertEqual(out.shape, k.shape)

    def test_unpack_and_dequant_vcache_uses_quant_param_dtype(self):
        base = torch.arange(
            1, 1 + 1 * 2 * 16 * 16, dtype=torch.float32
        ).view(1, 2, 16, 16)

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                v = base.to(dtype)
                code, scale, mn = quant_and_pack_vcache(v, group_size=4, bits=2)
                out = unpack_and_dequant_vcache(
                    code, scale, mn, group_size=4, bits=2
                )

                self.assertEqual(scale.dtype, dtype)
                self.assertEqual(mn.dtype, dtype)
                self.assertEqual(out.dtype, dtype)
                self.assertEqual(out.shape, v.shape)


if __name__ == "__main__":
    unittest.main()
