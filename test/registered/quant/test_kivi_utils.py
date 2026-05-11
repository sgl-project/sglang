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
kivi_roundtrip_kv_chunk = _KIVI_UTILS.kivi_roundtrip_kv_chunk


class TestKIVIUtils(unittest.TestCase):
    def test_unpack_and_dequant_kcache_uses_quant_param_dtype(self):
        base = torch.arange(1, 1 + 1 * 2 * 16 * 16, dtype=torch.float32).view(
            1, 2, 16, 16
        )

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                k = base.to(dtype)
                code, scale, mn = quant_and_pack_kcache(k, group_size=4, bits=2)
                out = unpack_and_dequant_kcache(code, scale, mn, group_size=4, bits=2)

                self.assertEqual(scale.dtype, dtype)
                self.assertEqual(mn.dtype, dtype)
                self.assertEqual(out.dtype, dtype)
                self.assertEqual(out.shape, k.shape)

    def test_unpack_and_dequant_vcache_uses_quant_param_dtype(self):
        base = torch.arange(1, 1 + 1 * 2 * 16 * 16, dtype=torch.float32).view(
            1, 2, 16, 16
        )

        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                v = base.to(dtype)
                code, scale, mn = quant_and_pack_vcache(v, group_size=4, bits=2)
                out = unpack_and_dequant_vcache(code, scale, mn, group_size=4, bits=2)

                self.assertEqual(scale.dtype, dtype)
                self.assertEqual(mn.dtype, dtype)
                self.assertEqual(out.dtype, dtype)
                self.assertEqual(out.shape, v.shape)

    def test_constant_group_quantization_does_not_produce_nan(self):
        k = torch.ones((1, 2, 16, 16), dtype=torch.float16)
        v = torch.ones((1, 2, 16, 16), dtype=torch.float16)

        k_code, k_scale, k_mn = quant_and_pack_kcache(k, group_size=4, bits=2)
        v_code, v_scale, v_mn = quant_and_pack_vcache(v, group_size=4, bits=2)
        k_out = unpack_and_dequant_kcache(k_code, k_scale, k_mn, group_size=4, bits=2)
        v_out = unpack_and_dequant_vcache(v_code, v_scale, v_mn, group_size=4, bits=2)

        self.assertFalse(torch.isnan(k_out).any())
        self.assertFalse(torch.isnan(v_out).any())
        self.assertTrue(torch.equal(k_out, k))
        self.assertTrue(torch.equal(v_out, v))

    def test_roundtrip_keeps_residual_tokens_full_precision(self):
        cache_k = torch.randn((24, 2, 16), dtype=torch.float16)
        cache_v = torch.randn((24, 2, 16), dtype=torch.float16)
        k_out, v_out = kivi_roundtrip_kv_chunk(
            cache_k,
            cache_v,
            k_bits=2,
            v_bits=2,
            k_group_size=4,
            v_group_size=4,
            residual_length=8,
        )

        self.assertEqual(k_out.shape, cache_k.shape)
        self.assertEqual(v_out.shape, cache_v.shape)
        self.assertTrue(torch.equal(k_out[-8:], cache_k[-8:]))
        self.assertTrue(torch.equal(v_out[-8:], cache_v[-8:]))

    def test_roundtrip_does_not_pad_short_chunks(self):
        cache_k = torch.randn((7, 2, 16), dtype=torch.float16)
        cache_v = torch.randn((7, 2, 16), dtype=torch.float16)
        k_out, v_out = kivi_roundtrip_kv_chunk(
            cache_k,
            cache_v,
            k_bits=2,
            v_bits=2,
            k_group_size=4,
            v_group_size=4,
            residual_length=8,
        )

        self.assertTrue(torch.equal(k_out, cache_k))
        self.assertTrue(torch.equal(v_out, cache_v))


if __name__ == "__main__":
    unittest.main()
