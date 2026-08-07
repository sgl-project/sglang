import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache import memory_pool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMLAFP8WriteRouting(unittest.TestCase):
    def test_packed_dsa_writer_precedes_hip_raw_fp8_writer(self):
        pool = SimpleNamespace(
            dsa_kv_cache_store_fp8=True,
            use_dsa=True,
            dtype=memory_pool.fp8_dtype,
        )
        packed_nope = object()
        packed_rope = object()

        with (
            mock.patch.object(
                memory_pool,
                "quantize_k_cache_separate",
                return_value=(packed_nope, packed_rope),
            ) as quantize,
            mock.patch.object(memory_pool, "set_mla_kv_buffer_triton") as write_packed,
            mock.patch.object(
                memory_pool, "set_mla_kv_buffer_triton_fp8_quant"
            ) as write_raw,
            mock.patch.object(memory_pool, "_is_hip", True),
        ):
            memory_pool.MLATokenToKVPool._write_mla_kv_buffer(
                pool, "dst", "loc", "nope", "rope"
            )

        quantize.assert_called_once_with("nope", "rope")
        write_packed.assert_called_once_with("dst", "loc", packed_nope, packed_rope)
        write_raw.assert_not_called()

    def test_raw_hip_fp8_writer_remains_the_fallback(self):
        pool = SimpleNamespace(
            dsa_kv_cache_store_fp8=False,
            use_dsa=True,
            dtype=memory_pool.fp8_dtype,
        )

        with (
            mock.patch.object(memory_pool, "quantize_k_cache_separate") as quantize,
            mock.patch.object(memory_pool, "set_mla_kv_buffer_triton") as write_packed,
            mock.patch.object(
                memory_pool, "set_mla_kv_buffer_triton_fp8_quant"
            ) as write_raw,
            mock.patch.object(memory_pool, "_is_hip", True),
        ):
            memory_pool.MLATokenToKVPool._write_mla_kv_buffer(
                pool, "dst", "loc", "nope", "rope"
            )

        quantize.assert_not_called()
        write_packed.assert_not_called()
        write_raw.assert_called_once_with(
            "dst", "loc", "nope", "rope", memory_pool.fp8_dtype
        )


if __name__ == "__main__":
    unittest.main()
