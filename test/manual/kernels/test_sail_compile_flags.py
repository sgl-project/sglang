"""SAIL cold-build regressions for shared HGCC and TileLang options.

Run with an empty SGLANG_JIT_CACHE_DIR and TILELANG_CACHE_DIR on a PPU.
The copy probe uses the actual v2 options without requiring its SM90 TMA path.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import tilelang
import tilelang.language as T
import torch

from sglang.kernels.jit.utils import load_jit
from sglang.kernels.jit.utils.compile.flags import filter_hgcc_cuda_flags
from sglang.kernels.ops.attention import rope
from sglang.kernels.ops.attention.dsa import tilelang_kernel as dsa
from sglang.srt.utils.device import is_sail
from sglang.test.test_utils import CustomTestCase


def _copy_prim_func(n: int):
    @T.prim_func
    def main(a: T.Tensor((n,), "float32"), b: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            for i in T.Parallel(128):
                if block * 128 + i < n:
                    b[block * 128 + i] = a[block * 128 + i]

    return main


class TestSailCompileFlags(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not is_sail() or not torch.cuda.is_available():
            raise unittest.SkipTest("requires the SAIL PPU compiler and runtime")

    @torch.inference_mode()
    def test_tilelang_v2_options_copy_and_replay(self):
        with patch.object(dsa, "is_sail", return_value=False):
            cuda_flags = dsa._sparse_attention_v2_compile_flags()
        self.assertIn("--ptxas-options=-v,--register-usage-level=10", cuda_flags)
        sail_flags = dsa._sparse_attention_v2_compile_flags()
        self.assertEqual(sail_flags, filter_hgcc_cuda_flags(cuda_flags))
        self.assertNotIn("--ptxas-options=-v,--register-usage-level=10", sail_flags)
        flags = sail_flags + ["--ftz=false", "--fmad=false"]
        kernel = tilelang.jit(out_idx=[-1], compile_flags=flags)(_copy_prim_func)(256)
        values = torch.randn(256, device="cuda", dtype=torch.float32)
        output = kernel(values)
        self.assertTrue(torch.equal(output, values))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = kernel(values)
        for offset in (0.5, -1.25, 2.0):
            values.add_(offset)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output, values))

    def test_split_codegen_options_build_real_jit_modules(self):
        # A real load_jit build catches orphaned arch=... operands and confirms
        # that both math switches reach HGCC alongside each normalization case.
        source = """#include <cstdint>
#include <tvm/ffi/container/tensor.h>
namespace sglang {
struct SailFlagsProbe {
  static int32_t run() { return 37; }
};
}
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.cuh"
            path.write_text(source)
            for option, operand in (
                ("-gencode", "arch=compute_89,code=sm_89"),
                ("--generate-code", "arch=compute_89,code=sm_89"),
                ("-Xptxas", "--split-compile=0"),
                ("--ptxas-options", "-v,--register-usage-level=10"),
            ):
                with self.subTest(option=option):
                    module = load_jit(
                        "sail_compile_flags_probe",
                        option.removeprefix("-"),
                        cuda_files=[str(path)],
                        cuda_wrappers=[("run", "SailFlagsProbe::run")],
                        extra_cuda_cflags=[
                            "--ftz=false",
                            option,
                            operand,
                            "--fmad=false",
                        ],
                    )
                    self.assertEqual(module.run(), 37)

    @torch.inference_mode()
    def test_rope_scale_store_matches_unfused_and_replays(self):
        torch.manual_seed(37519)
        tokens, heads, kv_heads, dim = 4, 2, 1, 64
        q_width, k_width = heads * dim, kv_heads * dim
        qkv = torch.randn(tokens, q_width + 2 * k_width, device="cuda").bfloat16()
        query = torch.empty(tokens, q_width, device="cuda", dtype=torch.bfloat16)
        key_cache = torch.full((16, kv_heads, dim), -7, device="cuda").bfloat16()
        value_cache = torch.full_like(key_cache, -9)
        positions = torch.tensor([1, 3, 5, 7], device="cuda", dtype=torch.int64)
        locations = torch.tensor([2, 5, 9, 0], device="cuda", dtype=torch.int32)
        angles = torch.randn(32, dim // 2, device="cuda")
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()

        def operation():
            rope.bf16_rope_scale_store(
                qkv, query, key_cache, value_cache, cos_sin, positions, locations, 1.5
            )

        def check():
            q = qkv[:, :q_width].contiguous().clone()
            k = qkv[:, q_width : q_width + k_width].contiguous().clone()
            rope.rotary_embedding_with_key(positions, q, k, dim, cos_sin, True)
            v = (qkv[:, q_width + k_width :].float() * 1.5).bfloat16()
            valid = locations > 0
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(query, q))
            self.assertTrue(
                torch.equal(
                    key_cache[locations[valid]], k[valid].view(-1, kv_heads, dim)
                )
            )
            self.assertTrue(
                torch.equal(
                    value_cache[locations[valid]], v[valid].view(-1, kv_heads, dim)
                )
            )
            self.assertTrue(torch.all(key_cache[0] == -7))
            self.assertTrue(torch.all(value_cache[0] == -9))

        operation()
        check()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            operation()
        for offset in (0.125, -0.25, 0.5):
            qkv.add_(offset)
            positions.add_(1)
            graph.replay()
            check()


if __name__ == "__main__":
    unittest.main()
