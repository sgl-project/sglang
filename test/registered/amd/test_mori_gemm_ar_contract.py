"""The mori surface the fused wo_b all-reduce binds to, on one GPU.

Every mori symbol ``mori_gemm_ar`` calls is bound here with the exact arguments
it passes, and the two operand conversions are checked against mori's own
helpers. A mori whose signatures or layouts moved turns this red at the pinned
commit instead of inside a forward on an eight-rank run.
"""

import inspect
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

import sglang.srt.layers.mori_gemm_ar as mg  # noqa: E402

N, K, M = 5120, 4096, 512

try:
    import mori.ops.gemm_ar  # noqa: F401

    HAS_MORI = True
except ImportError:
    HAS_MORI = False

requires_mori = unittest.skipUnless(HAS_MORI, "this build has no mori.ops.gemm_ar")


@requires_mori
class TestMoriSignatures(CustomTestCase):
    """Bind each call site's arguments without running it.

    ``_compiled`` is mori-private and the rest are keyword-only, so a rename or
    a dropped default is a TypeError raised from inside a forward -- after the
    window is allocated and the peers are in a barrier.
    """

    def test_module_level_helpers(self):
        from mori.ops.gemm_ar import (
            padded_m,
            preshuffle_a_scale,
            preshuffle_b,
            supports,
        )

        inspect.signature(padded_m).bind(16384, 8, mg._BLOCK_M)
        inspect.signature(supports).bind(16384, N, K, 8, quant="mxfp8")
        inspect.signature(preshuffle_b).bind(torch.empty(0))
        inspect.signature(preshuffle_a_scale).bind(torch.empty(0))

    def test_communicator(self):
        from mori.cco import Communicator, UniqueId

        inspect.signature(Communicator.init).bind(8, 0, None, per_rank_vmm=1 << 30)
        self.assertTrue(callable(Communicator.get_unique_id))
        self.assertTrue(callable(UniqueId.from_bytes))

    def test_window_bytes_for(self):
        from mori.ops.gemm_ar import GemmAllReduceOp

        inspect.signature(GemmAllReduceOp.window_bytes_for).bind(
            8, m_max=16384, n=N, block_m=mg._BLOCK_M, gather_dtype="bf16"
        )

    def test_op_construction(self):
        from mori.ops.gemm_ar import GemmAllReduceOp

        inspect.signature(GemmAllReduceOp.__init__).bind(
            None,
            None,
            n=N,
            k=K,
            m_max=16384,
            quant="mxfp8",
            gather_dtype="bf16",
            gather_transport="lsa",
        )

    def test_op_methods(self):
        from mori.ops.gemm_ar import GemmAllReduceOp

        inspect.signature(GemmAllReduceOp._compiled).bind(None, 16384)
        inspect.signature(GemmAllReduceOp.self_test).bind(None)
        inspect.signature(GemmAllReduceOp.pad_rows).bind(None, torch.empty(0), 16384)
        inspect.signature(GemmAllReduceOp.__call__).bind(
            None, torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        )

    def test_quant_and_transport_values_are_accepted(self):
        """The strings this module passes, checked against mori's own tables."""
        from mori.ops.gemm_ar import supports

        for gather_dtype in ("bf16", "fp8"):
            for transport in ("lsa", "sdma"):
                supports(
                    16384,
                    N,
                    K,
                    8,
                    quant="mxfp8",
                    gather_dtype=gather_dtype,
                    gather_transport=transport,
                )


@requires_mori
class TestOperandConversions(CustomTestCase):
    """The two layouts this module writes itself, against mori's helpers.

    Both are silent when wrong: a misplaced scale moves whole rows by a power of
    two and a mislaid weight returns an uncorrelated result, neither of which
    raises anywhere.
    """

    def test_packed_a_scale_matches_preshuffle_a_scale(self):
        """``quantize_packed`` writes mori's A-scale layout without a second pass.

        Also pins that its scale rule is ``fp8_grid_quant``'s, which is what
        ``fused_wo_b``'s claim that re-quantising an ``Fp8GridActivation`` is
        lossless rests on.
        """
        from mori.ops.gemm_ar import preshuffle_a_scale

        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            mxfp8_e4m3_quantize,
        )

        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        q, packed = mg.quantize_packed(x)
        ref_q, ref_exps = mxfp8_e4m3_quantize(x)

        self.assertTrue(torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8)))
        self.assertTrue(torch.equal(packed, preshuffle_a_scale(ref_exps)))

    def test_native_route_weight_matches_preshuffle_b(self):
        """sglang's scaled-MFMA weight order and mori's differ by one permute.

        The two keep a lane's 64-wide K sub-blocks differently; ``mori_weight``
        undoes that rather than re-shuffling from the checkpoint.
        """
        from mori.ops.gemm_ar import preshuffle_b

        from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
            prepare_mxfp8_native_weight,
        )

        plain = torch.randint(0, 255, (N, K), dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
        scale = torch.full((N // 32, K // 32), 1.0, device="cuda")
        shuffled, ue8m0 = prepare_mxfp8_native_weight(plain, scale, (32, 32))

        layer = _layer(shuffled.view(torch.float8_e4m3fn), ue8m0, "native")
        b, _ = mg.mori_weight(layer)
        self.assertTrue(
            torch.equal(b.view(torch.uint8), preshuffle_b(plain).view(torch.uint8))
        )

    def test_aiter_route_weight_is_preshuffle_b(self):
        from mori.ops.gemm_ar import preshuffle_b

        plain = torch.randint(0, 255, (N, K), dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
        ue8m0 = torch.randint(
            120, 135, (N // 32, K // 32), dtype=torch.uint8, device="cuda"
        )
        layer = _layer(plain, ue8m0, "aiter")
        b, _ = mg.mori_weight(layer)
        self.assertTrue(
            torch.equal(b.view(torch.uint8), preshuffle_b(plain).view(torch.uint8))
        )

    def test_b_scale_is_k_major(self):
        """The op reads the B scale transposed and flat; the shapes agree."""
        plain = torch.randint(0, 255, (N, K), dtype=torch.uint8, device="cuda").view(
            torch.float8_e4m3fn
        )
        ue8m0 = torch.randint(
            120, 135, (N // 32, K // 32), dtype=torch.uint8, device="cuda"
        )
        layer = _layer(plain, ue8m0, "aiter")
        _, b_scale = mg.mori_weight(layer)
        self.assertEqual(b_scale.dtype, torch.int32)
        self.assertTrue(
            torch.equal(b_scale, ue8m0.t().contiguous().to(torch.int32).reshape(-1))
        )


def _layer(weight, ue8m0, route):
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale_mx_e8m0 = torch.nn.Parameter(ue8m0, requires_grad=False)
    layer.mxfp8_native_ready = route == "native"
    layer.mxfp8_aiter_ready = route == "aiter"
    return layer


if __name__ == "__main__":
    unittest.main()
