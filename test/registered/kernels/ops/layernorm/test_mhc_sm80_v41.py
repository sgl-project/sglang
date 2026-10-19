"""V4.1 mHC public-wrapper regression for CUDA SM80.

Run with ``python test_mhc_sm80_v41.py`` (or pytest). No server, checkpoint,
process group, or initialized runtime context is needed. CPU selector tests AST-
load the *unchanged* public wrapper with mock kernels. GPU tests import the real
module and retain every TileLang kernel, including its warnings/race checker.
"""

import ast
import inspect
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple
from unittest.mock import Mock, patch

import torch


_ROOT = Path(__file__).resolve().parents[5]
_SOURCE = _ROOT / "python/sglang/kernels/ops/layernorm/mhc.py"
HC, H, MIX, K = 4, 5120, 24, 20480
EPS, SINKHORN_ITERS = 1e-6, 20


def _load_functions(*names):
    """AST isolation avoids importing distributed/model providers on CPU.

    Extract function bodies verbatim, not a copied selector or reimplementation.
    This also exposes the existing Torch oracle without a private-symbol import.
    """
    tree = ast.parse(_SOURCE.read_text(), filename=str(_SOURCE))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"mHC source no longer defines {names}")
    namespace = {"torch": torch, "Tuple": Tuple}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(_SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _is_sm80():
    return (
        torch.version.hip is None
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (8, 0)
    )


def _standalone_context(module):
    """Mock runtime plumbing only; never replace a GPU compute kernel."""
    stack = ExitStack()
    for name, replacement in {
        "get_parallel": lambda: SimpleNamespace(tp_group=None),
        "use_symmetric_memory": lambda *args, **kwargs: nullcontext(),
        "is_allocation_symmetric": lambda: False,
        "_use_deep_gemm_hc_prenorm": lambda: False,
    }.items():
        stack.enter_context(patch.object(module, name, replacement))
    return stack


class TestMHCPreSelectorCPU(unittest.TestCase):
    def setUp(self):
        self.module = _load_functions("mhc_pre")
        self.stage0, self.stage1 = Mock(), Mock()
        self.builder = Mock(return_value=(self.stage0, self.stage1))
        self.fuse, self.norm_fuse = Mock(), Mock()
        # The extracted function resolves globals in its original exec namespace.
        self.globals = self.module.mhc_pre.__globals__
        self.globals.update(
            get_parallel=lambda: SimpleNamespace(tp_group=None),
            use_symmetric_memory=lambda *args, **kwargs: nullcontext(),
            is_allocation_symmetric=lambda: False,
            _use_deep_gemm_hc_prenorm=lambda: False,
            mhc_pre_gemm_sqrsum_splitk_kernel=self.builder,
            mhc_pre_big_fuse_tilelang=self.fuse,
            mhc_pre_big_fuse_with_norm_tilelang=self.norm_fuse,
        )

    def _call(self, tokens, hidden, *, use_norm=False):
        # Meta tensors exercise real dtype/shape checks and allocations without
        # CUDA initialization or allocating prewarm-sized residual buffers.
        residual = torch.empty(tokens, HC, hidden, dtype=torch.bfloat16, device="meta")
        fn = torch.empty(MIX, HC * hidden, dtype=torch.float32, device="meta")
        scale = torch.empty(3, dtype=torch.float32, device="meta")
        base = torch.empty(MIX, dtype=torch.float32, device="meta")
        norm = (
            torch.empty(hidden, dtype=torch.bfloat16, device="meta")
            if use_norm
            else None
        )
        return self.module.mhc_pre(
            residual, fn, scale, base, EPS, EPS, EPS, 2.0, SINKHORN_ITERS,
            norm_weight=norm, norm_eps=EPS if use_norm else None,
        )

    def test_public_defaults(self):
        signature = inspect.signature(self.module.mhc_pre)
        self.assertEqual(signature.parameters["n_splits"].default, 1)
        self.assertEqual(signature.parameters["n_splits_pre"].default, 32)

    def test_prewarm_width_selector_and_norm_dispatch(self):
        for tokens in (1, 32):
            for hidden, block in ((4096, 256), (H, 128), (7168, 128)):
                for use_norm in (False, True):
                    with self.subTest(tokens=tokens, hidden=hidden, norm=use_norm):
                        post, comb, layer = self._call(tokens, hidden, use_norm=use_norm)
                        self.builder.assert_called_with(
                            MIX, HC * hidden, split_k=32,
                            token_block=32, hidden_block=block,
                        )
                        self.assertEqual(post.shape, (tokens, HC, 1))
                        self.assertEqual(comb.shape, (tokens, HC, HC))
                        self.assertEqual(layer.shape, (tokens, hidden))
                        call = (self.norm_fuse if use_norm else self.fuse).call_args
                        # Fused reduction must consume all 32 padded split-K
                        # partials, not default to the 24-column GEMM layout.
                        self.assertEqual(call.args[-3:], (32, HC, 32))
                        self.assertEqual(call.args[0].shape, (32, tokens, 32))
                        self.assertEqual(call.args[1].shape, (32, tokens))
        self.assertEqual(self.stage0.call_count, 12)
        self.stage1.assert_not_called()

    def test_unsupported_splitk_width_fails_before_kernel(self):
        with self.assertRaisesRegex(NotImplementedError, "18432"):
            self._call(1, 4608)
        self.builder.assert_not_called()
        self.stage0.assert_not_called()
        self.fuse.assert_not_called()
        self.norm_fuse.assert_not_called()


@unittest.skipUnless(_is_sm80(), "requires CUDA SM80 (8, 0)")
class TestMHCSM80V41(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Import after the architecture gate: CPU collection does not import SRT
        # providers or initialize model/runtime state. No kernel is mocked here.
        import sglang.kernels.ops.layernorm.mhc as mhc

        cls.mhc = mhc
        cls.oracle = _load_functions("_mhc_pre_torch", "_mhc_post_torch")
        cls.addClassCleanup(
            setattr, torch.backends.cuda.matmul, "allow_tf32",
            torch.backends.cuda.matmul.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32 = False

    def setUp(self):
        self.context = _standalone_context(self.mhc)
        self.addCleanup(self.context.close)
        # mhc_post's CP-contiguity predicate otherwise reads the model runtime.
        # This is topology plumbing, not a numerical or kernel replacement.
        self.context.enter_context(
            patch.object(self.mhc, "is_dsa_prefill_cp_interleave", lambda: False)
        )

    def _inputs(self, tokens):
        torch.manual_seed(4100 + tokens)
        residual = torch.randn(tokens, HC, H, dtype=torch.bfloat16, device="cuda") * 0.1
        fn = torch.randn(MIX, K, dtype=torch.float32, device="cuda") * 0.01
        scale = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32, device="cuda")
        base = torch.randn(MIX, dtype=torch.float32, device="cuda") * 0.1
        norm = torch.linspace(0.75, 1.25, H, dtype=torch.bfloat16, device="cuda")
        return residual, fn, scale, base, norm

    def _assert_unnorm_layer(self, layer, reference, residual, fn, scale, base):
        """Keep the old threshold; diagnose/prove rare BF16 midpoint crossings.

        Only elements rejected by the existing 2e-3 check get a one-ULP
        exception, and only after an independent scalar FP64 reconstruction
        places the true weighted sum at that rounding boundary. This does not
        grant every element a wider absolute/relative tolerance.
        """
        actual, expected = layer.float(), reference.float()
        rejected = (actual - expected).abs() > 2e-3 + 2e-3 * expected.abs()
        if not rejected.any().item():
            torch.testing.assert_close(layer, reference, rtol=2e-3, atol=2e-3)
            return

        def ordered_bf16(value):
            bits = value.contiguous().view(torch.int16).int()
            # Monotonic signed encoding, with both signed zeros at zero.
            return torch.where(bits < 0, -32768 - bits, bits)

        ulps = (ordered_bf16(layer) - ordered_bf16(reference)).abs()
        coordinates = rejected.nonzero()
        print(
            f"mHC BF16 diagnostic T={layer.shape[0]}: "
            f"different={(layer != reference).sum().item()}, "
            f"max_ulp_all={ulps.max().item()}, "
            f"old_threshold_rejected={coordinates.shape[0]}, "
            f"max_ulp_rejected={ulps[rejected].max().item()}, "
            f"max_abs={(actual - expected).abs().max().item()}",
            flush=True,
        )
        self.assertLessEqual(ulps[rejected].max().item(), 1)
        # Preserve the full token-block shape of the failing public GEMM.
        tokens = residual.shape[0]
        self.assertGreater(tokens, 2048, "unexpected split-K threshold exception")
        gemm = torch.empty(tokens, MIX, dtype=torch.float32, device="cuda")
        sqrsum = torch.empty(tokens, dtype=torch.float32, device="cuda")
        self.mhc.mhc_pre_gemm_sqrsum_tilelang(
            residual.view(tokens, K), fn, gemm, sqrsum, MIX, K
        )
        fp32_eps = torch.finfo(torch.float32).eps
        # Copy weights once; scalar proofs transfer only exceptional token rows,
        # never the full 2049x20480 residual to CPU.
        fn64 = fn.cpu().double()
        self.assertTrue(
            ((fn64 == 0) | (fn64.abs() >= torch.finfo(torch.float32).tiny)).all().item(),
            "TF32 relative quantization proof requires normal-or-zero FP32 weights",
        )
        scale64, base64 = scale.cpu().double(), base.cpu().double()
        for token, hidden in coordinates.cpu().tolist():
            row = residual[token].cpu().double()
            xd = row.flatten()
            # Elementwise FP64 products + scalar reductions, independent of
            # cuBLAS GEMM and the Torch pre oracle's FP32 reduction order.
            dot = (fn64 * xd.unsqueeze(0)).sum(-1)
            square_sum = xd.square().sum()
            kernel_dot, kernel_square = gemm[token].cpu().double(), sqrsum[token].cpu().double()
            scalar_z = dot * torch.rsqrt(square_sum / K + EPS)
            kernel_z = kernel_dot * torch.rsqrt(kernel_square / K + EPS)
            # SM80 T.gemm on FP32 operands uses TF32 MMA independently of
            # PyTorch's allow_tf32 setting. BF16 x is exactly representable in
            # TF32, but FP32 fn loses 13 fraction bits. One TF32 spacing bounds
            # either round-to-nearest or truncation; do not assume a conversion
            # mode or compare this kernel against a FP32-only error budget.
            # For normal fn, |delta fn| <= 2^-10 |fn|. Propagate that input
            # error through the dot product, plus the standard conservative
            # gamma_K bound for FP32 accumulation (not a relaxed fixed atol).
            product_abs_sum = (fn64 * xd.unsqueeze(0)).abs().sum(-1)
            tf32_input_bound = (2.0 ** -10) * product_abs_sum
            gamma_k = K * fp32_eps / (1 - K * fp32_eps)
            dot_bound = tf32_input_bound + gamma_k * product_abs_sum * (1 + 2.0 ** -10)
            dot_error = (kernel_dot - dot).abs()
            self.assertTrue((dot_error <= dot_bound).all().item())
            torch.testing.assert_close(kernel_square, square_sum, rtol=2e-5, atol=2e-4)
            pre = torch.sigmoid(scalar_z[:HC] * scale64[0] + base64[:HC]) + EPS
            column = row[:, hidden]
            scalar_sum = (pre * column).sum()
            # sigmoid is globally 1/4-Lipschitz. Propagate the *measured*
            # normalized GEMM error; eight FP32 eps cover sigmoid/FMA/rsqrt
            # roundoff, and gamma_8 bounds the four-term weighted reduction.
            pre_error = (
                0.25 * scale64[0].abs() * (kernel_z[:HC] - scalar_z[:HC]).abs()
                + 8 * fp32_eps
            )
            gamma8 = 8 * fp32_eps / (1 - 8 * fp32_eps)
            budget = (column.abs() * pre_error).sum() + gamma8 * (pre * column).abs().sum()
            a = actual[token, hidden].cpu().double()
            b = expected[token, hidden].cpu().double()
            gap = (a - b).abs()
            midpoint = (a + b) / 2
            # The proof budget must be tiny relative to the BF16 spacing: a
            # generic numerical error cannot hide behind an arbitrary 1 ULP.
            self.assertLess(budget.item(), gap.item() / 100)
            self.assertLessEqual((scalar_sum - midpoint).abs().item(), budget.item())
            for rounded in (a, b):
                self.assertLessEqual((rounded - scalar_sum).abs().item(), (gap / 2 + budget).item())
            print(
                f"  scalar ({token},{hidden}): ulp={ulps[token, hidden].item()}, "
                f"gemm_max_abs={dot_error.max().item():.3g}, "
                f"tf32_input_bound_max={tf32_input_bound.max().item():.3g}, "
                f"dot_error_bound_ratio={(dot_error / dot_bound).max().item():.3g}, "
                f"sqrsum_abs={(kernel_square - square_sum).abs().item():.3g}, "
                f"midpoint_distance={(scalar_sum - midpoint).abs().item():.3g}, "
                f"budget={budget.item():.3g}, bf16_gap={gap.item():.3g}",
                flush=True,
            )

    def test_public_pre_norm_and_post_match_torch(self):
        # T=1/32 exercise decode/prewarm's split-K public selector. T=2049
        # crosses the <=2048 branch and exercises the nonsplit GEMM dispatch.
        for tokens in (1, 32, 2049):
            residual, fn, scale, base, norm = self._inputs(tokens)
            expected = self.oracle._mhc_pre_torch(
                residual, fn, scale, base, EPS, EPS, EPS, 2.0, SINKHORN_ITERS
            )
            for use_norm in (False, True):
                with self.subTest(tokens=tokens, norm=use_norm):
                    actual = self.mhc.mhc_pre(
                        residual, fn, scale, base, EPS, EPS, EPS, 2.0,
                        SINKHORN_ITERS,
                        norm_weight=norm if use_norm else None,
                        norm_eps=EPS if use_norm else None,
                    )
                    self.assertIsInstance(actual, tuple)
                    self.assertEqual(len(actual), 3)
                    post, comb, layer = actual
                    for value, shape, dtype in zip(
                        actual,
                        ((tokens, HC, 1), (tokens, HC, HC), (tokens, H)),
                        (torch.float32, torch.float32, torch.bfloat16),
                    ):
                        self.assertEqual(value.shape, shape)
                        self.assertEqual(value.dtype, dtype)
                        self.assertTrue(torch.isfinite(value).all().item())
                    for dim in (-1, -2):
                        torch.testing.assert_close(
                            comb.sum(dim), torch.ones(tokens, HC, device="cuda"),
                            rtol=0, atol=2e-5,
                        )
                    torch.testing.assert_close(post, expected[0], rtol=2e-3, atol=2e-3)
                    torch.testing.assert_close(comb, expected[1], rtol=2e-3, atol=2e-3)
                    layer_ref = expected[2]
                    if use_norm:
                        value = layer_ref.float()
                        layer_ref = (
                            value * torch.rsqrt(value.square().mean(-1, keepdim=True) + EPS)
                            * norm.float()
                        ).bfloat16()
                    # Fused norm accumulates the unrounded weighted-sum square
                    # before staging BF16; the existing Torch path rounds first.
                    if use_norm:
                        torch.testing.assert_close(layer, layer_ref, rtol=2e-2, atol=2e-2)
                    else:
                        self._assert_unnorm_layer(layer, layer_ref, residual, fn, scale, base)
                    x = (layer_ref.float() * 0.75).bfloat16()
                    out = self.mhc.mhc_post(x, residual, post, comb)
                    # Existing in-tree post oracle, with FP32 operands so the
                    # first product is not prematurely rounded to BF16.
                    out_ref = self.oracle._mhc_post_torch(
                        x.float(), residual.float(), post, comb
                    ).bfloat16()
                    torch.cuda.synchronize()
                    self.assertEqual(out.shape, residual.shape)
                    self.assertEqual(out.dtype, torch.bfloat16)
                    self.assertTrue(torch.isfinite(out).all().item())
                    torch.testing.assert_close(out, out_ref, rtol=2e-2, atol=2e-3)

    def test_actual_gemm_and_sqrsum_match_reference(self):
        for tokens in (1, 32, 2049):
            with self.subTest(tokens=tokens):
                residual, fn, _, _, _ = self._inputs(tokens)
                x = residual.view(tokens, K)
                if tokens <= 2048:
                    stage0, stage1 = self.mhc.mhc_pre_gemm_sqrsum_splitk_kernel(
                        MIX, K, split_k=32, token_block=32, hidden_block=128
                    )
                    partial = torch.empty(32, tokens, 32, dtype=torch.float32, device="cuda")
                    sq_partial = torch.empty(32, tokens, dtype=torch.float32, device="cuda")
                    stage0(x, fn, partial, sq_partial)
                    gemm = torch.empty(tokens, MIX, dtype=torch.float32, device="cuda")
                    sqrsum = torch.empty(tokens, dtype=torch.float32, device="cuda")
                    stage1(partial, sq_partial, gemm, sqrsum)
                    torch.testing.assert_close(
                        partial[:, :, :MIX].sum(0), gemm, rtol=2e-5, atol=2e-5
                    )
                else:
                    gemm = torch.empty(tokens, MIX, dtype=torch.float32, device="cuda")
                    sqrsum = torch.empty(tokens, dtype=torch.float32, device="cuda")
                    self.mhc.mhc_pre_gemm_sqrsum_tilelang(x, fn, gemm, sqrsum, MIX, K)
                # FP64 oracle avoids TF32 policy affecting the expected GEMM.
                xd = x.double()
                gemm_ref = (xd @ fn.double().T).float()
                sqrsum_ref = xd.square().sum(-1).float()
                torch.cuda.synchronize()
                torch.testing.assert_close(gemm, gemm_ref, rtol=2e-3, atol=2e-3)
                torch.testing.assert_close(sqrsum, sqrsum_ref, rtol=2e-5, atol=2e-4)
                torch.testing.assert_close(
                    torch.rsqrt(sqrsum / K + EPS),
                    torch.rsqrt(sqrsum_ref / K + EPS), rtol=2e-5, atol=2e-5,
                )


if __name__ == "__main__":
    unittest.main()
