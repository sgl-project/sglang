"""Numeric unit tests for the XPU int4 *dense* linear kernels (GPTQ / AWQ)."""

import unittest

import torch

from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=45, suite="stage-b-test-1-gpu-xpu")

DEV = "xpu"

REL_TOL = {torch.float16: 1.5e-3, torch.bfloat16: 2e-2}
M_VALUES = (1, 8, 256)

# (K, N, group_size); K % 8 == 0, N % 8 == 0, K % group_size == 0.
SHAPES = [
    (128, 64, 32),
    (256, 128, 64),
    (256, 128, 128),
    (512, 256, 256),
]

# AutoAWQ forward pack order (inverse of reverse [0, 4, 1, 5, 2, 6, 3, 7]).
AWQ_PACK_ORDER = [0, 2, 4, 6, 1, 3, 5, 7]


def _awq_pack(codes: torch.Tensor) -> torch.Tensor:
    """``[R, C]`` codes (0..15) -> ``[R, C // 8]`` int32 in AutoAWQ order."""
    r, c = codes.shape
    codes = codes.reshape(r, c // 8, 8)[:, :, AWQ_PACK_ORDER]
    packed = torch.zeros(r, c // 8, dtype=torch.int32, device=codes.device)
    for i in range(8):
        packed |= codes[:, :, i].to(torch.int32) << (4 * i)
    return packed


def _gptq_pack_qweight(codes: torch.Tensor) -> torch.Tensor:
    """``[K, N]`` codes -> ``[K // 8, N]`` int32 (packed sequentially along K)."""
    k, n = codes.shape
    codes = codes.reshape(k // 8, 8, n)
    packed = torch.zeros(k // 8, n, dtype=torch.int32, device=codes.device)
    for i in range(8):
        packed |= codes[:, i, :].to(torch.int32) << (4 * i)
    return packed


def _gptq_pack_qzeros(zc: torch.Tensor) -> torch.Tensor:
    """``[ng, N]`` codes -> ``[ng, N // 8]`` int32 (packed sequentially along N)."""
    ng, n = zc.shape
    zc = zc.reshape(ng, n // 8, 8)
    packed = torch.zeros(ng, n // 8, dtype=torch.int32, device=zc.device)
    for j in range(8):
        packed |= zc[:, :, j].to(torch.int32) << (4 * j)
    return packed


def _ct_pack_qweight(codes: torch.Tensor) -> torch.Tensor:
    """``[N, K]`` codes (0..15) -> ``[N, K // 8]`` int32 packed along K.

    compressed-tensors uses natural nibble order; validated against the library's
    own ``pack_to_int32`` in ``test_unpack_matches_compressed_tensors_packer``.
    """
    n, k = codes.shape
    codes = codes.reshape(n, k // 8, 8)
    packed = torch.zeros(n, k // 8, dtype=torch.int32, device=codes.device)
    for i in range(8):
        packed |= codes[:, :, i].to(torch.int32) << (4 * i)
    return packed


def _make_layer():
    """A bare ``LinearBase`` with only ``nn.Module`` machinery initialised."""
    from sglang.srt.layers.linear import LinearBase

    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    return layer


def _awq_config(group_size: int):
    from sglang.srt.layers.quantization.awq import AWQXPUConfig

    cfg = AWQXPUConfig.__new__(AWQXPUConfig)
    cfg.group_size = group_size
    cfg.weight_bits = 4
    cfg.pack_factor = 8
    cfg.zero_point = True
    cfg.lm_head_quantized = False
    cfg.modules_to_not_convert = []
    return cfg


def _gptq_config(group_size: int, desc_act: bool, fmt: str):
    from sglang.srt.layers.quantization.gptq import GPTQXPUConfig

    cfg = GPTQXPUConfig.__new__(GPTQXPUConfig)
    cfg.group_size = group_size
    cfg.desc_act = desc_act
    cfg.checkpoint_format = fmt
    cfg.weight_bits = 4
    cfg.lm_head_quantized = False
    cfg.dynamic = {}
    return cfg


@unittest.skipIf(not is_xpu(), "XPU int4 dense UT requires an Intel XPU")
class TestXPUInt4DenseKernel(CustomTestCase):
    """AWQ / GPTQ int4pack kernel numerics vs a pure-torch dequant reference."""

    def test_awq_numeric(self):
        for dtype in (torch.float16, torch.bfloat16):
            for m in M_VALUES:
                for k, n, gs in SHAPES:
                    with self.subTest(dtype=dtype, M=m, K=k, N=n, gs=gs):
                        self._run_awq(m, k, n, gs, dtype)

    def _run_awq(self, m, k, n, gs, dtype):
        from sglang.srt.hardware_backend.xpu.quantization.awq_kernels import (
            AWQXPULinearKernel,
        )

        torch.manual_seed(0)
        ng = k // gs
        wcodes = torch.randint(0, 16, (k, n), device=DEV)
        zcodes = torch.randint(0, 16, (ng, n), device=DEV)
        scales = torch.rand(ng, n, device=DEV, dtype=dtype) * 0.05 + 0.005

        gidx = torch.arange(k, device=DEV) // gs
        w_ref = (wcodes.to(dtype) - zcodes[gidx].to(dtype)) * scales[gidx]
        x = torch.randn(m, k, device=DEV, dtype=dtype)
        ref = x @ w_ref

        layer = _make_layer()
        layer.qweight = torch.nn.Parameter(_awq_pack(wcodes), requires_grad=False)
        layer.qzeros = torch.nn.Parameter(_awq_pack(zcodes), requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)

        kernel = AWQXPULinearKernel(_awq_config(gs))
        kernel.process_weights_after_loading(layer)
        out = kernel.apply(layer, x)

        self.assertEqual(tuple(out.shape), (m, n))
        self.assertTrue(torch.isfinite(out).all())
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(rel, REL_TOL[dtype], f"rel={rel:.2e}")

    def test_gptq_numeric(self):
        for dtype in (torch.float16, torch.bfloat16):
            for fmt in ("", "gptq_v2"):
                for desc_act in (False, True):
                    for m in M_VALUES:
                        for k, n, gs in SHAPES:
                            with self.subTest(
                                dtype=dtype,
                                fmt=fmt or "gptq_v1",
                                desc_act=desc_act,
                                M=m,
                                K=k,
                                N=n,
                                gs=gs,
                            ):
                                self._run_gptq(m, k, n, gs, dtype, desc_act, fmt)

    def _run_gptq(self, m, k, n, gs, dtype, desc_act, fmt, tp_size=1):
        from sglang.srt.hardware_backend.xpu.quantization.gptq_kernels import (
            GPTQXPULinearKernel,
        )

        torch.manual_seed(0)
        ng = k // gs
        qnat = torch.randint(0, 16, (k, n), device=DEV)
        zc = torch.randint(0, 14, (ng, n), device=DEV)  # room for v1 +1
        scales = torch.rand(ng, n, device=DEV, dtype=dtype) * 0.05 + 0.005

        if desc_act:
            base = torch.arange(k, device=DEV) // gs
            g_idx = base[torch.randperm(k, device=DEV)].to(torch.int32)
        else:
            g_idx = (torch.arange(k, device=DEV) // gs).to(torch.int32)

        zp_eff = zc + (0 if fmt == "gptq_v2" else 1)
        w_true = (qnat.to(dtype) - zp_eff[g_idx].to(dtype)) * scales[g_idx]
        x = torch.randn(m, k, device=DEV, dtype=dtype)
        ref = x @ w_true

        layer = _make_layer()
        layer.qweight = torch.nn.Parameter(
            _gptq_pack_qweight(qnat), requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(_gptq_pack_qzeros(zc), requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)
        layer.g_idx = torch.nn.Parameter(g_idx, requires_grad=False)

        kernel = GPTQXPULinearKernel(_gptq_config(gs, desc_act, fmt))
        with get_parallel().override(tp_size=tp_size):
            kernel.process_weights_after_loading(layer)
        out = kernel.apply(layer, x)

        self.assertEqual(tuple(out.shape), (m, n))
        self.assertTrue(torch.isfinite(out).all())
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(rel, REL_TOL[dtype], f"rel={rel:.2e}")

    def test_gptq_act_order_rejects_split_group_shard(self):
        from sglang.srt.hardware_backend.xpu.quantization.gptq_kernels import (
            GPTQXPULinearKernel,
        )

        k, n, gs = 128, 64, 32
        # A row-parallel shard of a permuted K owns only part of every group, so
        # after sorting each gs-block still straddles two groups.
        g_idx = (torch.arange(2 * k, device=DEV) // gs)[0::2].to(torch.int32)

        # Only g_idx matters here; the payload is never reached.
        layer = _make_layer()
        layer.qweight = torch.nn.Parameter(
            torch.zeros(k // 8, n, dtype=torch.int32, device=DEV), requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(
            torch.zeros(k // gs, n // 8, dtype=torch.int32, device=DEV),
            requires_grad=False,
        )
        layer.scales = torch.nn.Parameter(
            torch.ones(k // gs, n, device=DEV, dtype=torch.float16),
            requires_grad=False,
        )
        layer.g_idx = torch.nn.Parameter(g_idx, requires_grad=False)
        kernel = GPTQXPULinearKernel(_gptq_config(gs, True, ""))

        # The limit is representability, not TP: tp_size only decides whether the
        # actionable --tp-size hint is appended. The layer is safe to reuse
        # because the check fires before any weight is replaced.
        for tp_size, pattern in (
            (1, r"K boundary\.$"),
            (2, r"tp_size=2.*--tp-size 1"),
        ):
            with self.subTest(tp_size=tp_size):
                with get_parallel().override(tp_size=tp_size):
                    with self.assertRaisesRegex(NotImplementedError, pattern):
                        kernel.process_weights_after_loading(layer)

    def test_gptq_group_aligned_shard_allows_tensor_parallel(self):
        # Whole-group shards stay representable, so TP is only rejected when a
        # group is actually split (act_order) -- not for TP as such.
        for dtype in (torch.float16, torch.bfloat16):
            for desc_act in (False, True):
                for k, n, gs in SHAPES:
                    with self.subTest(dtype=dtype, desc_act=desc_act, K=k, N=n, gs=gs):
                        self._run_gptq(8, k, n, gs, dtype, desc_act, "", tp_size=2)


@unittest.skipIf(not is_xpu(), "XPU int4 dense UT requires an Intel XPU")
class TestXPUCompressedTensorsWNA16Kernel(CustomTestCase):
    """compressed-tensors WNA16 int4pack kernel numerics vs a pure-torch dequant."""

    def test_unpack_matches_compressed_tensors_packer(self):
        # Pin the format against the library that writes it, so the unpacker and
        # this file's packer cannot be self-consistently wrong together.
        try:
            from compressed_tensors.compressors.pack_quantized.helpers import (
                pack_to_int32,
            )
        except ImportError as exc:  # pragma: no cover - depends on installed version
            self.skipTest(f"compressed_tensors pack_to_int32 unavailable: {exc}")

        from sglang.srt.hardware_backend.xpu.quantization.int4pack_utils import (
            unpack_compressed_tensors_qweight,
        )

        torch.manual_seed(0)
        n, k = 64, 256
        w_int = torch.randint(-8, 8, (n, k), dtype=torch.int8)
        ref_packed = pack_to_int32(w_int, num_bits=4, packed_dim=1)
        self.assertEqual(tuple(ref_packed.shape), (n, k // 8))

        # uint4b8 stores every signed code biased by +8.
        biased = w_int.to(torch.int32) + 8
        codes = unpack_compressed_tensors_qweight(ref_packed.to(DEV))
        self.assertTrue(torch.equal(codes.cpu(), biased))
        self.assertTrue(
            torch.equal(_ct_pack_qweight(biased), ref_packed.to(torch.int32))
        )

    def test_wna16_numeric(self):
        for dtype in (torch.float16, torch.bfloat16):
            for m in M_VALUES:
                for k, n, gs in SHAPES:
                    with self.subTest(dtype=dtype, M=m, K=k, N=n, gs=gs):
                        self._run_wna16(m, k, n, gs, dtype)

    def test_wna16_bias(self):
        # xpu_int4pack_mm adds bias outside the GEMM; keep that path covered.
        for dtype in (torch.float16, torch.bfloat16):
            for k, n, gs in SHAPES:
                with self.subTest(dtype=dtype, K=k, N=n, gs=gs):
                    self._run_wna16(8, k, n, gs, dtype, with_bias=True)

    def _run_wna16(self, m, k, n, gs, dtype, with_bias=False):
        from sglang.srt.hardware_backend.xpu.quantization.compressed_tensors_kernels import (
            CompressedTensorsWNA16XPULinearKernel,
        )

        torch.manual_seed(0)
        codes = torch.randint(0, 16, (n, k), device=DEV)  # [N, K] uint4b8 codes
        scales = torch.rand(n, k // gs, device=DEV, dtype=dtype) * 0.05 + 0.005

        w_ref = (codes.to(dtype) - 8) * scales.repeat_interleave(gs, dim=1)
        x = torch.randn(m, k, device=DEV, dtype=dtype)
        bias = torch.randn(n, device=DEV, dtype=dtype) if with_bias else None
        ref = x @ w_ref.t()
        if bias is not None:
            ref = ref + bias

        layer = _make_layer()
        layer.weight_packed = torch.nn.Parameter(
            _ct_pack_qweight(codes), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)

        kernel = CompressedTensorsWNA16XPULinearKernel(
            group_size=gs, symmetric=True, has_g_idx=False
        )
        kernel.process_weights_after_loading(layer)
        out = kernel.apply(layer, x, bias)

        self.assertEqual(tuple(out.shape), (m, n))
        self.assertTrue(torch.isfinite(out).all())
        rel = (out - ref).abs().max().item() / ref.abs().max().item()
        self.assertLess(rel, REL_TOL[dtype], f"rel={rel:.2e}")

    def test_wna16_rejects_unsupported_configs(self):
        # The native XPU op cannot express these; fail loudly instead of
        # producing silently wrong weights.
        from sglang.srt.hardware_backend.xpu.quantization.compressed_tensors_kernels import (
            CompressedTensorsWNA16XPULinearKernel,
        )

        k, n, gs = 128, 64, 32
        # Only the config is under test; the checks fire before any weight is
        # replaced, so one layer is safe to reuse.
        layer = _make_layer()
        layer.weight_packed = torch.nn.Parameter(
            torch.zeros(n, k // 8, dtype=torch.int32, device=DEV), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(n, k // gs, device=DEV, dtype=torch.float16),
            requires_grad=False,
        )

        cases = (
            (dict(group_size=48, symmetric=True, has_g_idx=False), r"group_size"),
            (dict(group_size=-1, symmetric=True, has_g_idx=False), r"group_size"),
            (dict(group_size=gs, symmetric=False, has_g_idx=False), r"symmetric"),
            (
                dict(group_size=gs, symmetric=True, has_g_idx=True),
                r"activation reordering",
            ),
        )
        for kwargs, pattern in cases:
            with self.subTest(**kwargs):
                kernel = CompressedTensorsWNA16XPULinearKernel(**kwargs)
                with self.assertRaisesRegex(ValueError, pattern):
                    kernel.process_weights_after_loading(layer)


if __name__ == "__main__":
    unittest.main()
