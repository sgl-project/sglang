# SPDX-License-Identifier: Apache-2.0
"""SM80 (A100) M=1 FP8-E4M3 group-32 GEMV: contract, graph capture, Marlin wiring.

Run directly with unittest (a real SM80 GPU is required):
    python test/registered/kernels/ops/gemm/test_m1_gemv_sm80.py

The kernel under test is ``sglang.kernels.ops.gemm.dsv4_m1_gemv``; the second
class drives the production gate in ``marlin_utils_fp8`` end to end.

WHAT IS CHECKED
---------------
The reference is the exact contract Marlin's SM80 FP8 path implements, evaluated
in float64 on the CPU:

    w_eff[n, k] = bf16( bf16(fp8_value[n, k]) * bf16(scale[n, k // 32]) )
    out[n]      = bf16( sum_k fp64( w_eff[n, k] * x[k] ) )

where ``fp8_value * 2**-120`` is rebuilt from the raw byte with the same bit
shift the kernel uses (``((b << 8) & 0x8000) | ((b << 4) & 0x07F0)`` then a
bf16 bitcast) and the single ``.to(bfloat16)`` is the one contract rounding.
This is the oracle of the qualified bench
``test/manual/layers/quantization/bench_fp8_sm80_m1_gemv.py``.

Every case must satisfy the same allclose bounds that qualified bench uses
(rtol 1.5e-2, atol 4e-3). On top of that:

  * well-conditioned fixtures must land within one bf16 ulp of the largest
    output magnitude. The kernel and the oracle associate the fp32 additions
    differently, so "bitwise equal to the oracle" is *reported* per case rather
    than required;
  * fixtures whose folded product lands in the fp32 subnormal range (bf16
    subnormal scales) are allowed the derived FTZ magnitude: every flushed term
    is at most one bf16 step at the top of that range, so the difference is
    bounded by K * 2**-126 * max|x|;
  * the cancellation fixture (exact +/- weight pairs inside one 32-group, i.e.
    an exact-zero result in real arithmetic) is judged by the qualified bounds
    only, and reports the fp32 residual, which is the point of that fixture.

A strict failure on ``subnormal_payload`` or on the codebook fixtures means the
sm80 bf16 multiply is being promoted to fp32 with FTZ, which flushes the folded
payload of E4M3 subnormal codes (v * 2**-120 is an fp32 subnormal for
|v| < 2**-6). That is the one numerical mechanism this file cannot settle
without the GPU, and it is probed deliberately instead of being tolerated.

No tolerance here was chosen to make a specific number pass.
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
import unittest

import torch

# The kernel module imports Triton, so it is only imported after the SM80 gate:
# a CPU or non-A100 host must be able to collect and skip this file.
IS_SM80 = (
    torch.version.hip is None
    and torch.cuda.is_available()
    and torch.cuda.get_device_capability() == (8, 0)
)
if IS_SM80:
    from sglang.kernels.ops.gemm.dsv4_m1_gemv import (
        dsv41_m1_gemv,
        dsv41_m1_gemv_config,
        dsv41_m1_gemv_eligible,
    )

GROUP = 32
# The five dense DeepSeek-V4.1-Flash TP8 projections. (5120, 320) is the
# shared-expert down projection, whose K arrives padded 288 -> 320 from the
# existing SM80 Marlin K-tail padding: the kernel always matches the PADDED K.
REAL_SHAPES = ((1792, 5120), (4096, 1280), (5120, 1024), (576, 5120), (5120, 320))
PADDED_K_SHAPE = (5120, 320)
# Divides the generic (4, 64, 4) tiling and is not one of the measured shapes.
FALLBACK_SHAPE = (256, 192)
# The table the kernel actually ships, pinned so an unintended edit is caught.
#
# It is NOT the fastest isolated configuration per shape. Those are, in the
# same (NCOL, BKK, num_warps) form, (4, 1024, 8) / (4, 64, 4) / (16, 128, 8) /
# (4, 1024, 8) / (16, 64, 4); the shipped table takes the smallest accumulator
# among the configurations within a few percent of that best, because the
# accumulator is held in registers for the whole K sweep and its size decides how
# much room is left for the concurrent allreduce blocks. See the selection
# rationale next to _EXACT_CONFIGS in dsv4_m1_gemv.py.
MEASURED_CONFIGS = {
    (1792, 5120): (4, 128, 8),
    (4096, 1280): (4, 64, 4),
    (5120, 1024): (8, 64, 8),
    (576, 5120): (4, 128, 8),
    (5120, 320): (8, 64, 8),
}
GENERIC_CONFIG = (4, 64, 4)
QUALIFIED_RTOL = 1.5e-2
QUALIFIED_ATOL = 4e-3
FP32_SUBNORMAL_BOUND = 2.0**-126
EXPERIMENTAL_FLAG = "SGLANG_DSV41_SM80_EXPERIMENTAL"
GEMV_FLAG = "SGLANG_DSV41_SM80_M1_GEMV"

_FINITE_CODES = torch.arange(256, dtype=torch.uint8)
# 0x7F and 0xFF are the only E4M3 NaN encodings; no checkpoint carries them.
_FINITE_CODES = _FINITE_CODES[(_FINITE_CODES & 0x7F) != 0x7F]
assert _FINITE_CODES.numel() == 254
# E4M3 subnormals (+/-): their folded payload is an fp32 subnormal, so this is
# the fixture that isolates a would-be FTZ flush of the payload.
_SUBNORMAL_CODES = torch.tensor(
    [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
     0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87],
    dtype=torch.uint8,
)


def _is_sm80() -> bool:
    return IS_SM80


@contextlib.contextmanager
def _env(**values):
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# ---------------------------------------------------------------------------
# Exact decode mirror, float64 oracle, metrics
# ---------------------------------------------------------------------------


def _fp8_payload_to_bf16(raw: torch.Tensor) -> torch.Tensor:
    """Torch mirror of the kernel decode: fp8_value * 2**-120, bf16, bit-exact."""
    b = raw.view(torch.uint8).to(torch.int32)
    bits = ((b << 8) & 0x8000) | ((b << 4) & 0x07F0)
    return bits.to(torch.int16).view(torch.bfloat16)


def _contract_reference(
    w8: torch.Tensor, s: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """float64 CPU oracle of the Marlin group-32 contract.

    w8: (N, K) uint8 raw E4M3; s: (N, K // 32) bf16 plain scales; x: (K,) bf16.
    Returns (N,) bf16.
    """
    w8 = w8.detach().cpu()
    s = s.detach().cpu()
    x = x.detach().cpu()
    n, k = w8.shape
    ks = s.shape[1]
    assert k == ks * GROUP, (k, ks)
    fp8_value = _fp8_payload_to_bf16(w8).double() * float(2.0**120)
    scale = s.double()[:, :, None].expand(n, ks, GROUP).reshape(n, k)
    # Both operands are bf16, so the float64 product is exact and this single
    # .to(bfloat16) is the one contract rounding.
    w_eff = (fp8_value * scale).to(torch.bfloat16).double()
    return (w_eff @ x.double()).to(torch.bfloat16)


def _metrics(got: torch.Tensor, ref: torch.Tensor) -> dict:
    got_f = got.detach().float().cpu()
    ref_f = ref.detach().float().cpu()
    ulp = float(ref_f.abs().max().item()) * 2.0**-8
    return {
        "max_abs": float((got_f - ref_f).abs().max().item()),
        "one_bf16_ulp_at_max": ulp,
        # -0.0 and +0.0 count as different bit patterns; this is a reported
        # count, not a pass criterion.
        "bitwise_equal": int(
            (
                got_f.to(torch.bfloat16).view(torch.int16)
                == ref_f.to(torch.bfloat16).view(torch.int16)
            )
            .sum()
            .item()
        ),
        "total": int(ref.numel()),
        "qualified_allclose": bool(
            torch.allclose(got_f, ref_f, rtol=QUALIFIED_RTOL, atol=QUALIFIED_ATOL)
        ),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _codebook_weight(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """(N, K) uint8 covering every finite E4M3 code, plus random finite bytes."""
    flat = torch.randint(0, 256, (n * k,), generator=generator).to(torch.uint8)
    flat[(flat & 0x7F) == 0x7F] = 0x7E
    flat[: _FINITE_CODES.numel()] = _FINITE_CODES
    covered = torch.unique(flat[: _FINITE_CODES.numel()]).numel()
    assert covered == _FINITE_CODES.numel(), f"codebook coverage {covered}"
    return flat.view(n, k).contiguous()


def _value_weight(
    n: int, k: int, generator: torch.Generator, spread: float = 0.25
) -> torch.Tensor:
    """(N, K) uint8 from quantized Gaussian values."""
    values = torch.randn(n, k, generator=generator) * spread
    return values.to(torch.float8_e4m3fn).view(torch.uint8).contiguous()


def _subnormal_weight(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """(N, K) uint8 drawn only from the E4M3 subnormal codes."""
    idx = torch.randint(0, _SUBNORMAL_CODES.numel(), (n, k), generator=generator)
    return _SUBNORMAL_CODES[idx].contiguous()


def _paired_weight(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """(N, K) uint8 where w[:, 2i] == -w[:, 2i+1], and 2i/2i+1 share a 32-group."""
    half = torch.randn(n, k // 2, generator=generator) * 0.25
    w = torch.empty(n, k)
    w[:, 0::2] = half
    w[:, 1::2] = -half
    converted = w.to(torch.float8_e4m3fn).view(torch.uint8)
    assert torch.equal(converted[:, 0::2], converted[:, 1::2] ^ 0x80), "pairing lost"
    return converted.contiguous()


def _scales_pow2(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """Exact powers of two, the shape a ue8m0 block checkpoint has."""
    exponents = torch.randint(-6, 1, (n, k // GROUP), generator=generator)
    return torch.exp2(exponents.float()).to(torch.bfloat16)


def _scales_general(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    values = 0.01 + torch.rand(n, k // GROUP, generator=generator) * 1.99
    return values.to(torch.bfloat16)


def _scales_mixed(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    exponents = torch.randint(-8, 4, (n, k // GROUP), generator=generator).float()
    jitter = 1.0 + torch.rand(n, k // GROUP, generator=generator)
    return (torch.exp2(exponents) * jitter).to(torch.bfloat16)


def _scales_huge(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """Large scales that still keep bf16(scale * 2**120) finite (|scale| < 2**8)."""
    choices = torch.tensor([127.0, 200.0, 255.0], dtype=torch.bfloat16)
    idx = torch.randint(0, choices.numel(), (n, k // GROUP), generator=generator)
    scales = choices[idx]
    assert scales.dtype == torch.bfloat16
    assert float(scales.abs().max()) < 2.0**8
    return scales


def _scales_subnormal(n: int, k: int, generator: torch.Generator) -> torch.Tensor:
    """bf16 subnormal scales: folded products land in the fp32 subnormal range."""
    exponents = torch.randint(-133, -120, (n, k // GROUP), generator=generator)
    scales = torch.exp2(exponents.float()).to(torch.bfloat16)
    assert float(scales.abs().max()) < 2.0**-120
    return scales


def _x_random(k: int, generator: torch.Generator, spread: float = 0.05) -> torch.Tensor:
    return (torch.randn(k, generator=generator) * spread).to(torch.bfloat16)


def _fixture(kind: str, n: int, k: int, seed: int):
    """Build (w8 uint8 (N, K), s bf16 (N, K/32), x bf16 (K,), strictness)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if kind == "codebook_pow2":
        return (
            _codebook_weight(n, k, generator),
            _scales_pow2(n, k, generator),
            _x_random(k, generator),
            "strict",
        )
    if kind == "codebook_general":
        return (
            _codebook_weight(n, k, generator),
            _scales_general(n, k, generator),
            _x_random(k, generator, 0.02),
            "strict",
        )
    if kind == "random_bytes":
        return (
            _codebook_weight(n, k, generator),
            _scales_mixed(n, k, generator),
            _x_random(k, generator),
            "strict",
        )
    if kind == "values_pow2":
        return (
            _value_weight(n, k, generator),
            _scales_pow2(n, k, generator),
            _x_random(k, generator),
            "strict",
        )
    if kind == "subnormal_payload":
        return (
            _subnormal_weight(n, k, generator),
            _scales_general(n, k, generator),
            _x_random(k, generator),
            "strict",
        )
    if kind == "huge_scales":
        return (
            _value_weight(n, k, generator),
            _scales_huge(n, k, generator),
            _x_random(k, generator, 0.002),
            "strict",
        )
    if kind == "subnormal_scales":
        return (
            _value_weight(n, k, generator),
            _scales_subnormal(n, k, generator),
            _x_random(k, generator),
            "subnormal",
        )
    if kind == "cancelling_pairs":
        x = torch.full((k,), 0.125, dtype=torch.bfloat16)
        return (
            _paired_weight(n, k, generator),
            _scales_general(n, k, generator),
            x,
            "cancellation",
        )
    raise AssertionError(f"unknown fixture {kind}")


FIXTURE_KINDS = (
    "codebook_pow2",
    "codebook_general",
    "random_bytes",
    "values_pow2",
    "subnormal_payload",
    "huge_scales",
    "subnormal_scales",
    "cancelling_pairs",
)


def _graph_time(call, inner: int = 20, rounds: int = 5) -> float:
    """Median microseconds per call, CUDA-graph replayed on a non-default stream."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            call()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / inner)
    del graph
    return statistics.median(samples)


class M1GemvCaseMixin:
    def check_case(
        self,
        name: str,
        got: torch.Tensor,
        ref: torch.Tensor,
        strictness: str,
        *,
        k: int | None = None,
        x_max: float | None = None,
        extra: dict | None = None,
    ) -> dict:
        stats = _metrics(got, ref)
        record = {"case": name, "strictness": strictness, **stats}
        if strictness == "subnormal":
            # Each flushed term is at most one bf16 step at the top of the fp32
            # subnormal range; the accumulation is fp32, so this is the derived
            # worst case, not a tuned constant.
            assert k is not None and x_max is not None
            bound = k * FP32_SUBNORMAL_BOUND * x_max
            record["ftz_bound"] = bound
        elif strictness == "strict":
            bound = max(stats["one_bf16_ulp_at_max"], FP32_SUBNORMAL_BOUND)
        else:
            self.assertEqual(strictness, "cancellation", name)
            bound = None
        if bound is not None:
            record["bound"] = bound
        if extra:
            record.update(extra)
        print(json.dumps(record), flush=True)
        message = f"{name}: {json.dumps(record)}"
        self.assertTrue(stats["qualified_allclose"], message)
        if bound is not None:
            self.assertLessEqual(stats["max_abs"], bound, message)
        return stats


@unittest.skipUnless(_is_sm80(), "requires CUDA SM80 (8, 0)")
class TestM1GemvContract(M1GemvCaseMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The hand-rolled pipeline below multiplies in fp32; TF32 would change it.
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_measured_config_table(self):
        for shape, expected in MEASURED_CONFIGS.items():
            with self.subTest(shape=shape):
                self.assertEqual(dsv41_m1_gemv_config(*shape), expected)
                self.assertTrue(dsv41_m1_gemv_eligible(*shape))

    def test_generic_fallback_and_declines(self):
        self.assertEqual(dsv41_m1_gemv_config(*FALLBACK_SHAPE), GENERIC_CONFIG)
        self.assertTrue(dsv41_m1_gemv_eligible(*FALLBACK_SHAPE))
        # Divisible by ncol=4 but not by bkk=64, and a shape failing both.
        for shape in ((100, 96), (103, 64), (11, 4096)):
            with self.subTest(shape=shape):
                self.assertFalse(dsv41_m1_gemv_eligible(*shape))
                self.assertIsNone(dsv41_m1_gemv_config(*shape))
                with self.assertRaises(ValueError):
                    dsv41_m1_gemv(
                        torch.zeros(shape, dtype=torch.uint8, device="cuda"),
                        torch.zeros(
                            shape[0],
                            shape[1] // GROUP,
                            dtype=torch.bfloat16,
                            device="cuda",
                        ),
                        torch.zeros(shape[1], dtype=torch.bfloat16, device="cuda"),
                    )

    def test_contract_all_shapes_and_fixtures(self):
        for n, k in REAL_SHAPES + (FALLBACK_SHAPE,):
            for index, kind in enumerate(FIXTURE_KINDS):
                with self.subTest(shape=(n, k), fixture=kind):
                    # Deterministic per-case seed: a failure is reproducible.
                    w8, s, x, strictness = _fixture(
                        kind, n, k, seed=n * 31 + k + index * 7919
                    )
                    w = w8.to(device="cuda")
                    s_cuda = s.to(device="cuda")
                    x_cuda = x.to(device="cuda")
                    out = dsv41_m1_gemv(w, s_cuda, x_cuda)
                    self.assertEqual(tuple(out.shape), (n,))
                    self.assertEqual(out.dtype, torch.bfloat16)
                    ref = _contract_reference(w8, s, x)
                    self.check_case(
                        f"contract:{n}x{k}:{kind}",
                        out,
                        ref,
                        strictness,
                        k=k,
                        x_max=float(x.abs().max().item()),
                        extra={"config": dsv41_m1_gemv_config(n, k)},
                    )

    def test_contract_matches_the_explicit_2d_pipeline(self):
        """A hand-rolled torch pipeline of the same formula must agree to 1 ulp."""
        n, k = 5120, 1024
        generator = torch.Generator(device="cpu").manual_seed(2024)
        w8 = _codebook_weight(n, k, generator)
        s = _scales_general(n, k, generator)
        x = _x_random(k, generator)
        w = w8.to(device="cuda")
        out = dsv41_m1_gemv(w, s.to(device="cuda"), x.to(device="cuda"))

        # (N, K) payload (fp8_value * 2**-120, bf16) times the 2**120-folded
        # bf16 scale: one bf16 multiply, then an fp32 sum of the bf16 products.
        two120 = torch.tensor(2.0**120, dtype=torch.bfloat16, device="cuda")
        payload = _fp8_payload_to_bf16(w8).to(device="cuda")
        scale = s.to(device="cuda").repeat_interleave(GROUP, dim=1)
        wk = (payload * (scale * two120)).to(torch.bfloat16)
        direct = (wk.float() @ x.to(device="cuda").float()).to(torch.bfloat16)
        stats = _metrics(out, direct)
        print(json.dumps({"case": "explicit_2d_pipeline", **stats}), flush=True)
        # Both sides round each product to bf16 first and sum in fp32; only the
        # association differs, so allow the same one-ulp bound as the contract.
        self.assertLessEqual(
            stats["max_abs"], max(stats["one_bf16_ulp_at_max"], FP32_SUBNORMAL_BOUND)
        )

    def test_cuda_graph_capture_on_non_default_stream_and_replay(self):
        for n, k in ((1792, 5120), PADDED_K_SHAPE):
            with self.subTest(shape=(n, k)):
                generator = torch.Generator(device="cpu").manual_seed(7 + k)
                w8 = _codebook_weight(n, k, generator)
                s = _scales_general(n, k, generator)
                x = _x_random(k, generator)
                w = w8.to(device="cuda")
                s_cuda = s.to(device="cuda")
                x_cuda = x.to(device="cuda")
                out = torch.empty(n, dtype=torch.bfloat16, device="cuda")

                # Warm on the non-default stream, then capture there: capturing
                # on the default stream raises.
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        dsv41_m1_gemv(w, s_cuda, x_cuda, out=out)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                eager = out.clone()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    dsv41_m1_gemv(w, s_cuda, x_cuda, out=out)
                graph.replay()
                torch.cuda.synchronize()

                self.assertTrue(
                    torch.equal(out.view(torch.int16), eager.view(torch.int16)),
                    "graph replay does not reproduce the eager result bitwise",
                )
                self.check_case(
                    f"graph_first_capture:{n}x{k}",
                    out,
                    _contract_reference(w8, s, x),
                    "strict",
                )

                saved_x = x_cuda.clone()
                saved_s = s_cuda.clone()
                saved_w = w.clone()

                def _neg_x():
                    x_cuda.neg_()

                def _restore_x():
                    x_cuda.copy_(saved_x)

                def _halve_scales():
                    s_cuda.mul_(0.5)

                def _restore_scales():
                    s_cuda.copy_(saved_s)

                def _flip_weight_sign():
                    w.bitwise_xor_(0x80)

                def _restore_weight():
                    w.copy_(saved_w)

                mutations = (
                    ("x_neg", _neg_x, _restore_x),
                    ("scale_half", _halve_scales, _restore_scales),
                    ("weight_sign", _flip_weight_sign, _restore_weight),
                )
                for name, mutate, revert in mutations:
                    mutate()
                    graph.replay()
                    torch.cuda.synchronize()
                    self.check_case(
                        f"graph_replay_after_{name}:{n}x{k}",
                        out,
                        _contract_reference(
                            w.cpu(), s_cuda.cpu(), x_cuda.cpu().reshape(-1)
                        ),
                        "strict",
                    )
                    revert()

    def test_benchmark_kernel_only(self):
        """Informational: median us and effective GB/s per measured shape."""
        for n, k in REAL_SHAPES:
            with self.subTest(shape=(n, k)):
                generator = torch.Generator(device="cpu").manual_seed(k)
                w = _codebook_weight(n, k, generator).to(device="cuda")
                s = _scales_general(n, k, generator).to(device="cuda")
                x = _x_random(k, generator).to(device="cuda")
                out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
                us = _graph_time(lambda: dsv41_m1_gemv(w, s, x, out=out))
                bytes_moved = n * k + n * (k // GROUP) * 2
                print(
                    json.dumps(
                        {
                            "case": "gemv_bench",
                            "N": n,
                            "K": k,
                            "median_us": us,
                            "effective_GB/s": bytes_moved / (us * 1e-6) / 1e9,
                            "config": dsv41_m1_gemv_config(n, k),
                            "note": "untested here: compare against Marlin on the host",
                        }
                    ),
                    flush=True,
                )


@unittest.skipUnless(_is_sm80(), "requires CUDA SM80 (8, 0)")
class TestM1GemvMarlinIntegration(M1GemvCaseMixin, unittest.TestCase):
    """Drives the real linear-method lifecycle, the stash and the apply gate."""

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod

        cls.fp8_config = Fp8Config
        cls.fp8_method = Fp8LinearMethod
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        cls.addClassCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", old_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False

    def _build_layer(self, n, k, *, seed):
        """A Marlin SM80 group-32 layer with a codebook weight and general scales."""
        config = self.fp8_config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[GROUP, GROUP],
            scale_fmt="ue8m0",
        )
        method = self.fp8_method(config)
        self.assertTrue(method.use_marlin)
        self.assertTrue(method.block_quant)
        layer = torch.nn.Module()
        layer.tp_group = None
        with torch.device("cuda"):
            method.create_weights(layer, k, [n], k, n, torch.bfloat16)
        self.assertEqual(tuple(layer.weight.shape), (n, k))
        self.assertEqual(
            tuple(layer.weight_scale_inv.shape), (n // GROUP, k // GROUP)
        )
        self.assertEqual(layer.weight_scale_inv.dtype, torch.float32)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        w8 = _codebook_weight(n, k, generator)
        scales = _scales_general(n // GROUP, k, generator)
        self.assertEqual(tuple(scales.shape), (n // GROUP, k // GROUP))
        with torch.no_grad():
            layer.weight.copy_(w8.view(torch.float8_e4m3fn).to(device="cuda"))
            layer.weight_scale_inv.copy_(scales.float().to(device="cuda"))
        return method, layer, w8, scales

    def _expected_stash(self, n, k, w8, scales, padded_k):
        expected_w = torch.zeros(n, padded_k, dtype=torch.uint8)
        expected_w[:, :k] = w8
        expected_s = torch.ones(n, padded_k // GROUP, dtype=torch.bfloat16)
        expanded = scales.to(torch.bfloat16).repeat_interleave(GROUP, dim=0)[:n]
        expected_s[:, : k // GROUP] = expanded
        return expected_w, expected_s

    def _check_stash(self, layer, n, k, w8, scales, padded_k):
        stash_w = layer.dsv41_m1_w
        stash_s = layer.dsv41_m1_s
        self.assertTrue(stash_w.dtype == torch.uint8)
        self.assertTrue(stash_w.is_contiguous())
        self.assertTrue(stash_s.dtype == torch.bfloat16)
        self.assertTrue(stash_s.is_contiguous())
        self.assertEqual(tuple(stash_w.shape), (n, padded_k))
        self.assertEqual(tuple(stash_s.shape), (n, padded_k // GROUP))
        expected_w, expected_s = self._expected_stash(n, k, w8, scales, padded_k)
        self.assertTrue(torch.equal(stash_w.cpu(), expected_w), "stashed bytes differ")
        self.assertTrue(torch.equal(stash_s.cpu(), expected_s), "stashed scales differ")
        # The repacked parameter must carry the stash: that is the only channel
        # apply_fp8_marlin_linear has.
        self.assertIs(layer.weight.dsv41_m1_w, stash_w)
        self.assertIs(layer.weight.dsv41_m1_s, stash_s)
        return stash_w, stash_s

    def test_gate_is_default_off(self):
        method, layer, _, _ = self._build_layer(1792, 5120, seed=3)
        with _env(**{EXPERIMENTAL_FLAG: "1"}):
            os.environ.pop(GEMV_FLAG, None)
            method.process_weights_after_loading(layer)
        self.assertFalse(hasattr(layer, "dsv41_m1_w"))
        self.assertFalse(hasattr(layer, "dsv41_m1_s"))
        self.assertIsNone(getattr(layer.weight, "dsv41_m1_w", None))
        self.assertIsNone(getattr(layer.weight, "dsv41_m1_s", None))

    def test_gate_needs_both_flags(self):
        for experimental, gemv in (("1", "0"), ("0", "1"), (None, "1")):
            with self.subTest(experimental=experimental, gemv=gemv):
                method, layer, _, _ = self._build_layer(1792, 5120, seed=5)
                env = {GEMV_FLAG: gemv}
                if experimental is not None:
                    env[EXPERIMENTAL_FLAG] = experimental
                with _env(**env):
                    if experimental is None:
                        os.environ.pop(EXPERIMENTAL_FLAG, None)
                    method.process_weights_after_loading(layer)
                self.assertFalse(hasattr(layer, "dsv41_m1_w"))

    def test_stash_and_apply_end_to_end(self):
        # (n, k, padded_k): the first shape is unpadded, the second is the
        # shared-expert down projection whose K the Marlin path pads 288 -> 320.
        for n, k, padded_k in ((1792, 5120, 5120), (5120, 288, 320)):
            with self.subTest(shape=(n, k, padded_k)):
                method, layer, w8, scales = self._build_layer(n, k, seed=11 + k)
                with _env(**{EXPERIMENTAL_FLAG: "1", GEMV_FLAG: "1"}):
                    method.process_weights_after_loading(layer)
                    self.assertTrue(hasattr(layer, "dsv41_m1_w"))
                    stash_w, stash_s = self._check_stash(
                        layer, n, k, w8, scales, padded_k
                    )

                    generator = torch.Generator(device="cpu").manual_seed(31 + k)
                    x_flat = _x_random(k, generator)
                    x = x_flat.reshape(1, k).to(device="cuda")
                    out = torch.ops.sglang.apply_fp8_marlin_linear(
                        input=x,
                        weight=layer.weight,
                        weight_scale=layer.weight_scale,
                        workspace=layer.workspace,
                        size_n=n,
                        size_k=k,
                        bias=None,
                    )
                    self.assertEqual(tuple(out.shape), (1, n))

                    # Independent reference: checkpoint fixtures + the same K
                    # padding rule (zero bytes, neutral scale, zero activations).
                    ref_w, ref_s = self._expected_stash(n, k, w8, scales, padded_k)
                    ref_x = torch.zeros(padded_k, dtype=torch.bfloat16)
                    ref_x[:k] = x_flat
                    ref = _contract_reference(ref_w, ref_s, ref_x)
                    stats = self.check_case(
                        f"apply:{n}x{k}->{padded_k}", out.reshape(-1), ref, "strict"
                    )

                    # The gate really took the M1 path: a direct kernel call on
                    # the stash reproduces it bitwise.
                    direct = dsv41_m1_gemv(stash_w, stash_s, ref_x.to(device="cuda"))
                    same_bits = torch.equal(
                        direct.view(torch.int16), out.reshape(-1).view(torch.int16)
                    )
                    self.assertTrue(same_bits, "apply did not run the stashed M1 GEMV")

                    # Without the stash (same packed bytes, no attributes) the
                    # unchanged Marlin path must produce the same contract.
                    marlin_weight = layer.weight.detach().clone()
                    marlin_out = torch.ops.sglang.apply_fp8_marlin_linear(
                        input=x,
                        weight=marlin_weight,
                        weight_scale=layer.weight_scale,
                        workspace=layer.workspace,
                        size_n=n,
                        size_k=k,
                        bias=None,
                    )
                    marlin_stats = _metrics(marlin_out.reshape(-1), ref)
                    cross_stats = _metrics(out.reshape(-1), marlin_out.reshape(-1))
                    print(
                        json.dumps(
                            {
                                "case": f"marlin_cross:{n}x{k}->{padded_k}",
                                "gemv_vs_ref": stats["max_abs"],
                                "marlin_vs_ref": marlin_stats["max_abs"],
                                "gemv_vs_marlin": cross_stats["max_abs"],
                                "gemv_vs_marlin_bitwise": cross_stats[
                                    "bitwise_equal"
                                ],
                            }
                        ),
                        flush=True,
                    )
                    self.assertTrue(marlin_stats["qualified_allclose"])

    def test_benchmark_gemv_vs_marlin(self):
        """Informational: the same call, graph-timed, both paths."""
        for n, k in ((1792, 5120), (5120, 1024)):
            with self.subTest(shape=(n, k)):
                method, layer, _, _ = self._build_layer(n, k, seed=41 + k)
                with _env(**{EXPERIMENTAL_FLAG: "1", GEMV_FLAG: "1"}):
                    method.process_weights_after_loading(layer)
                    generator = torch.Generator(device="cpu").manual_seed(53 + k)
                    x = _x_random(k, generator).reshape(1, k).to(device="cuda")
                    out = torch.empty(n, dtype=torch.bfloat16, device="cuda")
                    stash_w = layer.dsv41_m1_w
                    stash_s = layer.dsv41_m1_s
                    x_flat = x.reshape(-1)

                    gemv_us = _graph_time(
                        lambda: dsv41_m1_gemv(stash_w, stash_s, x_flat, out=out)
                    )
                    marlin_weight = layer.weight.detach().clone()
                    marlin_us = _graph_time(
                        lambda: torch.ops.sglang.apply_fp8_marlin_linear(
                            input=x,
                            weight=marlin_weight,
                            weight_scale=layer.weight_scale,
                            workspace=layer.workspace,
                            size_n=n,
                            size_k=k,
                            bias=None,
                        )
                    )
                    bytes_moved = n * k + n * (k // GROUP) * 2
                    print(
                        json.dumps(
                            {
                                "case": "bench_gemv_vs_marlin",
                                "N": n,
                                "K": k,
                                "gemv_us": gemv_us,
                                "marlin_us": marlin_us,
                                "speedup": marlin_us / gemv_us,
                                "gemv_GB/s": bytes_moved / (gemv_us * 1e-6) / 1e9,
                                "marlin_GB/s": bytes_moved / (marlin_us * 1e-6) / 1e9,
                                "note": "informational only; no speed assertion",
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
