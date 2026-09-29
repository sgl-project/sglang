"""Correctness, graph replay, and performance tests for the gfx950 kernel."""

import unittest
from itertools import product

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")

_ROWS = (1, 2, 4, 8, 10, 16, 32, 40, 64, 96, 128)

_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    try:
        import triton
        from aiter.ops.cache import indexer_k_quant_and_cache

        from sglang.kernels.ops.attention.dsa.hip_gfx950 import (
            full_indexer_prepare,
            is_full_indexer_prepare_available,
        )
        from sglang.kernels.ops.attention.dsa.hip_gfx950.gluon.generic import (
            indexer_prepare as small_m_indexer_prepare,
        )
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant

        _RUNNABLE = is_full_indexer_prepare_available()
    except Exception:
        _RUNNABLE = False


def _rotate_reference(
    x: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int = 64,
    is_neox_style: bool = False,
) -> torch.Tensor:
    half = rope_dim // 2
    cos = cos_sin[positions, :half]
    sin = cos_sin[positions, half:]
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    rotated = x.clone()
    if is_neox_style:
        first = x[..., :half]
        second = x[..., half:rope_dim]
        rotated[..., :half] = first * cos - second * sin
        rotated[..., half:rope_dim] = second * cos + first * sin
    else:
        pairs = x[..., :rope_dim].reshape(*x.shape[:-1], half, 2)
        even, odd = pairs[..., 0], pairs[..., 1]
        rotated_pairs = rotated[..., :rope_dim].reshape(*x.shape[:-1], half, 2)
        rotated_pairs[..., 0] = even * cos - odd * sin
        rotated_pairs[..., 1] = odd * cos + even * sin
    return rotated.to(torch.bfloat16)


def _make_cos_sin(rope_dim: int) -> torch.Tensor:
    positions = torch.arange(256, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, rope_dim, 2, device="cuda", dtype=torch.float32) / rope_dim)
    )
    freqs = torch.outer(positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)


def _randn(*shape: int, scale: float = 0.01) -> torch.Tensor:
    return torch.randn(*shape, device="cuda", dtype=torch.bfloat16).mul_(scale)


def _snr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    signal = expected.square().sum()
    noise = (actual - expected).square().sum().clamp_min(1e-30)
    return float((10 * torch.log10(signal / noise)).item())


def _decode_cache_row(cache: torch.Tensor, slot: int) -> torch.Tensor:
    page_size = cache.shape[1]
    page, offset = divmod(slot, page_size)
    dims = torch.arange(128, device=cache.device)
    byte_offsets = (
        page * page_size * 132
        + (offset // 16) * 2048
        + (dims // 16) * 256
        + (offset % 16) * 16
        + dims % 16
    )
    raw = cache.view(-1)
    quant = raw[byte_offsets].contiguous().view(torch.float8_e4m3fn).float()
    scale_offset = page * page_size * 132 + page_size * 128 + offset * 4
    scale = raw[scale_offset : scale_offset + 4].contiguous().view(torch.float32)
    return quant * scale


@unittest.skipUnless(
    _RUNNABLE, "requires HIP gfx950 and Triton >= 3.5 with Gluon CDNA4 support"
)
class TestFullIndexerPrepareGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.wq = _randn(4096, 2048)
        cls.wk = _randn(128, 6144)
        cls.wgate = _randn(32, 6144)
        cls.gamma = _randn(128, scale=0.05).add_(1)
        cls.beta = _randn(128)
        cls.cos_sin = _make_cos_sin(64)

    def _inputs(
        self,
        rows: int,
        hidden_size: int = 6144,
        q_lora_rank: int = 2048,
        page_size: int = 64,
    ):
        x = _randn(rows, hidden_size, scale=0.1)
        q_lora = _randn(rows, q_lora_rank, scale=0.1)
        positions = (
            torch.arange(rows, device="cuda", dtype=torch.int64) * 17 + 3
        ) % 256
        slots = torch.arange(rows, device="cuda", dtype=torch.int64) * 17
        cache = torch.zeros(
            (int(slots[-1]) // page_size) + 1,
            page_size,
            132,
            device="cuda",
            dtype=torch.uint8,
        )
        return x, q_lora, positions, slots, cache

    def _native_prepare(
        self,
        inputs,
        model_weights=None,
        norm=None,
        cos_sin=None,
        eps: float = 1e-6,
        rope_dim: int = 64,
        is_neox_style: bool = False,
    ):
        x, q_lora, positions, slots, cache = inputs
        wq, wk, wgate = model_weights or (self.wq, self.wk, self.wgate)
        gamma, beta = norm or (self.gamma, self.beta)
        cos_sin = self.cos_sin if cos_sin is None else cos_sin
        rows = x.shape[0]
        heads = wgate.shape[0]
        q = F.linear(q_lora, wq).reshape(rows, heads, 128)
        key = F.linear(x, wk)
        key = F.layer_norm(key.float(), (128,), gamma.float(), beta.float(), eps).to(
            torch.bfloat16
        )
        q = _rotate_reference(q, cos_sin, positions, rope_dim, is_neox_style)
        key = _rotate_reference(key, cos_sin, positions, rope_dim, is_neox_style)
        q_fp8, q_scale = act_quant(q, 128, "ue8m0")
        indexer_k_quant_and_cache(
            key,
            cache.view(torch.float8_e4m3fn),
            slots,
            128,
            "ue8m0",
            preshuffle=True,
        )
        gate = F.linear(x, wgate)
        weights = gate.float() * (heads**-0.5)
        weights = weights.unsqueeze(-1) * q_scale * (128**-0.5)
        return q_fp8, weights

    def _weights(self, heads: int, q_lora_rank: int, hidden_size: int):
        if (heads, q_lora_rank, hidden_size) == (32, 2048, 6144):
            return self.wq, self.wk, self.wgate
        return (
            _randn(heads * 128, q_lora_rank),
            _randn(128, hidden_size),
            _randn(heads, hidden_size),
        )

    def _prepare(
        self,
        inputs,
        model_weights=None,
        norm=None,
        cos_sin=None,
        eps: float = 1e-6,
        rope_dim: int = 64,
        is_neox_style: bool = False,
        kernel=None,
    ):
        x, q_lora, positions, slots, cache = inputs
        model_weights = model_weights or (self.wq, self.wk, self.wgate)
        norm = norm or (self.gamma, self.beta)
        cos_sin = self.cos_sin if cos_sin is None else cos_sin
        kernel = full_indexer_prepare if kernel is None else kernel
        return kernel(
            x,
            q_lora,
            *model_weights,
            *norm,
            cos_sin,
            positions,
            slots,
            cache,
            eps=eps,
            rope_dim=rope_dim,
            is_neox_style=is_neox_style,
        )

    def _benchmark(
        self,
        inputs,
        model_weights=None,
        norm=None,
        cos_sin=None,
        eps: float = 1e-6,
        rope_dim: int = 64,
        is_neox_style: bool = False,
        warmup: int = 50,
        rep: int = 100,
        compare_generic: bool = False,
    ):
        kwargs = dict(
            model_weights=model_weights,
            norm=norm,
            cos_sin=cos_sin,
            eps=eps,
            rope_dim=rope_dim,
            is_neox_style=is_neox_style,
        )
        fused_ms = triton.testing.do_bench(
            lambda: self._prepare(inputs, **kwargs), warmup=warmup, rep=rep
        )
        native_ms = triton.testing.do_bench(
            lambda: self._native_prepare(inputs, **kwargs), warmup=warmup, rep=rep
        )
        generic_ms = None
        if compare_generic:
            generic_ms = triton.testing.do_bench(
                lambda: self._prepare(inputs, **kwargs, kernel=small_m_indexer_prepare),
                warmup=warmup,
                rep=rep,
            )

        x, q_lora, _, _, cache = inputs
        heads = (model_weights or (self.wq, self.wk, self.wgate))[2].shape[0]
        result = (
            f"M={x.shape[0]} hidden={x.shape[1]} heads={heads} "
            f"q_lora_rank={q_lora.shape[1]} page={cache.shape[1]}: "
            f"fused={fused_ms * 1000:.1f}us native={native_ms * 1000:.1f}us "
            f"speedup={native_ms / fused_ms:.2f}x"
        )
        if generic_ms is not None:
            result += (
                f" generic={generic_ms * 1000:.1f}us "
                f"schedule-speedup={generic_ms / fused_ms:.2f}x"
            )
            self.assertLess(fused_ms, generic_ms * 0.8)
        print(result)
        self.assertLess(fused_ms, native_ms * 0.9)

    def _run_case(
        self,
        rows: int,
        heads: int = 32,
        q_lora_rank: int = 2048,
        hidden_size: int = 6144,
        page_size: int = 64,
        eps: float = 1e-6,
        norm_dtype: torch.dtype = torch.bfloat16,
        rope_dim: int = 64,
        is_neox_style: bool = False,
        benchmark: bool = False,
    ):
        inputs = self._inputs(rows, hidden_size, q_lora_rank, page_size)
        model_weights = self._weights(heads, q_lora_rank, hidden_size)
        norm = (self.gamma.to(norm_dtype), self.beta.to(norm_dtype))
        cos_sin = self.cos_sin if rope_dim == 64 else _make_cos_sin(rope_dim)
        kwargs = dict(
            model_weights=model_weights,
            norm=norm,
            cos_sin=cos_sin,
            eps=eps,
            rope_dim=rope_dim,
            is_neox_style=is_neox_style,
        )
        q, weights = self._prepare(inputs, **kwargs)
        torch.cuda.synchronize()

        x, q_lora, positions, slots, cache = inputs
        wq, wk, wgate = model_weights
        gamma, beta = norm
        q_ref = F.linear(q_lora, wq).reshape(rows, heads, 128)
        q_ref = _rotate_reference(q_ref, cos_sin, positions, rope_dim, is_neox_style)
        gate = F.linear(x, wgate)
        gate = (gate.float() * heads**-0.5).to(torch.bfloat16).float()
        combined_ref = q_ref.float() * gate.unsqueeze(-1) * (128**-0.5)
        combined_actual = q.float() * weights.unsqueeze(-1)
        self.assertGreater(_snr(combined_actual, combined_ref), 20)

        key_ref = F.linear(x, wk)
        key_ref = F.layer_norm(
            key_ref.float(),
            (128,),
            gamma.float(),
            beta.float(),
            eps,
        ).to(torch.bfloat16)
        key_ref = _rotate_reference(
            key_ref, cos_sin, positions, rope_dim, is_neox_style
        )
        for row, slot in enumerate(slots.tolist()):
            self.assertGreater(_snr(_decode_cache_row(cache, slot), key_ref[row]), 20)

        graph_inputs = (*inputs[:-1], torch.zeros_like(cache))
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_q, graph_weights = self._prepare(graph_inputs, **kwargs)
        graph_q.zero_()
        graph_weights.zero_()
        graph_inputs[-1].zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(graph_q, q))
        self.assertTrue(torch.equal(graph_weights, weights))
        self.assertTrue(torch.equal(graph_inputs[-1], cache))

        if benchmark:
            self._benchmark(inputs, **kwargs)

    def test_decode_shapes_correctness_and_graph_replay(self):
        for rows in _ROWS:
            with self.subTest(rows=rows):
                self._run_case(rows)

    def test_non_default_layer_norm_epsilon(self):
        self._run_case(10, eps=1e-5)

    def test_fp32_layer_norm_parameters(self):
        self._run_case(10, norm_dtype=torch.float32)

    def test_generalized_rope_dimensions_and_layouts(self):
        for rows, rope_dim, is_neox_style in product(
            (10, 64), (32, 64, 128), (False, True)
        ):
            with self.subTest(
                rows=rows, rope_dim=rope_dim, is_neox_style=is_neox_style
            ):
                self._run_case(rows, rope_dim=rope_dim, is_neox_style=is_neox_style)

    def test_negative_slots_compute_query_without_writing_cache(self):
        for q_lora_rank, rows in product((1024, 2048), (1, 3, 10, 64)):
            wq = self.wq if q_lora_rank == 2048 else _randn(32 * 128, q_lora_rank)
            with self.subTest(rows=rows, q_lora_rank=q_lora_rank):
                inputs = self._inputs(rows, q_lora_rank=q_lora_rank)
                inputs[3].fill_(-1)
                inputs[4].fill_(0xA5)
                before = inputs[4].clone()
                kwargs = dict(model_weights=(wq, self.wk, self.wgate))

                outputs = self._prepare(inputs, **kwargs)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    graph_outputs = self._prepare(inputs, **kwargs)
                graph.replay()
                torch.cuda.synchronize()

                self.assertTrue(torch.equal(inputs[4], before))
                for output in (*outputs, *graph_outputs):
                    self.assertTrue(torch.isfinite(output.float()).all())

    def test_generalized_geometries(self):
        cases = [
            (10, 16, 1024, 1536, 16),
            (64, 32, 1536, 3072, 32),
            (64, 48, 1536, 4608, 48),
            (10, 64, 2560, 7680, 128),
        ]
        cases.extend((rows, 32, 1024, 6144, 64) for rows in range(1, 9))
        for rows, heads, q_lora_rank, hidden_size, page_size in cases:
            with self.subTest(
                rows=rows,
                heads=heads,
                q_lora_rank=q_lora_rank,
                hidden_size=hidden_size,
                page_size=page_size,
            ):
                self._run_case(
                    rows, heads, q_lora_rank, hidden_size, page_size, benchmark=True
                )

    def test_decode_shapes_performance(self):
        for rows in _ROWS:
            with self.subTest(rows=rows):
                self._benchmark(
                    self._inputs(rows),
                    warmup=100,
                    rep=300,
                    compare_generic=rows >= 64,
                )


if __name__ == "__main__":
    unittest.main(verbosity=3)
