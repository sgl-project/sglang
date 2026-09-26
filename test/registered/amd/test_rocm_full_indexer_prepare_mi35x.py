"""Correctness and graph-capture coverage for the gfx950 indexer prepare JIT.

The reference intentionally matches the current AITER fallback's no-Hadamard
cache representation so switching paths between calls remains safe.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=300, suite="stage-b-test-1-gpu-small-amd-mi35x")

_RUNNABLE = is_hip() and is_gfx95_supported()
if _RUNNABLE:
    try:
        import triton
        from aiter.ops.cache import indexer_k_quant_and_cache

        from sglang.kernels.ops.attention.dsa.hip_gfx950 import (
            full_indexer_prepare,
            is_full_indexer_prepare_available,
        )
        from sglang.kernels.ops.attention.dsa.hip_gfx950.indexer_prepare_m4 import (
            indexer_prepare as small_m_indexer_prepare,
        )
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant

        _RUNNABLE = is_full_indexer_prepare_available()
    except Exception:
        _RUNNABLE = False


def _rotate_reference(
    x: torch.Tensor, cos_sin: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    pairs = x.reshape(*x.shape[:-1], 64, 2)
    even, odd = pairs[..., 0], pairs[..., 1]
    cos = cos_sin[positions, :32]
    sin = cos_sin[positions, 32:]
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    rotated = pairs.clone()
    rotated[..., :32, 0] = even[..., :32] * cos - odd[..., :32] * sin
    rotated[..., :32, 1] = odd[..., :32] * cos + even[..., :32] * sin
    return rotated.flatten(-2).to(torch.bfloat16)


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
class TestROCmFullIndexerPrepare(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        device = "cuda"
        cls.wq = torch.randn(4096, 2048, device=device, dtype=torch.bfloat16).mul_(0.01)
        cls.wk = torch.randn(128, 6144, device=device, dtype=torch.bfloat16).mul_(0.01)
        cls.wgate = torch.randn(32, 6144, device=device, dtype=torch.bfloat16).mul_(
            0.01
        )
        cls.gamma = (
            torch.randn(128, device=device, dtype=torch.bfloat16).mul_(0.05).add_(1)
        )
        cls.beta = torch.randn(128, device=device, dtype=torch.bfloat16).mul_(0.01)

        positions = torch.arange(256, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, 64, 2, device=device, dtype=torch.float32) / 64)
        )
        freqs = torch.outer(positions, inv_freq)
        cls.cos_sin = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16)

    def _inputs(self, rows: int):
        x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        q_lora = torch.randn(rows, 2048, device="cuda", dtype=torch.bfloat16).mul_(0.1)
        positions = (
            torch.arange(rows, device="cuda", dtype=torch.int64) * 17 + 3
        ) % 256
        slots = torch.arange(rows, device="cuda", dtype=torch.int64) * 17
        cache = torch.zeros(
            (int(slots[-1]) // 64) + 1, 64, 132, device="cuda", dtype=torch.uint8
        )
        return x, q_lora, positions, slots, cache

    def _native_prepare(self, x, q_lora, positions, slots, cache, wq=None, wgate=None):
        wq = self.wq if wq is None else wq
        wgate = self.wgate if wgate is None else wgate
        rows = x.shape[0]
        heads = wgate.shape[0]
        q = F.linear(q_lora, wq).reshape(rows, heads, 128)
        key = F.linear(x, self.wk)
        key = F.layer_norm(
            key.float(),
            (128,),
            self.gamma.float(),
            self.beta.float(),
            1e-6,
        ).to(torch.bfloat16)
        q = _rotate_reference(q, self.cos_sin, positions)
        key = _rotate_reference(key, self.cos_sin, positions)
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

    def _run_case(
        self,
        rows: int,
        eps: float = 1e-6,
        norm_dtype: torch.dtype = torch.bfloat16,
    ):
        x, q_lora, positions, slots, cache = self._inputs(rows)
        gamma = self.gamma.to(norm_dtype)
        beta = self.beta.to(norm_dtype)

        q, weights = full_indexer_prepare(
            x,
            q_lora,
            self.wq,
            self.wk,
            self.wgate,
            gamma,
            beta,
            self.cos_sin,
            positions,
            slots,
            cache,
            eps=eps,
        )
        torch.cuda.synchronize()

        q_ref = F.linear(q_lora, self.wq).reshape(rows, 32, 128)
        q_ref = _rotate_reference(q_ref, self.cos_sin, positions)
        gate = F.linear(x, self.wgate)
        gate = (gate.float() * (32**-0.5)).to(torch.bfloat16).float()
        combined_ref = q_ref.float() * gate.unsqueeze(-1) * (128**-0.5)
        combined_actual = q.float() * weights.unsqueeze(-1)
        self.assertGreater(_snr(combined_actual, combined_ref), 20)

        key_ref = F.linear(x, self.wk)
        key_ref = F.layer_norm(
            key_ref.float(),
            (128,),
            gamma.float(),
            beta.float(),
            eps,
        ).to(torch.bfloat16)
        key_ref = _rotate_reference(key_ref, self.cos_sin, positions)
        for row, slot in enumerate(slots.tolist()):
            self.assertGreater(_snr(_decode_cache_row(cache, slot), key_ref[row]), 20)

        graph_cache = torch.zeros_like(cache)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_q, graph_weights = full_indexer_prepare(
                x,
                q_lora,
                self.wq,
                self.wk,
                self.wgate,
                gamma,
                beta,
                self.cos_sin,
                positions,
                slots,
                graph_cache,
                eps=eps,
            )
        graph_q.zero_()
        graph_weights.zero_()
        graph_cache.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(graph_q, q))
        self.assertTrue(torch.equal(graph_weights, weights))
        self.assertTrue(torch.equal(graph_cache, cache))

    def test_decode_shapes_correctness_and_graph_replay(self):
        for rows in (1, 2, 4, 8, 10, 16, 32, 40, 64, 96, 128):
            with self.subTest(rows=rows):
                self._run_case(rows)

    def test_non_default_layer_norm_epsilon(self):
        self._run_case(10, eps=1e-5)

    def test_fp32_layer_norm_parameters(self):
        self._run_case(10, norm_dtype=torch.float32)

    def test_negative_slots_compute_query_without_writing_cache(self):
        for rows in (1, 10, 64):
            with self.subTest(rows=rows):
                x, q_lora, positions, slots, cache = self._inputs(rows)
                slots.fill_(-1)
                cache.fill_(0xA5)
                before = cache.clone()

                q, weights = full_indexer_prepare(
                    x,
                    q_lora,
                    self.wq,
                    self.wk,
                    self.wgate,
                    self.gamma,
                    self.beta,
                    self.cos_sin,
                    positions,
                    slots,
                    cache,
                    eps=1e-6,
                )
                torch.cuda.synchronize()

                self.assertTrue(torch.equal(cache, before))
                self.assertTrue(torch.isfinite(q.float()).all())
                self.assertTrue(torch.isfinite(weights).all())

    def test_generalized_heads_and_q_lora_rank(self):
        for heads, q_lora_rank in ((16, 1024), (16, 2048), (32, 1024)):
            for rows in (10, 64):
                with self.subTest(heads=heads, q_lora_rank=q_lora_rank, rows=rows):
                    self._run_generalized_case(rows, heads, q_lora_rank)

    def _run_generalized_case(self, rows: int, heads: int, q_lora_rank: int):
        x, _, positions, slots, cache = self._inputs(rows)
        q_lora = torch.randn(
            rows, q_lora_rank, device="cuda", dtype=torch.bfloat16
        ).mul_(0.1)
        wq = torch.randn(
            heads * 128,
            q_lora_rank,
            device="cuda",
            dtype=torch.bfloat16,
        ).mul_(0.01)
        wgate = torch.randn(heads, 6144, device="cuda", dtype=torch.bfloat16).mul_(0.01)

        q, weights = full_indexer_prepare(
            x,
            q_lora,
            wq,
            self.wk,
            wgate,
            self.gamma,
            self.beta,
            self.cos_sin,
            positions,
            slots,
            cache,
            eps=1e-6,
        )
        torch.cuda.synchronize()

        q_ref = F.linear(q_lora, wq).reshape(rows, heads, 128)
        q_ref = _rotate_reference(q_ref, self.cos_sin, positions)
        gate = F.linear(x, wgate)
        gate = (gate.float() * heads**-0.5).to(torch.bfloat16).float()
        combined_ref = q_ref.float() * gate.unsqueeze(-1) * (128**-0.5)
        combined_actual = q.float() * weights.unsqueeze(-1)
        self.assertGreater(_snr(combined_actual, combined_ref), 20)

        key_ref = F.linear(x, self.wk)
        key_ref = F.layer_norm(
            key_ref.float(),
            (128,),
            self.gamma.float(),
            self.beta.float(),
            1e-6,
        ).to(torch.bfloat16)
        key_ref = _rotate_reference(key_ref, self.cos_sin, positions)
        for row, slot in enumerate(slots.tolist()):
            self.assertGreater(_snr(_decode_cache_row(cache, slot), key_ref[row]), 20)

        graph_cache = torch.zeros_like(cache)
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            graph_q, graph_weights = full_indexer_prepare(
                x,
                q_lora,
                wq,
                self.wk,
                wgate,
                self.gamma,
                self.beta,
                self.cos_sin,
                positions,
                slots,
                graph_cache,
                eps=1e-6,
            )
        graph_q.zero_()
        graph_weights.zero_()
        graph_cache.zero_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(graph_q, q))
        self.assertTrue(torch.equal(graph_weights, weights))
        self.assertTrue(torch.equal(graph_cache, cache))

        def fused():
            return full_indexer_prepare(
                x,
                q_lora,
                wq,
                self.wk,
                wgate,
                self.gamma,
                self.beta,
                self.cos_sin,
                positions,
                slots,
                cache,
                eps=1e-6,
            )

        def native():
            return self._native_prepare(
                x,
                q_lora,
                positions,
                slots,
                cache,
                wq=wq,
                wgate=wgate,
            )

        fused_ms = triton.testing.do_bench(fused, warmup=50, rep=100)
        native_ms = triton.testing.do_bench(native, warmup=50, rep=100)
        print(
            f"heads={heads} q_lora_rank={q_lora_rank} M={rows}: "
            f"fused={fused_ms * 1000:.1f}us native={native_ms * 1000:.1f}us "
            f"speedup={native_ms / fused_ms:.2f}x"
        )
        self.assertLess(fused_ms, native_ms * 0.9)

    def test_decode_shapes_performance(self):
        for rows in (1, 2, 4, 8, 10, 16, 32, 40, 64, 96, 128):
            with self.subTest(rows=rows):
                x, q_lora, positions, slots, cache = self._inputs(rows)

                def fused():
                    return full_indexer_prepare(
                        x,
                        q_lora,
                        self.wq,
                        self.wk,
                        self.wgate,
                        self.gamma,
                        self.beta,
                        self.cos_sin,
                        positions,
                        slots,
                        cache,
                        eps=1e-6,
                    )

                def native():
                    return self._native_prepare(x, q_lora, positions, slots, cache)

                fused_ms = triton.testing.do_bench(fused, warmup=100, rep=300)
                native_ms = triton.testing.do_bench(native, warmup=100, rep=300)
                small_m_ms = None
                if rows in (64, 96, 128):

                    def small_m():
                        return small_m_indexer_prepare(
                            x,
                            q_lora,
                            self.wq,
                            self.wk,
                            self.wgate,
                            self.gamma,
                            self.beta,
                            self.cos_sin,
                            positions,
                            slots,
                            cache,
                            eps=1e-6,
                        )

                    small_m_ms = triton.testing.do_bench(small_m, warmup=100, rep=300)
                print(
                    f"M={rows}: fused={fused_ms * 1000:.1f}us "
                    f"native={native_ms * 1000:.1f}us "
                    f"speedup={native_ms / fused_ms:.2f}x"
                    + (
                        f" small-M={small_m_ms * 1000:.1f}us "
                        f"schedule-speedup={small_m_ms / fused_ms:.2f}x"
                        if small_m_ms is not None
                        else ""
                    )
                )
                self.assertLess(fused_ms, native_ms * 0.9)
                if small_m_ms is not None:
                    self.assertLess(fused_ms, small_m_ms * 0.8)


if __name__ == "__main__":
    unittest.main(verbosity=3)
