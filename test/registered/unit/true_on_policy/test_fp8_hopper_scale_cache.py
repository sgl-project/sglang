"""SM90 TOP FP8: serving's linears and experts reuse the unpacked weight block scales, rewritten in place after a
weight write, with the bytes of the per-call unpack; the quantizer writes FP32 activation scales with the bits an
unpack of its packed UE8M0 bytes gives."""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


@unittest.skipUnless(SM90, "Hopper scale conversion")
class TestHopperScaleCache(unittest.TestCase):
    def setUp(self):
        from sglang.srt.layers.quantization.fp8_utils import (
            quant_weight_ue8m0,
            transform_scale_ue8m0,
        )

        self.quant, self.transform = quant_weight_ue8m0, transform_scale_ue8m0
        torch.manual_seed(0)

    def _pair(self, *shape):
        master = (torch.randn(*shape, device="cuda") * 0.02).bfloat16()
        if master.ndim == 2:
            w, s = self.quant(master, weight_block_size=[128, 128])
        else:
            w, s = map(
                torch.stack,
                zip(*(self.quant(m, weight_block_size=[128, 128]) for m in master)),
            )
        scale = torch.nn.Parameter(
            self.transform(s, mn=w.shape[-2]), requires_grad=False
        )
        return w, scale

    def test_linear_cached_scales_are_the_per_call_bytes(self):
        from sglang.srt.true_on_policy.fp8.fp8_linear import linear

        # N64 (a partial last 128-row block) and GLM-5.2 decode shapes.
        for n, k in ((576, 6144), (2048, 6144), (6144, 16384), (192, 256)):
            w, s = self._pair(n, k)
            for rows in (1, 16, 256, 4096):
                x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                expected = linear(x, w, s)
                self.assertTrue(
                    torch.equal(linear(x, w, s, cache_scales=True), expected)
                )
                # The second call hits the cache.
                self.assertTrue(
                    torch.equal(linear(x, w, s, cache_scales=True), expected)
                )

    def test_experts_cached_scales_are_the_per_call_bytes(self):
        from sglang.srt.true_on_policy.fp8.fp8_expert_forward import expert_forward

        experts, hidden, inter = 8, 2048, 768
        w13, w2 = (
            self._pair(experts, 2 * inter, hidden),
            self._pair(experts, hidden, inter),
        )
        rows = 128 * experts
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16)
        m_indices = torch.arange(
            experts, device="cuda", dtype=torch.int32
        ).repeat_interleave(128)
        expected = expert_forward(x, w13, w2, m_indices=m_indices)
        for _ in range(2):
            out = expert_forward(x, w13, w2, m_indices=m_indices, cache_scales=True)
            self.assertTrue(torch.equal(out, expected))
        masked_x = x.view(experts, 128, hidden)
        masked_m = torch.full((experts,), 100, device="cuda", dtype=torch.int32)
        kwargs = dict(masked_m=masked_m, expected_m=100)
        expected = expert_forward(masked_x, w13, w2, **kwargs)
        out = expert_forward(masked_x, w13, w2, cache_scales=True, **kwargs)
        self.assertTrue(torch.equal(out[:, :100], expected[:, :100]))

    def test_a_weight_write_refreshes_the_cache_in_place(self):
        from sglang.srt.true_on_policy import fp8
        from sglang.srt.true_on_policy.fp8.hopper_scales import block_scales

        w, s = self._pair(1024, 2048)
        _, new = self._pair(1024, 2048)
        _, newer = self._pair(1024, 2048)
        cached = block_scales(s, 16, cached=True)
        self.assertIs(block_scales(s, 16, cached=True), cached)
        # A loader write goes through .data, which no version counter sees.
        s.data.copy_(new)
        fp8.weights_changed()
        fresh = block_scales(s, 16, cached=True)
        # The same buffer, rewritten: a captured graph keeps reading it.
        self.assertIs(fresh, cached)
        self.assertTrue(torch.equal(fresh, block_scales(new, 16)))
        # Storage replaced (requant at load) rewrites it as well.
        s.data = newer.clone()
        self.assertIs(block_scales(s, 16, cached=True), cached)
        self.assertTrue(torch.equal(cached, block_scales(newer, 16)))

    def test_graph_replay_reads_the_refreshed_cache(self):
        from sglang.srt.true_on_policy import fp8
        from sglang.srt.true_on_policy.fp8.hopper_scales import block_scales

        w, s = self._pair(1024, 2048)
        _, new = self._pair(1024, 2048)
        cached = block_scales(s, 16, cached=True)  # before capture, as warmup does
        out = torch.empty(8, 16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            # The captured graph copies the cached buffer: no unpack runs on replay.
            self.assertIs(block_scales(s, 16, cached=True), cached)
            out.copy_(block_scales(s, 16, cached=True))
        torch.cuda.current_stream().wait_stream(stream)
        # Serving's weight write: bytes in place, then weights_changed and the
        # refresh the weight updater runs once the write is complete.
        s.data.copy_(new)
        fp8.weights_changed()
        fp8.refresh_derived()
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, block_scales(new, 16)))
        # Nothing changed: the refresh is a no-op compare.
        fp8.refresh_derived()

    def test_refresh_skips_weights_dropped_since(self):
        from sglang.srt.true_on_policy import fp8
        from sglang.srt.true_on_policy.fp8.hopper_scales import block_scales

        _, s = self._pair(256, 256)
        block_scales(s, 2, cached=True)
        del s
        fp8.weights_changed()
        fp8.refresh_derived()


def _unpack_reference(packed, groups, rows):
    """Packed UE8M0 int32 [..., rows, ceil(groups/4)] (any strides) -> FP32 bits [..., rows, groups], Hopper's rules."""
    words = packed[..., :rows, :].to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(4, device=packed.device) * 8
    exponent = ((words.unsqueeze(-1) >> shifts) & 255).flatten(-2)[..., :groups]
    bits = exponent << 23
    bits = torch.where(exponent == 0, 0x00400000, bits)
    bits = torch.where(exponent == 255, 0x7FC00000, bits)
    return bits.to(torch.int32).view(torch.float32)


@unittest.skipUnless(SM90, "Hopper scale conversion")
class TestHopperActivationScales(unittest.TestCase):
    """quantize_rows on SM90: E4M3 payload of the packed path, FP32 scales = unpack of its packed UE8M0 bytes."""

    def _packed(self, x, fused_swiglu, masked_m):
        from sglang.kernels.jit.utils import is_arch_support_pdl
        from sglang.kernels.ops.quantization.per_token_group_quant import (
            _allocate_outputs,
            _infer_scale_layout,
            _jit_module,
        )

        q, s = _allocate_outputs(x, 128, torch.float8_e4m3fn, True, True, fused_swiglu)
        row_major, aligned = _infer_scale_layout(s, True, q.shape[-1] // 128)
        module = _jit_module(
            torch.bfloat16,
            torch.float8_e4m3fn,
            128,
            True,
            row_major,
            aligned,
            fused_swiglu,
            masked_m is not None,
            is_arch_support_pdl(),
        )
        if masked_m is None:
            module.per_token_group_quant(x, q, s)
        else:
            module.per_token_group_quant(x, q, s, masked_m, -1)
        return q, s

    def _inputs(self, shape):
        torch.manual_seed(1)
        x = torch.randn(*shape, device="cuda") * 3
        flat = x.view(-1, shape[-1])
        # Rows of every scale regime: zeros, tiny, huge, subnormal, +-inf, NaN.
        specials = [0.0, 1e-30, 3e38, 1e-40, float("inf"), float("-inf"), float("nan")]
        for i, v in enumerate(specials):
            if i < flat.shape[0]:
                flat[i, : shape[-1] // 3] = v
        flat[len(specials) % flat.shape[0]] *= 2.0**-100
        return x.bfloat16().contiguous()

    def _check(self, x, *, fused_swiglu=False, masked_m=None):
        from sglang.srt.true_on_policy.fp8.fp8_expert_forward import quantize_rows

        q, s = quantize_rows(x, fused_swiglu=fused_swiglu, masked_m=masked_m)
        ref_q, ref_s = self._packed(x, fused_swiglu, masked_m)
        rows, groups = q.shape[-2], q.shape[-1] // 128
        aligned = -(-rows // 4) * 4
        # DeepGEMM's MN-major FP32 layout: rows aligned to 4, groups outermost.
        self.assertEqual(s.dtype, torch.float32)
        self.assertEqual(tuple(s.shape), (*q.shape[:-1], groups))
        self.assertEqual(s.stride()[-2:], (1, aligned))
        if s.ndim == 3:
            self.assertEqual(s.stride(0), groups * aligned)
        expected = _unpack_reference(ref_s, groups, rows)
        live = [rows] * (q.shape[0] if q.ndim == 3 else 1)
        if masked_m is not None:
            live = masked_m.tolist()
        q3, rq3 = q.view(-1, rows, q.shape[-1]), ref_q.view(-1, rows, q.shape[-1])
        s3, e3 = s.view(-1, rows, groups), expected.view(-1, rows, groups)
        for e, n in enumerate(live):
            self.assertTrue(
                torch.equal(q3[e, :n].view(torch.uint8), rq3[e, :n].view(torch.uint8))
            )
            self.assertTrue(
                torch.equal(s3[e, :n].view(torch.int32), e3[e, :n].view(torch.int32))
            )

    def test_flat(self):
        for rows in (1, 3, 7, 64, 129, 1000):
            for hidden in (128, 512, 6144, 7168):
                self._check(self._inputs((rows, hidden)))

    def test_flat_fused_swiglu(self):
        for rows in (1, 5, 256):
            for inter in (256, 2048, 3072):
                self._check(self._inputs((rows, 2 * inter)), fused_swiglu=True)

    def test_masked(self):
        experts, pad = 8, 96
        masked_m = torch.tensor(
            [0, 1, 5, 96, 33, 2, 64, 17], device="cuda", dtype=torch.int32
        )
        self._check(self._inputs((experts, pad, 6144)), masked_m=masked_m)
        self._check(
            self._inputs((experts, pad, 2 * 2048)), fused_swiglu=True, masked_m=masked_m
        )


if __name__ == "__main__":
    unittest.main()
