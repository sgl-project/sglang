import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import triton
import triton.language as tl
from torch import nn

import sglang.kernels.ops.layernorm.hy4_ihc as hy4_ihc
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import hunyuan_v4
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@triton.jit
def _reference_hy4_ihc_pre_kernel(
    x_ptr,
    fn_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    hidden_size: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    K_TOTAL: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    magnitude: tl.constexpr,
    norm_eps: tl.constexpr,
    hc_eps: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_row = x_ptr + pid * K_TOTAL
    m_idx = tl.arange(0, HC_POW2)
    m_mask = m_idx < HC_MULT

    sumsq = tl.zeros((), dtype=tl.float32)
    mix_pre = tl.zeros((HC_POW2,), dtype=tl.float32)
    mix_post = tl.zeros((HC_POW2,), dtype=tl.float32)
    for k_off in tl.range(0, K_TOTAL, BLOCK_K):
        k_offs = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K_TOTAL
        x_tile = tl.load(x_row + k_offs, mask=k_mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_tile * x_tile, axis=0)

        fn_offs = m_idx[:, None] * K_TOTAL + k_offs[None, :]
        fn_mask = m_mask[:, None] & k_mask[None, :]
        mix_pre += tl.sum(
            tl.load(fn_ptr + fn_offs, mask=fn_mask, other=0.0) * x_tile[None, :],
            axis=1,
        )
        mix_post += tl.sum(
            tl.load(
                fn_ptr + HC_MULT * K_TOTAL + fn_offs,
                mask=fn_mask,
                other=0.0,
            )
            * x_tile[None, :],
            axis=1,
        )

    rsqrt = tl.rsqrt(sumsq / K_TOTAL + norm_eps)
    scale_pre = tl.load(scale_ptr)
    scale_post = tl.load(scale_ptr + 1)
    base_pre = tl.load(base_ptr + m_idx, mask=m_mask, other=0.0)
    base_post = tl.load(base_ptr + HC_MULT + m_idx, mask=m_mask, other=0.0)

    pre = tl.sigmoid(mix_pre * rsqrt * scale_pre + base_pre) + hc_eps
    post = magnitude * tl.sigmoid(mix_post * rsqrt * scale_post + base_post) + hc_eps
    tl.store(post_ptr + pid * HC_MULT + m_idx, post, mask=m_mask)

    y_row = y_ptr + pid * hidden_size
    for d_off in tl.range(0, hidden_size, BLOCK_D):
        d_offs = d_off + tl.arange(0, BLOCK_D)
        d_mask = d_offs < hidden_size
        y_block = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for m in tl.static_range(HC_MULT):
            x_m = tl.load(x_row + m * hidden_size + d_offs, mask=d_mask, other=0.0)
            pre_m = tl.sum(tl.where(m_idx == m, pre, 0.0), axis=0)
            y_block += pre_m * x_m.to(tl.float32)
        tl.store(
            y_row + d_offs,
            y_block.to(y_ptr.dtype.element_ty),
            mask=d_mask,
        )


def _reference_hy4_ihc_pre(x, hc_fn, hc_scale, hc_base):
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    y = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    if num_tokens == 0:
        return y, post

    _reference_hy4_ihc_pre_kernel[(num_tokens,)](
        x,
        hc_fn,
        hc_scale,
        hc_base,
        y,
        post,
        hidden_size=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=triton.next_power_of_2(hc_mult),
        K_TOTAL=k_total,
        BLOCK_K=1024,
        BLOCK_D=1024,
        magnitude=2.0,
        norm_eps=1e-6,
        hc_eps=1e-6,
        num_warps=8,
        enable_fp_fusion=False,
    )
    return y, post


def _ihc_pre_inputs(num_tokens, hidden_size, hc_mult=4):
    x = torch.randn(
        num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16
    )
    hc_fn = (
        torch.randn(
            2 * hc_mult, hc_mult * hidden_size, device="cuda", dtype=torch.float32
        )
        * 0.02
    )
    hc_scale = torch.tensor([0.7, 1.3], device="cuda", dtype=torch.float32)
    hc_base = torch.randn(2 * hc_mult, device="cuda", dtype=torch.float32) * 0.5
    return x, hc_fn, hc_scale, hc_base


def _fp64_hy4_ihc_pre(x, hc_fn, hc_scale, hc_base, magnitude, norm_eps, hc_eps):
    num_tokens, hc_mult, _ = x.shape
    x64 = x.double()
    flat = x64.reshape(num_tokens, -1)
    rsqrt = torch.rsqrt(flat.square().mean(dim=-1, keepdim=True) + norm_eps)
    scale = hc_scale.double().repeat_interleave(hc_mult)
    gate = torch.sigmoid(flat @ hc_fn.double().T * rsqrt * scale + hc_base.double())
    pre = gate[:, :hc_mult] + hc_eps
    post = magnitude * gate[:, hc_mult:] + hc_eps
    return (pre.unsqueeze(-1) * x64).sum(dim=1), post


class _TupleLinear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))

    def forward(self, inputs):
        return nn.functional.linear(inputs, self.weight), None


_TINY_HC_CONFIG = SimpleNamespace(
    hidden_size=8,
    hc_mult=2,
    hc_magnitude=2.0,
    hc_eps=1e-6,
    rms_norm_eps=1e-5,
)


def _tiny_hc_layers():
    config = _TINY_HC_CONFIG
    with patch.object(hunyuan_v4, "ReplicatedLinear", _TupleLinear):
        pre_layer = hunyuan_v4.HYV4HCPreLayer(config, "pre").cuda()
        layer = hunyuan_v4.HYV4HCLayer(config, "layer").cuda()
        next_layer = hunyuan_v4.HYV4HCLayer(config, "next").cuda()
        head_layer = hunyuan_v4.HYV4HCHeadLayer(config, "head").cuda()
    norm = hunyuan_v4.RMSNorm(
        config.hidden_size, eps=config.rms_norm_eps, force_native=True
    ).cuda()
    return pre_layer, layer, next_layer, head_layer, norm


@unittest.skipUnless(torch.cuda.is_available(), "HYV4 Triton kernels need CUDA")
class TestHy4DecodeKernels(CustomTestCase):
    def test_split_k_matches_single_cta(self):
        torch.manual_seed(0)
        for num_tokens, hidden_size in (
            (0, 6144),
            (1, 4096),
            (3, 4100),
            (31, 6144),
            (hy4_ihc._IHC_TOKEN_BLOCKED_MIN_T - 1, 6144),
        ):
            with self.subTest(num_tokens=num_tokens, hidden_size=hidden_size):
                x, hc_fn, hc_scale, hc_base = _ihc_pre_inputs(num_tokens, hidden_size)

                expected = _reference_hy4_ihc_pre(x, hc_fn, hc_scale, hc_base)
                with patch.object(hy4_ihc, "_hpc_ihc_op", return_value=None):
                    actual = hy4_ihc.fused_hy4_ihc_pre(
                        x, hc_fn, hc_scale, hc_base, 2.0, 1e-6, 1e-6
                    )

                self.assertTrue(torch.equal(actual[0], expected[0]))
                self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_token_blocked_split_k_matches_fp64_reference(self):
        """From _IHC_TOKEN_BLOCKED_MIN_T tokens the pre kernel sums in another
        order, so it is held to an fp64 reference instead of bitwise."""
        torch.manual_seed(0)
        min_t = hy4_ihc._IHC_TOKEN_BLOCKED_MIN_T
        # 65 tokens leave a partial token block; 4100 leaves K and D tail tiles.
        for num_tokens, hidden_size in ((min_t, 6144), (min_t + 1, 4100)):
            with self.subTest(num_tokens=num_tokens, hidden_size=hidden_size):
                inputs = _ihc_pre_inputs(num_tokens, hidden_size)
                with patch.object(hy4_ihc, "_hpc_ihc_op", return_value=None):
                    y, post = hy4_ihc.fused_hy4_ihc_pre(*inputs, 2.0, 1e-6, 1e-6)
                y_ref, post_ref = _fp64_hy4_ihc_pre(*inputs, 2.0, 1e-6, 1e-6)

                self.assertTrue(torch.isfinite(y).all())
                self.assertTrue(torch.isfinite(post).all())
                # rtol on y is half a bf16 ulp; each atol is ~10x the worst fp32
                # error measured over 40 seeds on GB300 (9.4e-7 y, 1.7e-6 post).
                torch.testing.assert_close(y.double(), y_ref, rtol=2**-8, atol=1e-5)
                torch.testing.assert_close(
                    post.double(), post_ref, rtol=1e-5, atol=1e-5
                )

    def test_fused_ihc_failure_disables_each_path(self):
        config = _TINY_HC_CONFIG
        counts = {"pre": 0, "post": 0, "post_pre": 0, "head": 0}

        def fail(name):
            def raise_error(*args, **kwargs):
                counts[name] += 1
                raise RuntimeError(name)

            return raise_error

        pre_layer, layer, next_layer, head_layer, norm = _tiny_hc_layers()
        next_layer.hc_pre._fused_ihc_pre_disabled = True
        hidden_states = torch.randn(3, 2, 8, device="cuda")
        output = torch.randn(3, 8, device="cuda")
        residual = torch.randn(3, 2, 8, device="cuda")
        post = torch.randn(3, 2, device="cuda")

        with (
            patch.object(hy4_ihc, "fused_hy4_ihc_pre", fail("pre")),
            patch.object(hy4_ihc, "fused_hy4_ihc_post", fail("post")),
            patch.object(hy4_ihc, "fused_hy4_ihc_post_pre", fail("post_pre")),
            patch.object(hy4_ihc, "fused_hy4_ihc_head", fail("head")),
            patch.object(hunyuan_v4, "_hpc_ihc_available", return_value=True),
            self.assertLogs(hunyuan_v4.logger, level="WARNING") as logs,
        ):
            for _ in range(2):
                pre_layer(hidden_states)
                layer.post(output, residual, post)
                layer.post_pre(output, residual, post, next_layer, norm)
                head_layer(hidden_states)

        self.assertEqual(counts, {"pre": 1, "post": 1, "post_pre": 1, "head": 1})
        self.assertEqual(len(logs.records), 4)
        self.assertTrue(all(record.exc_info is not None for record in logs.records))

    def test_ihc_policy_normalizes_each_path_exactly_once(self):
        """Whichever backend the policy picks, RMSNorm must run exactly once
        (model norm or folded hpc op), never the multi-kernel torch fallback."""
        _, layer, next_layer, head_layer, norm = _tiny_hc_layers()
        hc_mult, hidden_size = _TINY_HC_CONFIG.hc_mult, _TINY_HC_CONFIG.hidden_size
        hidden_states = torch.randn(3, hc_mult, hidden_size, device="cuda")
        output = torch.randn(3, hidden_size, device="cuda")
        post = torch.rand(3, hc_mult, device="cuda")
        folded = []

        def fake_pre(x, *args):
            folded.append(("pre", args[6] is not None))
            return x[:, 0], torch.ones(x.shape[:2], device=x.device)

        def fake_post_pre(output, residual, post, *args):
            folded.append(("post_pre", args[6] is not None))
            return residual, output, post

        def fake_head(hidden_states, *args):
            folded.append(("head", args[5] is not None))
            return hidden_states[:, 0]

        fakes = {
            "fuse_ihc_pre": fake_pre,
            "fuse_ihc_post_pre": fake_post_pre,
            "fuse_ihc_post": lambda output, residual, post: residual,
            "fuse_ihc_head": fake_head,
        }
        all_folded = [("pre", True), ("post_pre", True), ("head", True)]
        cases = [
            ("hpc", False, all_folded, 0),
            ("triton", False, [], 3),
            ("mode", False, all_folded, 0),
            ("mode", True, [], 3),
            ("batch", True, [("post_pre", True), ("head", True)], 1),
        ]
        self.addCleanup(hy4_ihc._ihc_policy.cache_clear)
        for policy, is_decode, expected_folded, expected_norms in cases:
            folded.clear()
            hy4_ihc._ihc_policy.cache_clear()
            with (
                self.subTest(policy=policy, is_decode=is_decode),
                envs.SGLANG_HY4_IHC_DECODE_POLICY.override(policy),
                hy4_ihc.hy4_ihc_forward_mode(is_decode),
                patch.object(hy4_ihc, "_hpc_ihc_op", lambda op, *_: fakes[op]),
                patch.object(hy4_ihc, "fused_hy4_ihc_head", fake_head),
                patch.object(
                    hy4_ihc, "_rms_norm", wraps=hy4_ihc._rms_norm
                ) as torch_norm,
                patch.object(norm, "forward", wraps=norm.forward) as norm_forward,
                self.assertNoLogs(hunyuan_v4.logger, level="WARNING"),
            ):
                layer.pre(hidden_states, norm)
                layer.post_pre(output, hidden_states, post, next_layer, norm)
                head_layer(hidden_states, norm)
                self.assertEqual(folded, expected_folded)
                self.assertEqual(norm_forward.call_count, expected_norms)
                self.assertEqual(torch_norm.call_count, 0)

    def test_target_verify_graph_keeps_the_capture_time_ihc_choice(self):
        """A speculative verify step is extend-side: its CUDA graph must bake in
        the hpc ops with RMSNorm folded, whatever context the replay runs under."""
        _, layer, next_layer, head_layer, norm = _tiny_hc_layers()
        hc_mult, hidden_size = _TINY_HC_CONFIG.hc_mult, _TINY_HC_CONFIG.hidden_size
        hidden_states = torch.randn(6, hc_mult, hidden_size, device="cuda")
        output = torch.randn(6, hidden_size, device="cuda")
        folded = []

        def fake_pre(x, *args):
            folded.append(("pre", args[6] is not None))
            return x[:, 0] + 100, x.new_ones(x.shape[:2])

        def fake_post_pre(output, residual, post, *args):
            folded.append(("post_pre", args[6] is not None))
            return residual + 100, output + 100, post

        def fake_head(hidden_states, *args):
            folded.append(("head", args[5] is not None))
            return hidden_states[:, 0] + 100

        fakes = {
            "fuse_ihc_pre": fake_pre,
            "fuse_ihc_post_pre": fake_post_pre,
            "fuse_ihc_post": lambda output, residual, post: residual,
            "fuse_ihc_head": fake_head,
        }

        def run(mode):
            folded.clear()
            with hunyuan_v4._ihc_forward_mode(SimpleNamespace(forward_mode=mode)):
                reduced, post, residual = layer.pre(hidden_states, norm)
                next_reduced, _, next_residual = layer.post_pre(
                    output, residual, post, next_layer, norm
                )
                return reduced, next_reduced, head_layer(next_residual, norm)

        all_folded = [("pre", True), ("post_pre", True), ("head", True)]
        # batch counts tokens, not requests: 2 requests x 3 draft tokens cross 4.
        cases = [("mode", 128, []), ("batch", 4, all_folded)]
        self.addCleanup(hy4_ihc._ihc_policy.cache_clear)
        for policy, threshold, decode_folded in cases:
            hy4_ihc._ihc_policy.cache_clear()
            with (
                self.subTest(policy=policy),
                envs.SGLANG_HY4_IHC_DECODE_POLICY.override(policy),
                envs.SGLANG_HY4_IHC_PRE_T_THRESHOLD.override(threshold),
                patch.object(hy4_ihc, "_hpc_ihc_op", lambda op, *_: fakes[op]),
                patch.object(hy4_ihc, "fused_hy4_ihc_head", fake_head),
                patch.object(
                    hy4_ihc, "_rms_norm", wraps=hy4_ihc._rms_norm
                ) as torch_norm,
                patch.object(norm, "forward", wraps=norm.forward) as norm_forward,
                self.assertNoLogs(hunyuan_v4.logger, level="WARNING"),
            ):
                decode = run(ForwardMode.DECODE)
                self.assertEqual(folded, decode_folded)
                run(ForwardMode.EXTEND)
                self.assertEqual(folded, all_folded)

                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    run(ForwardMode.TARGET_VERIFY)
                torch.cuda.current_stream().wait_stream(warmup_stream)
                eager = [t.clone() for t in run(ForwardMode.TARGET_VERIFY)]

                norm_forward.reset_mock()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run(ForwardMode.TARGET_VERIFY)
                self.assertEqual(folded, all_folded)
                self.assertEqual(norm_forward.call_count, 0)
                self.assertEqual(torch_norm.call_count, 0)

                for tensor in captured:
                    tensor.zero_()
                with hy4_ihc.hy4_ihc_forward_mode(True):
                    graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(all(map(torch.equal, eager, captured)))
                self.assertEqual(
                    not all(map(torch.equal, eager, decode)), policy == "mode"
                )


if __name__ == "__main__":
    unittest.main()
