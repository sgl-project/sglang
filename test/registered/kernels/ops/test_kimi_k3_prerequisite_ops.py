"""Representative parity coverage for the lightweight Kimi-K3 prerequisites."""

import unittest

import torch

from sglang.kernels.ops.attention.concat_mla import concat_mla_absorb_q
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
    commit_kda_replayssm_spec,
)
from sglang.kernels.ops.attention.set_mla_kv_concat_q import (
    can_use_set_mla_kv_concat_q,
    can_use_set_mla_kv_concat_q_fp8,
    set_mla_kv_concat_q,
    set_mla_kv_concat_q_fp8,
)
from sglang.kernels.ops.attention.utils import concat_mla_absorb_q_general
from sglang.kernels.ops.attention.vision_rope import (
    apply_fused_qk_complex_rope,
    apply_fused_qk_complex_rope_inplace,
    prepare_fused_qk_complex_rope_inplace,
)
from sglang.kernels.ops.elementwise import add3
from sglang.kernels.ops.gemm.tiny_gemm import (
    tiny_k_gemm_bf16,
    tiny_n_gemm_bf16,
)
from sglang.kernels.ops.kvcache.set_mla_kv_buffer import set_mla_kv_buffer
from sglang.kernels.ops.mm.process.image import (
    _normalize_and_patchify_torch,
    normalize_and_patchify,
)
from sglang.kernels.ops.moe import moe_route_quant_fused
from sglang.kernels.ops.moe.moe_route_radix import route_radix
from sglang.kernels.ops.moe.moe_topk_sum import moe_topk_sum
from sglang.kernels.ops.moe.pack_topk_ids import PackTopkIds
from sglang.kernels.ops.quantization.per_token_group_quant import (
    per_token_group_quant,
)
from sglang.kernels.ops.sampling.top_p_renorm_triton import (
    top_p_renorm_probs_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 896
TOPK = 16
NOPE_DIM = 512
ROPE_DIM = 64
MLA_DIM = NOPE_DIM + ROPE_DIM
MLA_PAGES = 256


def _route_oracle(
    scores, bias, topk, renormalize, routed_scaling_factor, apply_scale, sorted
):
    """Pure-torch fp32 reference for route_radix.

    Deliberately independent of moe_fused_gate: that entry dispatches back to
    route_radix whenever scoring is sigmoid with no shared experts, no expert
    groups and no softcapping (moe_fused_gate.py, the covered() fast path), which
    is exactly the configuration under test.

    Contract, from route_radix.cuh: bias participates in RANKING only and the
    emitted weight stays bias-free; NaN is floored so it can never win; ties go to
    the lower expert id; renormalize divides by the winners' sum (guarded to 1 when
    that sum is non-positive) and only then is routed scaling applied; sorted=True
    emits (biased desc, id asc) while sorted=False emits ascending expert id.
    """
    s = torch.sigmoid(scores.float())
    biased = s + bias.float()
    biased = torch.where(torch.isnan(biased), torch.full_like(biased, -1e30), biased)
    # stable + descending: equal biased values keep ascending-id order
    ranked = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :topk]
    w = s.gather(1, ranked)
    total = w.sum(-1, keepdim=True)
    norm = torch.where(total > 0, total, torch.ones_like(total))
    if renormalize:
        w = w / norm
    if apply_scale:
        w = w * routed_scaling_factor
    if sorted:
        return w, ranked.to(torch.int32)
    by_id = ranked.argsort(dim=-1)
    return w.gather(1, by_id), ranked.gather(1, by_id).to(torch.int32)


def _make_mla_inputs(batch_size, num_heads, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return (
            torch.randn(*shape, generator=generator, device="cuda", dtype=torch.float32)
            .mul(0.1)
            .to(torch.bfloat16)
        )

    pool = randn(MLA_PAGES, MLA_DIM)
    latent = randn(batch_size, MLA_DIM)
    query = randn(batch_size, num_heads, MLA_DIM)
    loc = torch.randperm(MLA_PAGES, generator=generator, device="cuda")[:batch_size].to(
        torch.int64
    )
    return (
        pool,
        loc,
        latent[:, :NOPE_DIM],
        latent[:, NOPE_DIM:],
        query[..., :NOPE_DIM],
        query[..., NOPE_DIM:],
    )


class TestKimiK3PrerequisiteOps(CustomTestCase):
    def test_mla_scatter_concat_bf16_and_fp8(self):
        batch_size, num_heads = 64, 8
        pool, loc, k_nope, k_rope, q_nope, q_rope = _make_mla_inputs(
            batch_size, num_heads, seed=0
        )

        if not can_use_set_mla_kv_concat_q(NOPE_DIM * 2, ROPE_DIM * 2):
            self.skipTest("fused MLA scatter+concat requires SM90+")
        pool_ref = pool.clone()
        query = set_mla_kv_concat_q(pool, loc, k_nope, k_rope, q_nope, q_rope)
        set_mla_kv_buffer(pool_ref, loc, k_nope, k_rope)
        query_ref = concat_mla_absorb_q(q_nope, q_rope)
        self.assertTrue(torch.equal(pool, pool_ref))
        self.assertTrue(torch.equal(query, query_ref))

        if not can_use_set_mla_kv_concat_q_fp8():
            self.skipTest("fused FP8 MLA scatter+concat requires SM90+")
        fp8_pool = torch.zeros(
            MLA_PAGES, MLA_DIM, device="cuda", dtype=torch.float8_e4m3fn
        )
        fp8_ref = fp8_pool.clone()
        fp8_query = set_mla_kv_concat_q_fp8(
            fp8_pool, loc, k_nope, k_rope, q_nope, q_rope
        )
        row = torch.cat([k_nope, k_rope], dim=-1).to(torch.float8_e4m3fn)
        fp8_ref[loc] = row
        fp8_query_ref = concat_mla_absorb_q_general(q_nope, q_rope).to(
            torch.float8_e4m3fn
        )
        self.assertTrue(
            torch.equal(fp8_pool.view(torch.uint8), fp8_ref.view(torch.uint8))
        )
        self.assertTrue(
            torch.equal(
                fp8_query.view(torch.uint8),
                fp8_query_ref.view(torch.uint8),
            )
        )

    def test_replayssm_ring_fold(self):
        batch_size, num_steps = 8, 4
        num_value_heads, num_key_heads = 8, 2
        key_dim = value_dim = 128
        ring_size = 16
        torch.manual_seed(6)

        q = torch.randn(
            batch_size,
            num_steps,
            num_key_heads,
            key_dim,
            device="cuda",
        )
        k = torch.randn_like(q)
        v = torch.randn(
            batch_size,
            num_steps,
            num_value_heads,
            value_dim,
            device="cuda",
        )
        a = torch.randn(
            batch_size,
            num_steps,
            num_value_heads,
            key_dim,
            device="cuda",
        )
        b = torch.randn(batch_size, num_steps, num_value_heads, device="cuda")
        a_log = torch.randn(num_value_heads, device="cuda")
        dt_bias = torch.randn(num_value_heads, key_dim, device="cuda")
        slots = torch.arange(1, batch_size + 1, device="cuda", dtype=torch.int32)
        slots[-1] = -1
        num_slots = batch_size + 1
        state = torch.randn(
            num_slots,
            num_value_heads,
            value_dim,
            key_dim,
            device="cuda",
        )
        intermediate = torch.zeros(
            num_slots,
            num_steps,
            num_value_heads,
            value_dim,
            key_dim,
            device="cuda",
        )
        raw_v = torch.zeros(
            num_slots,
            num_value_heads,
            ring_size,
            value_dim,
            device="cuda",
        )
        raw_k = torch.zeros(
            num_slots,
            num_key_heads,
            ring_size,
            key_dim,
            device="cuda",
        )
        gate = torch.zeros_like(raw_v)
        beta = torch.zeros(
            num_slots,
            num_value_heads,
            ring_size,
            device="cuda",
        )

        fused_sigmoid_gating_delta_rule_update(
            A_log=a_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=state,
            initial_state_indices=slots,
            scale=key_dim**-0.5,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            lower_bound=-5.0,
            disable_state_update=True,
            intermediate_states_buffer=intermediate,
            intermediate_state_indices=slots,
            cache_steps=num_steps,
            cache_ring=True,
            replayssm_rawv=raw_v,
            replayssm_rawk=raw_k,
            replayssm_g=gate,
            replayssm_beta=beta,
        )
        checkpoint = state.clone()
        commit_kda_replayssm_spec(
            checkpoint,
            raw_v,
            raw_k,
            gate,
            beta,
            slots,
            torch.full((batch_size,), num_steps, device="cuda", dtype=torch.int32),
            max_cache_len=ring_size,
            num_k_heads=num_key_heads,
            use_qk_l2norm_in_kernel=True,
            null_block_id=-1,
        )
        for slot in slots[:-1].tolist():
            expected = intermediate[slot, num_steps - 1]
            actual = checkpoint[slot]
            relative_error = (
                actual - expected
            ).abs().max() / expected.abs().max().clamp_min(1e-6)
            self.assertLess(relative_error.item(), 1e-3)

    def test_add3_bit_exact(self):
        torch.manual_seed(0)
        tensors = [
            torch.randn(9, 112, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        actual = add3.add3(*tensors, prefetch_bc=True)
        expected = (tensors[0] + tensors[1]) + tensors[2]
        self.assertTrue(torch.equal(actual, expected))

    def test_moe_auxiliary_kernels(self):
        x = torch.randn(2, TOPK, 7168, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(2, 7168, device="cuda", dtype=torch.bfloat16)
        self.assertIs(moe_topk_sum(x, out), out)
        self.assertTrue(torch.equal(out, x.float().sum(1).to(torch.bfloat16)))

    def test_moe_route_and_quant(self):
        torch.manual_seed(1)
        scores = torch.randn(8, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(NUM_EXPERTS, device="cuda", dtype=torch.float32)
        args = (scores, bias, TOPK, True, 2.5, True)
        weights, ids = route_radix(*args, sorted=True)
        # Oracle, NOT moe_fused_gate: for this exact configuration (sigmoid, no
        # shared experts, no expert groups, no softcapping) moe_fused_gate
        # dispatches straight back to route_radix, so using it as the reference
        # compares the kernel with itself and cannot see a selection, tie-break,
        # NaN, renormalize or scaling error.
        ref_weights, ref_ids = _route_oracle(*args, sorted=True)
        self.assertTrue(torch.equal(ids, ref_ids))
        # rtol is not 1e-6: the kernel computes sigmoid with __fdividef/__expf,
        # whose last bits differ from torch's. The old self-comparison could
        # afford atol=0; a real oracle cannot.
        torch.testing.assert_close(weights, ref_weights, rtol=1e-5, atol=1e-6)

        if not moe_route_quant_fused.available():
            self.skipTest("fused route+quant kernel unavailable")
        hidden = torch.randn(8, 3584, device="cuda", dtype=torch.bfloat16)
        ref_weights, ref_ids = route_radix(*args, sorted=False)
        ref_packed = PackTopkIds.execute(ref_ids, ref_weights)
        ref_q, ref_scale = per_token_group_quant(
            hidden, group_size=32, scale_ue8m0=True
        )
        actual = moe_route_quant_fused.route_quant_fused(
            scores,
            bias,
            hidden,
            TOPK,
            renormalize=True,
            routed_scaling_factor=2.5,
            apply_scale=True,
        )
        weights, ids, packed, quantized, scale = actual
        self.assertTrue(torch.equal(ids, ref_ids))
        self.assertTrue(
            torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
        )
        self.assertTrue(torch.equal(packed, ref_packed))
        self.assertTrue(
            torch.equal(quantized.view(torch.uint8), ref_q.view(torch.uint8))
        )
        torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)

    def test_route_radix_ties_and_nan(self):
        """The cases the self-comparison could not see.

        Exact ties: many experts share one biased value, so the winner set is only
        determined by the lowest-id rule. NaN: floored, so a NaN expert must never
        be selected while enough finite ones exist. Both run with renormalize and
        scaling on and off, since those are applied in a fixed order.
        """
        bias = torch.zeros(NUM_EXPERTS, device="cuda", dtype=torch.float32)

        tied = torch.full((4, NUM_EXPERTS), 0.25, device="cuda", dtype=torch.bfloat16)
        # a handful of strict winners above the tied plateau, the rest exactly equal
        tied[:, 300] = 2.0
        tied[:, 7] = 2.0
        tied[:, 800] = 1.5

        nan_scores = torch.randn(4, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
        nan_scores[:, 100] = float("nan")
        nan_scores[:, 500] = float("nan")
        # make the NaN experts the ones that would otherwise win outright
        nan_scores[:, 101] = 5.0

        for name, scores in (("ties", tied), ("nan", nan_scores)):
            for renormalize in (False, True):
                for apply_scale in (False, True):
                    for sorted_ in (False, True):
                        args = (scores, bias, TOPK, renormalize, 2.5, apply_scale)
                        ids = route_radix(*args, sorted=sorted_)[1]
                        ref_ids = _route_oracle(*args, sorted=sorted_)[1]
                        tag = (
                            f"{name} renorm={renormalize} "
                            f"scale={apply_scale} sorted={sorted_}"
                        )
                        self.assertTrue(torch.equal(ids, ref_ids), msg=tag)
                        if name == "nan":
                            self.assertFalse(
                                bool(((ids == 100) | (ids == 500)).any()),
                                msg=f"{tag}: a NaN expert was selected",
                            )

    def test_tiny_gemm_variants(self):
        torch.manual_seed(2)
        x = torch.randn(2, 7168, device="cuda", dtype=torch.bfloat16) / 8
        weight = torch.randn(144, 7168, device="cuda", dtype=torch.bfloat16) / 8
        actual = tiny_n_gemm_bf16(x, weight, out_dtype=torch.float32)
        torch.testing.assert_close(
            actual.double(), x.double() @ weight.double().t(), rtol=1e-3, atol=1e-3
        )

        x = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16) / 4
        weight = torch.randn(1536, 128, device="cuda", dtype=torch.bfloat16) / 4
        actual = tiny_k_gemm_bf16(x, weight)
        torch.testing.assert_close(
            actual.double(), x.double() @ weight.double().t(), rtol=2e-2, atol=2e-2
        )

    def test_top_p_renorm(self):
        torch.manual_seed(3)
        probs = torch.randn(3, 1024, device="cuda").softmax(-1)
        top_p = torch.tensor([0.5, 0.8, 0.95], device="cuda")
        sorted_probs = probs.sort(-1).values
        cutoff = torch.searchsorted(
            sorted_probs.cumsum(-1), (1 - top_p).unsqueeze(1)
        ).squeeze(1)
        cutoff.clamp_(max=probs.shape[1] - 1)
        pivot = sorted_probs.gather(1, cutoff[:, None])
        expected = torch.where(probs >= pivot, probs, 0)
        expected /= expected.sum(-1, keepdim=True)
        torch.testing.assert_close(
            top_p_renorm_probs_triton(probs, top_p),
            expected,
            rtol=2e-6,
            atol=1e-8,
        )

    def test_vision_rope(self):
        torch.manual_seed(4)
        qkv = torch.randn(480, 3, 12, 128, device="cuda", dtype=torch.bfloat16)
        q, k, _ = qkv.unbind(1)
        angles = torch.randn(480, 64, device="cuda")
        freqs = torch.polar(torch.ones_like(angles), angles)
        freqs_expanded = freqs.unsqueeze(-2)

        def reference(x):
            value = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
            return torch.view_as_real(value * freqs_expanded).flatten(-2).type_as(x)

        actual_q, actual_k = apply_fused_qk_complex_rope(q, k, freqs)
        atol = 2 * torch.finfo(torch.bfloat16).eps
        torch.testing.assert_close(actual_q, reference(q), rtol=0, atol=atol)
        torch.testing.assert_close(actual_k, reference(k), rtol=0, atol=atol)

    def test_vision_rope_inplace(self):
        # VisionAttention hands the applier contiguous q/k, which is what the
        # in-place kernel requires; mirror that rather than qkv.unbind views.
        for dtype in (torch.bfloat16, torch.float16):
            torch.manual_seed(4)
            q = torch.randn(480, 12, 128, device="cuda", dtype=dtype)
            k = torch.randn(480, 12, 128, device="cuda", dtype=dtype)
            angles = torch.randn(480, 64, device="cuda")
            freqs = torch.polar(torch.ones_like(angles), angles)
            freqs_expanded = freqs.unsqueeze(-2)

            def reference(x):
                value = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
                return torch.view_as_real(value * freqs_expanded).flatten(-2).type_as(x)

            expected_q, expected_k = reference(q), reference(k)
            prepared = prepare_fused_qk_complex_rope_inplace(freqs)
            actual_q, actual_k = apply_fused_qk_complex_rope_inplace(q, k, prepared)

            atol = 2 * torch.finfo(dtype).eps
            torch.testing.assert_close(actual_q, expected_q, rtol=0, atol=atol)
            torch.testing.assert_close(actual_k, expected_k, rtol=0, atol=atol)

    def test_vision_rope_inplace_rejects_non_complex_frequencies(self):
        with self.assertRaises(ValueError):
            prepare_fused_qk_complex_rope_inplace(torch.randn(8, 64, device="cuda"))

    def test_normalize_and_patchify(self):
        torch.manual_seed(5)
        image = torch.randn(2, 3, 17, 19, device="cuda")
        scale = torch.randn(1, 3, 1, 1, device="cuda")
        bias = torch.randn(1, 3, 1, 1, device="cuda")
        args = (image, scale, bias, 4, 20, 20)
        actual = normalize_and_patchify(
            args[0],
            args[1],
            args[2],
            patch_size=args[3],
            padded_height=args[4],
            padded_width=args[5],
        )
        expected = _normalize_and_patchify_torch(
            args[0],
            args[1],
            args[2],
            patch_size=args[3],
            padded_height=args[4],
            padded_width=args[5],
        )
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
