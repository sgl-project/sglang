"""Cake sparse / block-sparse attention and indexer kernels through sglang.kernels.

Registry resolution for every ``attention_sparse`` adapter; bitwise parity of
the facade against direct FlashInfer calls; FP32 torch references (BF16 1e-2,
Sage-FP8 5e-2 as measured by FlashInfer). GPU tests skip when FlashInfer lacks
the module or the device is outside sm_100a / sm_103a. The MSA NVFP4 decode
and sparse MQA plans need FlashInfer's test-only input packers and are
registry-tested only; dense MQA is parity-tested on 148/152-SM parts.
"""

import math
import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import attention_sparse as cake
from sglang.kernels.cake_kernels.attention_common import flashinfer_module_available
from sglang.kernels.ops.attention.cake import (
    cake_bsa_attn_sm100_blk64_sage_fwd,
    cake_create_block_sparse_attention_wrapper,
    cake_dsa_indexer_topk,
    cake_prepare_dense_mqa_logits,
    cake_prepare_dsa_indexer_topk,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "create_block_sparse_attention_wrapper",
    "create_variable_block_sparse_attention_wrapper_sm90",
    "bsa_attn_sm100_blk64_sage_fwd",
    "bsa_attn_sm120_blk64_sage_fwd",
    "prepare_msa_nvfp4_sparse_decode",
    "dsa_indexer_topk",
    "prepare_dsa_indexer_topk",
    "prepare_dense_mqa_logits",
    "prepare_sparse_mqa_metadata",
    "prepare_sparse_mqa_logits",
)


@pytest.mark.parametrize("name", OPS)
def test_registry_resolves_flashinfer_backend(name):
    spec = select_kernel(f"attention.{name}", backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.attention_sparse:")
    assert spec.load() is not None


def _skip_unless(archs, *modules):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(*modules):
        pytest.skip(f"installed FlashInfer lacks {', '.join(modules)}")
    cc = torch.cuda.get_device_capability()
    if cc not in archs:
        pytest.skip(f"Cake kernel is built for {archs}, device is {cc}")


def _ceil_div(a, b):
    return -(-a // b)


# --------------------------------------------------------------------------
# SM100 / SM103 VSA block-sparse attention
# --------------------------------------------------------------------------


def test_vsa_block_sparse_bf16_matches_flashinfer_and_dense_reference():
    _skip_unless(cake.VSA_ARCHS, cake.FI_VSA_MODULE, cake.FI_SPARSE_MODULE)
    device = torch.device("cuda")
    torch.manual_seed(20)
    block, heads, head_dim, M, N, selected = 128, 8, 128, 256, 512, 2
    mb, nb = M // block, N // block
    mask = torch.zeros((heads, mb, nb), dtype=torch.bool, device=device)
    for row in range(mb):
        mask[:, row, (torch.arange(selected, device=device) * 7 + row) % nb] = True
    q = torch.randn((M, heads, head_dim), dtype=torch.bfloat16, device=device)
    k = torch.randn((N, heads, head_dim), dtype=torch.bfloat16, device=device)
    v = torch.randn((N, heads, head_dim), dtype=torch.bfloat16, device=device)
    assert cake.supports_block_sparse_attention(
        q,
        k,
        v,
        block_size=block,
        num_qo_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
    )
    workspace = torch.empty(128 << 20, dtype=torch.uint8, device=device)
    plan_args = (None, None, M, N, block, block, heads, heads, head_dim)
    plan_kwargs = dict(
        q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16, block_mask=mask
    )
    wrapper = cake_create_block_sparse_attention_wrapper(workspace)
    wrapper.plan(*plan_args, **plan_kwargs)
    out, lse = wrapper.run(q, k, v, return_lse=True)

    from flashinfer.sparse import BlockSparseAttentionWrapper

    fi_wrapper = BlockSparseAttentionWrapper(workspace, backend="cake")
    fi_wrapper.plan(*plan_args, **plan_kwargs)
    out_fi, lse_fi = fi_wrapper.run(q, k, v, return_lse=True)
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(lse, lse_fi)

    scores = torch.einsum("mhd,nhd->hmn", q.float(), k.float()) / math.sqrt(head_dim)
    dense = mask.repeat_interleave(block, 1).repeat_interleave(block, 2)
    scores.masked_fill_(~dense, float("-inf"))
    ref = torch.einsum("hmn,nhd->mhd", torch.softmax(scores, dim=-1), v.float())
    torch.testing.assert_close(out.float(), ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        lse, torch.logsumexp(scores, dim=-1).transpose(0, 1), atol=1e-2, rtol=1e-2
    )


# --------------------------------------------------------------------------
# Sage-FP8 block-sparse attention (SM100 / SM103)
# --------------------------------------------------------------------------


def _sage_reference(q8, k8, v8, q_scale, k_scale, v_scale, index, count, scale):
    q, k, v = (t.float().transpose(1, 2) for t in (q8, k8, v8))
    batch, heads, sq, dim = q.shape
    kv_heads, sk = k.shape[1], k.shape[2]
    group = heads // kv_heads
    k_tok_scale = k_scale.repeat_interleave(16, dim=-1)[..., :sk]
    if v_scale.ndim == 2:
        v_scale = v_scale.unsqueeze(0).expand(batch, kv_heads, dim)
    out = torch.zeros((batch, heads, sq, dim), device=q.device)
    lse = torch.full((batch, heads, sq), float("-inf"), device=q.device)
    for b in range(batch):
        for h in range(heads):
            kh = h // group
            for qb in range(_ceil_div(sq, 64)):
                q0, q1 = qb * 64, min(sq, (qb + 1) * 64)
                tok = torch.cat(
                    [
                        torch.arange(blk * 64, min(sk, blk * 64 + 64), device=q.device)
                        for blk in index[b, h, qb, :count].tolist()
                    ]
                )
                kk = k[b, kh, tok] * k_tok_scale[b, kh, tok, None]
                vv = v[b, kh, tok] * v_scale[b, kh][None, :]
                qq = q[b, h, q0:q1] * q_scale[b, h, q0:q1, None]
                s = (qq @ kk.T) * scale
                m = s.amax(dim=-1, keepdim=True)
                p = torch.exp(s - m)
                d = p.sum(dim=-1, keepdim=True)
                out[b, h, q0:q1] = (p / d) @ vv
                lse[b, h, q0:q1] = (m + torch.log(d)).squeeze(-1)
    return out.transpose(1, 2).contiguous(), lse


def test_sage_sm100_block_sparse_matches_flashinfer_and_reference():
    _skip_unless(
        cake.SAGE_SM100_ARCHS,
        cake.FI_BSA_SM100_MODULE,
        cake.FI_SAGE_SM100_MODULE,
        cake.FI_SAGE_JIT_MODULE,
    )
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd
    from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import sage_fp8_quantize_sm100

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(21)
    batch, heads, kv_heads, sq, sk, sel = 1, 8, 8, 256, 256, 2
    q = torch.randn((batch, sq, heads, 128), device=device, generator=gen).bfloat16()
    k = torch.randn((batch, sk, kv_heads, 128), device=device, generator=gen).bfloat16()
    v = torch.randn((batch, sk, kv_heads, 128), device=device, generator=gen).bfloat16()
    q8, k8, v8, q_scale, k_scale, v_scale = sage_fp8_quantize_sm100(q, k, v)
    k_blocks = _ceil_div(sk, 64)
    scores = torch.rand(
        (batch, heads, _ceil_div(sq, 64), k_blocks), device=device, generator=gen
    )
    index = (
        scores.argsort(dim=-1, descending=True)[..., :sel].to(torch.int32).contiguous()
    )
    assert cake.supports_bsa_attn_sm100_blk64_sage_fwd(
        q8, k8, v8, index, q_scale=q_scale, k_scale=k_scale, v_scale=v_scale
    )
    scale = 128**-0.5
    out, lse = cake_bsa_attn_sm100_blk64_sage_fwd(
        q8,
        k8,
        v8,
        index,
        sel,
        softmax_scale=scale,
        return_lse=True,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    out_fi, lse_fi = bsa_attn_sm100_blk64_fwd(
        q8,
        k8,
        v8,
        index,
        sel,
        softmax_scale=scale,
        return_lse=True,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        backend="cake",
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_fi)
    assert torch.equal(lse, lse_fi)
    ref, ref_lse = _sage_reference(
        q8, k8, v8, q_scale, k_scale, v_scale, index, sel, scale
    )
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)


# --------------------------------------------------------------------------
# DSA indexer top-k (one-shot + prepared)
# --------------------------------------------------------------------------


def test_dsa_indexer_topk_matches_flashinfer_and_bruteforce():
    _skip_unless(
        cake.DSA_INDEXER_ARCHS,
        cake.FI_DSA_INDEXER_MODULE,
        cake.FI_DSA_INDEXER_BACKEND_MODULE,
    )
    from flashinfer.experimental.cake_dsa_indexer.cake_backend import (
        generated_program_available,
    )

    device = torch.device("cuda")
    if not generated_program_available(device):
        pytest.skip("no generated DSA indexer program registered")
    gen = torch.Generator(device=device).manual_seed(22)
    seg_q, seg_k, top_k = [64, 32], [256, 128], 16
    T, Tkv = sum(seg_q), sum(seg_k)
    q = torch.randn((T, 32, 128), generator=gen, device=device).to(torch.bfloat16)
    k = torch.randn((Tkv, 128), generator=gen, device=device).to(torch.bfloat16)
    w = torch.randn((T, 32), generator=gen, device=device) * (32**-0.5)
    cu_q = torch.tensor([0, seg_q[0], T], dtype=torch.int32, device=device)
    cu_k = torch.tensor([0, seg_k[0], Tkv], dtype=torch.int32, device=device)
    scale = 128**-0.5
    assert cake.supports_dsa_indexer_topk(q, k, w, cu_q, cu_k, top_k=top_k)
    indices, scores = cake_dsa_indexer_topk(
        q, k, w, cu_q, cu_k, top_k=top_k, softmax_scale=scale
    )
    from flashinfer.dsa_indexer import dsa_indexer_topk

    indices_fi, scores_fi = dsa_indexer_topk(
        q, k, w, cu_q, cu_k, top_k=top_k, softmax_scale=scale, backend="cake"
    )
    torch.cuda.synchronize()
    assert torch.equal(indices, indices_fi)
    assert torch.equal(scores, scores_fi)

    # Brute force: s[t, j] = sum_h w[t, h] * relu(scale * q[t, h] . k[j]) over
    # the causally visible keys (queries sit at the tail of their key segment).
    q_start, k_start = 0, 0
    for lq, lk in zip(seg_q, seg_k):
        logits = torch.einsum(
            "thd,jd->thj",
            q[q_start : q_start + lq].float(),
            k[k_start : k_start + lk].float(),
        )
        ref = torch.einsum(
            "th,thj->tj", w[q_start : q_start + lq], torch.relu(logits * scale)
        )
        for u in range(lq):
            visible = lk - lq + u + 1
            n = min(top_k, visible)
            row = indices[q_start + u].tolist()
            assert all(0 <= j < visible for j in row[:n]), row[:n]
            assert row[:n] == sorted(set(row[:n]))
            assert all(j == -1 for j in row[n:])
            assert torch.isneginf(scores[q_start + u, n:]).all()
            chosen = torch.tensor(row[:n], device=device)
            torch.testing.assert_close(
                scores[q_start + u, :n], ref[u, chosen], atol=1e-3, rtol=1e-3
            )
            kth = torch.topk(ref[u, :visible], n).values[-1]
            assert (ref[u, chosen] >= kth - 1e-3).all()
        q_start += lq
        k_start += lk

    workspace = torch.empty(
        cake.dsa_indexer_topk_workspace_size(top_k, device),
        dtype=torch.uint8,
        device=device,
    )
    runner = cake_prepare_dsa_indexer_topk(
        q,
        k,
        w,
        cu_q,
        cu_k,
        top_k=top_k,
        softmax_scale=scale,
        workspace_buffer=workspace,
        indices=torch.empty_like(indices),
        scores=torch.empty_like(scores),
    )
    r_indices, r_scores = runner.run()
    torch.cuda.synchronize()
    assert torch.equal(r_indices, indices)
    assert torch.equal(r_scores, scores)


# --------------------------------------------------------------------------
# Dense MQA logits (parity with FlashInfer; 148 / 152-SM parts only)
# --------------------------------------------------------------------------


def test_dense_mqa_fp8_logits_parity():
    _skip_unless(
        cake.MQA_ARCHS, cake.FI_DENSE_MQA_MODULE, cake.FI_DENSE_MQA_BACKEND_MODULE
    )
    from flashinfer.dense_mqa import prepare_dense_mqa_logits
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa as runtime

    device = torch.device("cuda")
    try:
        arch = runtime.device_arch(device)
    except RuntimeError as error:
        pytest.skip(str(error))
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    if sms not in runtime.supported_num_sms(arch):
        pytest.skip(
            f"exported {arch} schedules cover {runtime.supported_num_sms(arch)} SMs"
        )
    torch.manual_seed(23)
    queries, keys = 16, 4096
    q = torch.randn(queries, 32, 128, device=device).to(torch.float8_e4m3fn)
    kv = torch.randn(keys, 128, device=device).to(torch.float8_e4m3fn)
    ks = torch.ones(keys, device=device)
    weights = torch.randn(queries, 32, device=device)
    starts = torch.zeros(queries, device=device, dtype=torch.int32)
    ends = torch.full_like(starts, keys)
    ends[1::3] = 17
    assert cake.supports_dense_mqa_logits("fp8", q, kv, weights, starts, ends)
    plan = cake_prepare_dense_mqa_logits(
        "fp8", q, kv, weights, starts, ends, kv_scales=ks
    )
    plan.run()
    plan_fi = prepare_dense_mqa_logits(
        "fp8", q, kv, weights, starts, ends, kv_scales=ks
    )
    plan_fi.run()
    torch.cuda.synchronize()
    assert torch.equal(plan.logical_output, plan_fi.logical_output)
    finite = torch.isfinite(plan.logical_output)
    assert finite[0].all() and torch.isneginf(plan.logical_output[1, 17:]).all()
    # Masked FP32 reference over the visible window (FP8 operands, rows 0 and 2).
    ref = torch.einsum(
        "th,thj->tj",
        weights,
        torch.relu(torch.einsum("thd,jd->thj", q.float(), kv.float())),
    )
    torch.testing.assert_close(plan.logical_output[0], ref[0], atol=0.1, rtol=0.1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
