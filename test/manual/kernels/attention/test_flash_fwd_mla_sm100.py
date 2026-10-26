# Copyright (c) 2026, Colfax International.
"""Correctness check and benchmark for FlashAttentionMLAForwardSm100 (SM100)."""

import math
import time

import cuda.bindings.driver as cuda
import cutlass.cute as cute
import torch
import torch.utils.benchmark as benchmark
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.attention.flash_attn.cute.cute_dsl_utils import (
    dump_kernel_attributes,
)
from sglang.kernels.ops.attention.flash_attn.cute.flash_fwd_mla_sm100 import (
    FlashAttentionMLAForwardSm100,
)
from sglang.kernels.ops.attention.flash_attn.cute.testing import attention_ref


def test_mla_kernel(
    seqlen_q=2048,
    seqlen_k=2048,
    topk_length=2048,
    nheads=1,
    batch=1,
    iter=0,
    compile_cache=dict(),
    validate=True,
    seed=0,
    gather_kv=True,
    pack_gqa=False,
    is_causal=False,
    varlen_q=False,
    varlen_k=False,
    disable_bitmask=False,
    has_qk=True,
    store_P=False,
):
    torch.manual_seed(seed)
    hdim = 64
    hdimv = 512
    softmax_scale = 1.0 / math.sqrt(hdim + hdimv) if has_qk else 1.0 / math.sqrt(hdimv)

    nheads_kv = 1
    qhead_per_kvhead = nheads
    seqlen_k_rounded = (seqlen_k + 128 - 1) // 128 * 128
    P_k_length = seqlen_k_rounded if not gather_kv else topk_length

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compile_key = (
        is_causal,
        gather_kv,
        topk_length if gather_kv else None,
        pack_gqa,
        qhead_per_kvhead,
        nheads_kv,
        varlen_q,
        varlen_k,
        disable_bitmask,
        has_qk,
    )
    if compile_key not in compile_cache:
        total_q_dummy = batch * seqlen_q
        total_k_dummy = batch * seqlen_k

        if varlen_q:
            Q = torch.randn(
                total_q_dummy, nheads, hdim, dtype=torch.bfloat16, device="cuda"
            )
            Qv = torch.randn(
                total_q_dummy, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
            )
            O = torch.empty(
                total_q_dummy, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
            )
            P = torch.empty(
                total_q_dummy, nheads, P_k_length, dtype=torch.bfloat16, device="cuda"
            )
            lse = torch.empty(total_q_dummy, nheads, dtype=torch.float32, device="cuda")
            row_max = torch.empty(
                total_q_dummy,
                nheads,
                P_k_length // 128,
                dtype=torch.float32,
                device="cuda",
            )
            index_topk = (
                torch.rand(total_q_dummy, topk_length, device="cuda")
                .argsort(dim=-1)
                .to(torch.int32)
            )
            cu_seqlens_q_dummy = torch.arange(
                0, (batch + 1) * seqlen_q, seqlen_q, dtype=torch.int32, device="cuda"
            )
        else:
            Q = torch.randn(
                batch, seqlen_q, nheads, hdim, dtype=torch.bfloat16, device="cuda"
            )
            Qv = torch.randn(
                batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
            )
            O = torch.empty(
                batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
            )
            P = torch.empty(
                batch, seqlen_q, nheads, P_k_length, dtype=torch.bfloat16, device="cuda"
            )
            lse = torch.empty(
                batch, seqlen_q, nheads, dtype=torch.float32, device="cuda"
            )
            row_max = torch.empty(
                batch,
                seqlen_q,
                nheads,
                P_k_length // 128,
                dtype=torch.float32,
                device="cuda",
            )
            index_topk = (
                torch.rand(batch, seqlen_q, topk_length, device="cuda")
                .argsort(dim=-1)
                .to(torch.int32)
            )

        if varlen_k:
            K = torch.randn(
                total_k_dummy, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda"
            )
            V = torch.randn(
                total_k_dummy, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda"
            )
            cu_seqlens_k_dummy = torch.arange(
                0, (batch + 1) * seqlen_k, seqlen_k, dtype=torch.int32, device="cuda"
            )
        else:
            K = torch.randn(
                batch, seqlen_k, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda"
            )
            V = torch.randn(
                batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda"
            )

        mQ = from_dlpack(Q, assumed_align=16).mark_layout_dynamic(
            leading_dim=Q.ndim - 1
        )
        mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(
            leading_dim=Qv.ndim - 1
        )
        mK = from_dlpack(K, assumed_align=16).mark_layout_dynamic(
            leading_dim=K.ndim - 1
        )
        mV = from_dlpack(V, assumed_align=16).mark_layout_dynamic(
            leading_dim=V.ndim - 1
        )
        mO = from_dlpack(O, assumed_align=16).mark_layout_dynamic(
            leading_dim=O.ndim - 1
        )
        mP = from_dlpack(P, assumed_align=16).mark_layout_dynamic(
            leading_dim=P.ndim - 1
        )
        mLSE = from_dlpack(lse, assumed_align=4).mark_layout_dynamic(
            leading_dim=lse.ndim - 1
        )
        mRowMax = from_dlpack(row_max, assumed_align=4).mark_layout_dynamic(
            leading_dim=row_max.ndim - 1
        )
        if gather_kv:
            mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(
                leading_dim=index_topk.ndim - 1
            )
        else:
            mIndexTopk = None

        compile_kwargs = dict(mIndexTopk=mIndexTopk)
        if varlen_q:
            compile_kwargs["mCuSeqlensQ"] = from_dlpack(
                cu_seqlens_q_dummy, assumed_align=4
            )
        if varlen_k:
            compile_kwargs["mCuSeqlensK"] = from_dlpack(
                cu_seqlens_k_dummy, assumed_align=4
            )

        if not has_qk:
            mQ = mK = None

        if store_P is False:
            mP = mRowMax = None

        kernel = cute.compile(
            FlashAttentionMLAForwardSm100(
                is_causal=is_causal,
                use_cpasync_load_KV=gather_kv,
                topk_length=topk_length if gather_kv else 2048,
                is_topk_gather=gather_kv,
                pack_gqa=pack_gqa,
                qhead_per_kvhead=qhead_per_kvhead,
                nheads_kv=nheads_kv,
                is_varlen_q=varlen_q,
                disable_bitmask=disable_bitmask,
                has_qk=has_qk,
            ),
            mQ,
            mQv,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale,
            mP,
            mRowMax,
            **compile_kwargs,
            stream=stream,
            options="--keep-ptx --keep-cubin --generate-line-info",
        )
        dump_kernel_attributes(kernel)
        compile_cache[compile_key] = kernel

    # ================================================================
    # ---- Generate variable seqlens for this run ----
    if varlen_q:
        torch.manual_seed(seed + 1000)
        # When causal without varlen_k, every per-batch seqlen_q must not exceed seqlen_k.
        max_seqlen_q = seqlen_k if (is_causal and not varlen_k) else seqlen_q
        seqlens_q = torch.randint(1, max_seqlen_q + 1, (batch,), dtype=torch.int32)
        cu_seqlens_q = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_q[1:] = seqlens_q.cumsum(0).to(torch.int32).cuda()
        total_q = cu_seqlens_q[-1].item()
    else:
        seqlens_q = torch.full((batch,), seqlen_q, dtype=torch.int32)
        total_q = None  # unused

    if varlen_k:
        torch.manual_seed(seed + 2000)
        # Each batch item must have at least topk_length keys so topk gather is valid.
        min_seqlen_k = topk_length if gather_kv else 1
        seqlens_k = torch.randint(
            min_seqlen_k, seqlen_k + 1, (batch,), dtype=torch.int32
        )
        # When causal, every batch item needs seqlens_k[b] >= seqlens_q[b].
        if is_causal:
            seqlens_k = torch.maximum(seqlens_k, seqlens_q)
        cu_seqlens_k = torch.zeros(batch + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_k[1:] = seqlens_k.cumsum(0).to(torch.int32).cuda()
        total_k = cu_seqlens_k[-1].item()
    else:
        seqlens_k = torch.full((batch,), seqlen_k, dtype=torch.int32)
        total_k = None  # unused

    torch.manual_seed(seed)  # restore main seed before drawing actual tensors

    # ---- Allocate Q / Qv / O / lse ----
    if varlen_q:
        Q = torch.randn(total_q, nheads, hdim, dtype=torch.bfloat16, device="cuda")
        Qv = torch.randn(total_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda")
        O = torch.empty(total_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda")
        P = torch.empty(
            total_q, nheads, P_k_length, dtype=torch.bfloat16, device="cuda"
        )
        lse = torch.empty(total_q, nheads, dtype=torch.float32, device="cuda")
        row_max = torch.empty(
            total_q_dummy, P_k_length // 128, nheads, dtype=torch.float32, device="cuda"
        )
    else:
        Q = torch.randn(
            batch, seqlen_q, nheads, hdim, dtype=torch.bfloat16, device="cuda"
        )
        Qv = torch.randn(
            batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
        )
        O = torch.empty(
            batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
        )
        P = torch.empty(
            batch, seqlen_q, nheads, P_k_length, dtype=torch.bfloat16, device="cuda"
        )
        lse = torch.empty(batch, seqlen_q, nheads, dtype=torch.float32, device="cuda")
        row_max = torch.empty(
            batch,
            seqlen_q,
            P_k_length // 128,
            nheads,
            dtype=torch.float32,
            device="cuda",
        )

    # ---- Allocate K / V ----
    if varlen_k:
        K = torch.randn(total_k, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(total_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda")
    else:
        K = torch.randn(
            batch, seqlen_k, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda"
        )
        V = torch.randn(
            batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda"
        )

    # ---- Generate index_topk with per-batch valid ranges when varlen_k ----
    # index_topk shape: (total_q, topk_length) if varlen_q else (batch, seqlen_q, topk_length)
    if gather_kv:
        topk_parts = []
        for b in range(batch):
            sl_q_b = seqlens_q[b].item()
            sl_k_b = seqlens_k[b].item()
            # Draw topk_length unique indices from [0, sl_k_b) for each query in this batch item.
            topk_b = (
                torch.rand(sl_q_b, sl_k_b, device="cuda")
                .argsort(dim=-1)[..., :topk_length]
                .to(torch.int32)
            )  # (sl_q_b, topk_length), all < sl_k_b
            topk_parts.append(topk_b)

        if varlen_q:
            index_topk = torch.cat(topk_parts, dim=0)  # (total_q, topk_length)
        else:
            index_topk = torch.stack(
                topk_parts, dim=0
            )  # (batch, seqlen_q, topk_length)
    else:
        index_topk = None

    # ---- Reference computation (per-batch loop covers all four varlen combos) ----
    O_ref_list, O_pt_list, lse_ref_list, lse_pt_list = [], [], [], []
    for b in range(batch):
        qs = cu_seqlens_q[b].item() if varlen_q else b * seqlen_q
        qe = cu_seqlens_q[b + 1].item() if varlen_q else (b + 1) * seqlen_q
        ks = cu_seqlens_k[b].item() if varlen_k else b * seqlen_k
        ke = cu_seqlens_k[b + 1].item() if varlen_k else (b + 1) * seqlen_k

        Q_b = (
            Q[qs:qe].unsqueeze(0) if varlen_q else Q[b : b + 1]
        )  # (1, sl_q, nheads, hdim)
        Qv_b = (
            Qv[qs:qe].unsqueeze(0) if varlen_q else Qv[b : b + 1]
        )  # (1, sl_q, nheads, hdimv)
        K_b = (
            K[ks:ke].unsqueeze(0) if varlen_k else K[b : b + 1]
        )  # (1, sl_k, nheads_kv, hdim)
        V_b = (
            V[ks:ke].unsqueeze(0) if varlen_k else V[b : b + 1]
        )  # (1, sl_k, nheads_kv, hdimv)
        if gather_kv:
            topk_b = (
                index_topk[qs:qe].unsqueeze(0) if varlen_q else index_topk[b : b + 1]
            )
        else:
            topk_b = None

        O_b, _, lse_b = attention_ref(
            Q_b if has_qk else None,
            K_b if has_qk else None,
            V_b,
            qv=Qv_b,
            causal=is_causal,
            return_lse=True,
            gather_kv_indices=topk_b,
        )
        O_pt_b, _, lse_pt_b = attention_ref(
            Q_b if has_qk else None,
            K_b if has_qk else None,
            V_b,
            qv=Qv_b,
            causal=is_causal,
            upcast=False,
            reorder_ops=True,
            return_lse=True,
            gather_kv_indices=topk_b,
        )
        O_ref_list.append(O_b.squeeze(0))
        O_pt_list.append(O_pt_b.squeeze(0))
        lse_ref_list.append(lse_b.squeeze(0))
        lse_pt_list.append(lse_pt_b.squeeze(0))

    cat_dim_o = 0 if (varlen_q) else 0  # always 0: leading token/batch dim
    cat_dim_lse = -1 if (varlen_q) else -1  # always last: token dim

    if varlen_q:
        O_ref = torch.cat(O_ref_list, dim=0)  # (total_q, nheads, hdimv)
        O_pt = torch.cat(O_pt_list, dim=0)
        lse_ref = torch.cat(lse_ref_list, dim=-1)  # (nheads, total_q)
        lse_pt = torch.cat(lse_pt_list, dim=-1)
    else:
        O_ref = torch.stack(O_ref_list, dim=0)  # (batch, seqlen_q, nheads, hdimv)
        O_pt = torch.stack(O_pt_list, dim=0)
        lse_ref = torch.stack(lse_ref_list, dim=0)  # (batch, nheads, seqlen_q)
        lse_pt = torch.stack(lse_pt_list, dim=0)

    rtol = 2
    atol = 2 * (O_ref + 0.3 - 0.3 - O_ref).abs().max().item()

    # ---- CuTe tensor wrappers ----
    mQ = from_dlpack(Q, assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim - 1)
    mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(leading_dim=Qv.ndim - 1)
    mK = from_dlpack(K, assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim - 1)
    mV = from_dlpack(V, assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1)
    mO = from_dlpack(O, assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1)
    mP = from_dlpack(P, assumed_align=16).mark_layout_dynamic(leading_dim=P.ndim - 1)
    mLSE = from_dlpack(lse, assumed_align=4).mark_layout_dynamic(
        leading_dim=lse.ndim - 1
    )
    mRowMax = from_dlpack(row_max, assumed_align=4).mark_layout_dynamic(
        leading_dim=row_max.ndim - 1
    )
    if index_topk is not None:
        mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(
            leading_dim=index_topk.ndim - 1
        )
    else:
        mIndexTopk = None

    run_kwargs = dict(mIndexTopk=mIndexTopk)
    if varlen_q:
        run_kwargs["mCuSeqlensQ"] = from_dlpack(cu_seqlens_q, assumed_align=4)
    if varlen_k:
        run_kwargs["mCuSeqlensK"] = from_dlpack(cu_seqlens_k, assumed_align=4)

    if not has_qk:
        mQ = mK = None

    if store_P is False:
        mP = mRowMax = None

    # ---- Run kernel ----
    compile_cache[compile_key](
        mQ,
        mQv,
        mK,
        mV,
        mO,
        mLSE,
        softmax_scale,
        mP,
        mRowMax,
        **run_kwargs,
        stream=stream,
    )

    O_ref_max = O_ref.abs().max().item()
    O_max = O.abs().max().item()
    print(f"Pytorch O max = {O_ref_max} and our O max = {O_max}")
    print(f"Pytorch max O diff: {(O_pt - O_ref).abs().max().item()}")
    print(f"Pytorch mean O diff: {(O_pt - O_ref).abs().mean().item()}")
    print(f"Max abs diff O, O_ref: {(O - O_ref).abs().max().item()}")
    print(f"Mean abs diff O, O_ref: {(O - O_ref).abs().mean().item()}")

    lse = lse.transpose(-1, -2)
    lse_ref_max = lse_ref.abs().max().item()
    lse_max = lse.abs().max().item()
    print(f"Pytorch LSE max = {lse_ref_max} and our LSE max = {lse_max}")
    print(f"Pytorch LSE max diff: {(lse_pt - lse_ref).abs().max().item()}")
    print(f"Pytorch LSE mean diff: {(lse_pt - lse_ref).abs().mean().item()}")
    print(f"Max abs diff LSE: {(lse - lse_ref).abs().max().item()}")
    print(f"Mean abs diff LSE: {(lse - lse_ref).abs().mean().item()}")

    if validate:
        assert (O - O_ref).abs().max().item() <= rtol * (
            O_pt - O_ref
        ).abs().max().item() + atol
        varlen_tag = ""
        if varlen_q:
            varlen_tag += f", total_q:{total_q}"
        if varlen_k:
            varlen_tag += f", total_k:{total_k}"
        print(
            f"batch:{batch:3d}, nheads:{nheads:3d}, seqlen_q:{seqlen_q:5d}, seqlen_k:{seqlen_k:5d}"
            f"{varlen_tag}, iter:{iter:2d} PASSED"
        )
    else:
        print(mO)
        print(
            f"batch:{batch:3d}, nheads:{nheads:3d}, seqlen_q:{seqlen_q:5d}, seqlen_k:{seqlen_k:5d}"
            f", iter:{iter:2d} RUN (NOT TESTING CORRECTNESS)"
        )

    return None


def timeit(fn, *args, **kwargs):
    # Synchronize before timing
    torch.cuda.synchronize()

    # Warmup
    for _ in range(10):
        fn(*args, **kwargs)

    # Benchmark using PyTorch's Timer
    t = benchmark.Timer(
        stmt="fn(*args, **kwargs)", globals={"fn": fn, "args": args, "kwargs": kwargs}
    )

    # Time it multiple runs
    measurement = t.timeit(20)  # 20 repeats
    avg_time = measurement.mean  # Average time in seconds

    time.sleep(1)

    return avg_time


def benchmark_mla_kernel(
    batch=1,
    seqlen_q=2048,
    seqlen_k=2048,
    topk_length=2048,
    nheads=128,
    hdim=64,
    hdimv=512,
    compile_cache=dict(),
    gather_kv=True,
    is_causal=False,
    disable_bitmask=False,
    store_P=False,
):
    assert hdim == 64, "hdim must be 64"
    assert hdimv == 512, "hdimv must be 512"

    qhead_per_kvhead = nheads
    nheads_kv = 1
    pack_gqa = True
    softmax_scale = 1.0 / math.sqrt(hdim + hdimv)
    seqlen_k_rounded = (seqlen_k + 128 - 1) // 128 * 128
    P_k_length = seqlen_k_rounded if not gather_kv else topk_length

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    compile_key = (
        is_causal,
        gather_kv,
        topk_length if gather_kv else None,
        pack_gqa,
        qhead_per_kvhead,
        nheads_kv,
        disable_bitmask,
    )
    if compile_key not in compile_cache:
        Q = torch.randn(
            batch, seqlen_q, nheads, hdim, dtype=torch.bfloat16, device="cuda"
        )
        Qv = torch.randn(
            batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
        )
        K = torch.randn(
            batch, seqlen_k, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda"
        )
        V = torch.randn(
            batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda"
        )
        O = torch.empty(
            batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
        )
        P = torch.empty(
            batch, seqlen_q, nheads, P_k_length, dtype=torch.bfloat16, device="cuda"
        )
        index_topk = (
            torch.rand(batch, seqlen_q, topk_length, device="cuda")
            .argsort(dim=-1)
            .to(torch.int32)
        )

        mQ = from_dlpack(Q, assumed_align=16).mark_layout_dynamic(
            leading_dim=Q.ndim - 1
        )
        mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(
            leading_dim=Qv.ndim - 1
        )
        mK = from_dlpack(K, assumed_align=16).mark_layout_dynamic(
            leading_dim=K.ndim - 1
        )
        mV = from_dlpack(V, assumed_align=16).mark_layout_dynamic(
            leading_dim=V.ndim - 1
        )
        mO = from_dlpack(O, assumed_align=16).mark_layout_dynamic(
            leading_dim=O.ndim - 1
        )
        mP = from_dlpack(P, assumed_align=16).mark_layout_dynamic(
            leading_dim=P.ndim - 1
        )
        if gather_kv:
            mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(
                leading_dim=index_topk.ndim - 1
            )
        else:
            mIndexTopk = None

        mLSE = None

        if store_P is False:
            mP = None

        kernel = cute.compile(
            FlashAttentionMLAForwardSm100(
                is_causal=is_causal,
                use_cpasync_load_KV=gather_kv,
                topk_length=topk_length if gather_kv else 2048,
                is_topk_gather=gather_kv,
                pack_gqa=pack_gqa,
                qhead_per_kvhead=qhead_per_kvhead,
                nheads_kv=nheads_kv,
                disable_bitmask=disable_bitmask,
            ),
            mQ,
            mQv,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale,
            mP=mP,
            mIndexTopk=mIndexTopk,
            stream=stream,
        )
        compile_cache[compile_key] = kernel

    Q = torch.randn(batch, seqlen_q, nheads, hdim, dtype=torch.bfloat16, device="cuda")
    Qv = torch.randn(
        batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda"
    )
    K = torch.randn(
        batch, seqlen_k, nheads_kv, hdim, dtype=torch.bfloat16, device="cuda"
    )
    V = torch.randn(
        batch, seqlen_k, nheads_kv, hdimv, dtype=torch.bfloat16, device="cuda"
    )
    O = torch.empty(batch, seqlen_q, nheads, hdimv, dtype=torch.bfloat16, device="cuda")
    P = torch.empty(
        batch, seqlen_q, nheads, P_k_length, dtype=torch.bfloat16, device="cuda"
    )

    index_topk = (
        torch.rand(batch, seqlen_q, topk_length, device="cuda")
        .argsort(dim=-1)
        .to(torch.int32)
    )

    mQ = from_dlpack(Q, assumed_align=16).mark_layout_dynamic(leading_dim=Q.ndim - 1)
    mQv = from_dlpack(Qv, assumed_align=16).mark_layout_dynamic(leading_dim=Qv.ndim - 1)
    mK = from_dlpack(K, assumed_align=16).mark_layout_dynamic(leading_dim=K.ndim - 1)
    mV = from_dlpack(V, assumed_align=16).mark_layout_dynamic(leading_dim=V.ndim - 1)
    mO = from_dlpack(O, assumed_align=16).mark_layout_dynamic(leading_dim=O.ndim - 1)
    mP = from_dlpack(P, assumed_align=16).mark_layout_dynamic(leading_dim=P.ndim - 1)
    if gather_kv:
        mIndexTopk = from_dlpack(index_topk, assumed_align=16).mark_layout_dynamic(
            leading_dim=index_topk.ndim - 1
        )
    else:
        mIndexTopk = None
    mLSE = None

    if store_P is False:
        mP = None

    exec_time_in_s = timeit(
        compile_cache[compile_key],
        mQ,
        mQv,
        mK,
        mV,
        mO,
        mLSE,
        softmax_scale,
        mP=mP,
        mIndexTopk=mIndexTopk,
        stream=stream,
    )

    seqlen_k_eff = topk_length if gather_kv else seqlen_k

    FLOPs = 2 * batch * nheads * seqlen_q * seqlen_k_eff * (hdim + 2 * hdimv)
    if is_causal and not gather_kv:
        FLOPs /= 2

    TFLOPS = FLOPs / exec_time_in_s / 1e12

    q_bytes = 2 * batch * nheads * seqlen_q * hdim
    qv_bytes = 2 * batch * nheads * seqlen_q * hdimv
    k_bytes = 2 * batch * nheads_kv * seqlen_k_eff * hdim
    v_bytes = 2 * batch * nheads_kv * seqlen_k_eff * hdimv
    o_bytes = 2 * batch * nheads * seqlen_q * hdimv
    total_bytes = q_bytes + qv_bytes + k_bytes + v_bytes + o_bytes
    TBs = total_bytes / exec_time_in_s / 1e12

    print(
        f"batch: {batch}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, nheads: {nheads}, -> {exec_time_in_s * 1e3:.2f} ms, {TFLOPS:.2f} TFLOPS, {TBs:.2f} TBs"
    )


if __name__ == "__main__":
    run_test = True
    run_benchmark = True
    gather_kv = True
    is_causal = False
    pack_gqa = True
    topk_length = 2048
    varlen_q = False
    varlen_k = False
    disable_bitmask = False
    validate = True
    has_qk = True

    if run_test:
        if not gather_kv:
            seqlen_q_test_values = range(1, 4002, 400)
            seqlen_k_test_values = range(1, 4002, 400)
        else:
            seqlen_q_test_values = range(1, 1001, 200)
            seqlen_k_test_values = range(topk_length, 9001, 2000)
        seqlen_q_test_values = [4096]
        seqlen_k_test_values = [4096]
        nheads_test_values = [128]
        batch_test_values = [1]
        test_configs = [
            (
                batch,
                nheads,
                seqlen_q,
                seqlen_k,
            )
            for batch in batch_test_values
            for nheads in nheads_test_values
            for seqlen_q in seqlen_q_test_values
            for seqlen_k in seqlen_k_test_values
        ]
        iters_per_config = 1
        compile_cache = dict()
        print("=" * 40)
        print("Testing MLA Kernel")
        print("=" * 40)
        for config in test_configs:
            batch, nheads, seqlen_q, seqlen_k = config
            # if is_causal and seqlen_k < seqlen_q:
            #     continue
            for iter in range(iters_per_config):
                test_mla_kernel(
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    topk_length=topk_length,
                    nheads=nheads,
                    batch=batch,
                    iter=iter,
                    compile_cache=compile_cache,
                    validate=validate,
                    seed=0,
                    gather_kv=gather_kv,
                    pack_gqa=pack_gqa,
                    is_causal=is_causal,
                    varlen_q=varlen_q,
                    varlen_k=varlen_k,
                    disable_bitmask=disable_bitmask,
                    has_qk=has_qk,
                )
    if run_benchmark:
        if gather_kv:
            seqlen_q_benchmark_values = [1]
            seqlen_k_benchmark_values = [8192 * 2]
            nheads_benchmark_values = [128]
            batch_benchmark_values = [128]
        else:
            seqlen_q_benchmark_values = [1]
            seqlen_k_benchmark_values = [8192]
            nheads_benchmark_values = [128]
            batch_benchmark_values = [128]
        # seqlen_q_benchmark_values = [4096]
        # seqlen_k_benchmark_values = [4096]
        # nheads_benchmark_values = [16]
        # batch_benchmark_values = [8]
        benchmark_configs = [
            (
                batch,
                nheads,
                seqlen_q,
                seqlen_k,
            )
            for batch in batch_benchmark_values
            for nheads in nheads_benchmark_values
            for seqlen_q in seqlen_q_benchmark_values
            for seqlen_k in seqlen_k_benchmark_values
        ]
        compile_cache = dict()
        print("=" * 40)
        print("Benchmarking MLA Kernel")
        print("=" * 40)
        for config in benchmark_configs:
            batch, nheads, seqlen_q, seqlen_k = config
            benchmark_mla_kernel(
                batch=batch,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                topk_length=topk_length,
                nheads=nheads,
                gather_kv=gather_kv,
                is_causal=is_causal,
                disable_bitmask=disable_bitmask,
                compile_cache=compile_cache,
            )
