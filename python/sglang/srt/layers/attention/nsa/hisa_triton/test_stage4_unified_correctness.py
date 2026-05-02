"""Verify the unified persistent dispatch in block_sparse_mqa_triton
against the tilelang reference at all K∈{16,32,64,128}.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    _block_sparse_mqa_grouped_kernel,
    block_sparse_mqa_triton,
)


def grouped_reference(K, q_fp8, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke):
    """Triton grouped (non-persistent) kernel — used as K=8 reference since
    tilelang asserts M%16==0 and rejects K=8."""
    import torch as _t
    seq_len, H, D = q_fp8.shape
    seq_kv = k_fp8.shape[0]
    topk = int(topk_idx.shape[-1])
    GEMM_TILE = 128 if K == 8 else 256
    GROUP_SIZE = GEMM_TILE // K
    num_chunks = (topk + GROUP_SIZE - 1) // GROUP_SIZE
    logits = _t.empty((seq_len, topk * K), device=q_fp8.device, dtype=_t.float32)
    grid = (seq_len, num_chunks)
    _block_sparse_mqa_grouped_kernel[grid](
        q_fp8, k_fp8, k_scale, topk_idx, logits, w,
        cu_ks, cu_ke,
        q_fp8.stride(0), q_fp8.stride(1), q_fp8.stride(2),
        k_fp8.stride(0), k_fp8.stride(1),
        k_scale.stride(0),
        topk_idx.stride(0), topk_idx.stride(1),
        logits.stride(0), logits.stride(1),
        w.stride(0), w.stride(1),
        seq_kv,
        topk,
        HEADS=H, DIM=D,
        KV_BLOCK_SIZE=K,
        GROUP_SIZE=GROUP_SIZE,
    )
    return logits
from sglang.srt.layers.attention.nsa.hisa_triton.test_e2e_simulated import (
    BLOCK_TOPK_FORMULA, PREFILL_CHUNK, make_prefill_inputs,
)


def random_topk(K, sq, cu_ke, block_topk, seed=2):
    torch.manual_seed(seed)
    ke_blocks = ((cu_ke + K - 1) // K).long()
    n_blocks_max = int(ke_blocks.max().item())
    idx = torch.randint(
        0, n_blocks_max, (sq, block_topk),
        device=cu_ke.device, dtype=torch.int64,
    )
    idx = torch.minimum(idx, (ke_blocks - 1).clamp_min(0).unsqueeze(1))
    return idx


def chunked_max_abs_rel(ref, out, chunk=256):
    max_abs = 0.0
    scale = 0.0
    inf_match_all = True
    fin_match_all = True
    for i in range(0, ref.shape[0], chunk):
        r = ref[i:i+chunk]; o = out[i:i+chunk]
        inf_match_all &= bool((torch.isinf(r) == torch.isinf(o)).all().item())
        fin_match_all &= bool((torch.isfinite(r) == torch.isfinite(o)).all().item())
        both = torch.isfinite(r) & torch.isfinite(o)
        r2 = torch.where(both, r, 0.0)
        o2 = torch.where(both, o, 0.0)
        d = (r2 - o2).abs_()
        ma = d.max().item(); sc = r2.abs_().max().item()
        if ma > max_abs: max_abs = ma
        if sc > scale: scale = sc
        del r2, o2, d, both
    return inf_match_all, fin_match_all, max_abs, max_abs / max(scale, 1e-6)


def main():
    sq = PREFILL_CHUNK
    print("=" * 100)
    print(f"block_sparse_mqa_triton unified persistent vs tilelang correctness "
          f"(sq={sq})")
    print("=" * 100)
    print(f"{'K':>4} {'skv':>5} {'topk':>5} | {'inf':>5} {'fin':>5} "
          f"{'max_abs':>10} {'rel':>10}  status")
    print("-" * 100)

    fails = 0
    for K in (8, 16, 32, 64, 128):
        block_topk = BLOCK_TOPK_FORMULA // K
        for skv_k in (16, 32, 65, 128):
            skv = skv_k * 1024
            q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
            topk_idx = random_topk(K, sq, cu_ke, block_topk)

            if K == 8:
                # tilelang asserts M % 16 == 0; use the in-house grouped
                # triton kernel as reference (same fp8 math, different tile).
                ref = grouped_reference(K, q, k_fp8, k_scale, topk_idx, w, cu_ks, cu_ke)
                ref_name = "grouped triton"
            else:
                ref = fp8_native_block_sparse_mqa_attn_return_logits_interface(
                    q=q, k=k_fp8, k_scale=k_scale,
                    topk_block_index=topk_idx,
                    kv_block_size=K, weights=w,
                    cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
                )
                ref_name = "tilelang"
            out = block_sparse_mqa_triton(
                q_fp8=q, k_fp8=k_fp8, k_scale=k_scale,
                topk_block_index=topk_idx, kv_block_size=K,
                weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
            )

            inf_m, fin_m, max_abs, rel = chunked_max_abs_rel(ref, out)
            ok = inf_m and fin_m and rel < 5e-2
            if not ok:
                fails += 1
            status = "OK" if ok else "FAIL"
            print(f"{K:>4} {skv_k:>3}K {block_topk:>5} | "
                  f"{str(inf_m):>5} {str(fin_m):>5} "
                  f"{max_abs:>10.4f} {rel:>10.2e}  [{status}]")
            del q, k_fp8, k_scale, w, cu_ks, cu_ke, topk_idx, ref, out
            torch.cuda.empty_cache()

    print()
    print(f"TOTAL FAILS: {fails}")
    if fails == 0:
        print("ALL_OK")


if __name__ == "__main__":
    main()
