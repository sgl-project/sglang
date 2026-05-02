"""Correctness check: prefill stage 4 K=128 triton persistent kernel vs
tilelang reference. Same input distribution as production prefill.

Both paths consume identical inputs and must produce logits matching
within fp8 ULP noise. Stage 4's only nondeterminism is fp8 accumulation
ordering — typical max_abs ~1e-3, rel ~1e-3 across 8K rows.
"""
from __future__ import annotations

import torch

from sglang.srt.layers.attention.nsa.hisa.custom_ops import (
    fp8_native_block_sparse_mqa_attn_return_logits_interface,
)
from sglang.srt.layers.attention.nsa.hisa_triton.kernels import (
    block_sparse_mqa_triton,
)
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


def main():
    K = 128
    block_topk = BLOCK_TOPK_FORMULA // K
    sq = PREFILL_CHUNK
    print("=" * 100)
    print(f"K={K} prefill stage 4 — triton persistent vs tilelang correctness")
    print(f"sq={sq}  block_topk={block_topk}")
    print("=" * 100)
    print(f"{'skv':>5} | {'inf_match':>10} {'fin_match':>10} "
          f"{'max_abs':>10} {'max_rel':>10}  status")
    print("-" * 100)

    fails = 0
    for skv_k in (16, 32, 65, 128):
        skv = skv_k * 1024
        q, (k_fp8, k_scale), w, cu_ks, cu_ke = make_prefill_inputs(sq, skv)
        topk_idx = random_topk(K, sq, cu_ke, block_topk)

        ref = fp8_native_block_sparse_mqa_attn_return_logits_interface(
            q=q, k=k_fp8, k_scale=k_scale,
            topk_block_index=topk_idx,
            kv_block_size=K, weights=w,
            cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )
        out = block_sparse_mqa_triton(
            q_fp8=q, k_fp8=k_fp8, k_scale=k_scale,
            topk_block_index=topk_idx, kv_block_size=K,
            weights=w, cu_seqlen_ks=cu_ks, cu_seqlen_ke=cu_ke,
        )

        assert ref.shape == out.shape, f"shape mismatch: ref {ref.shape} vs out {out.shape}"

        # Memory-tight per-row stats: ref and out are each ~256MB at
        # K=128 sq=8K block_topk=64; avoid materialising any extra tensor.
        max_abs = 0.0
        scale = 0.0
        inf_match_all = True
        fin_match_all = True
        chunk = 256
        for i in range(0, ref.shape[0], chunk):
            r = ref[i:i+chunk]
            o = out[i:i+chunk]
            r_inf = torch.isinf(r); o_inf = torch.isinf(o)
            inf_match_all &= bool((r_inf == o_inf).all().item())
            r_fin = torch.isfinite(r); o_fin = torch.isfinite(o)
            fin_match_all &= bool((r_fin == o_fin).all().item())
            both = r_fin & o_fin
            # In-place: replace non-finite with 0, then sub.
            r2 = torch.where(both, r, 0.0)
            o2 = torch.where(both, o, 0.0)
            d = (r2 - o2).abs_()
            ma = d.max().item()
            sc = r2.abs_().max().item()
            if ma > max_abs:
                max_abs = ma
            if sc > scale:
                scale = sc
            del r2, o2, d, both, r_fin, o_fin, r_inf, o_inf
        inf_match = inf_match_all
        fin_match = fin_match_all
        max_rel = max_abs / max(scale, 1e-6)

        ok = inf_match and fin_match and max_rel < 5e-2
        status = "OK" if ok else "FAIL"
        if not ok:
            fails += 1
        print(f"{skv_k:>3}K | {str(inf_match):>10} {str(fin_match):>10} "
              f"{max_abs:>10.4f} {max_rel:>10.2e}  [{status}]")

        del q, k_fp8, k_scale, w, cu_ks, cu_ke, topk_idx, ref, out
        torch.cuda.empty_cache()

    print()
    if fails == 0:
        print("ALL_OK")
    else:
        print(f"FAILED: {fails} shapes")


if __name__ == "__main__":
    main()
