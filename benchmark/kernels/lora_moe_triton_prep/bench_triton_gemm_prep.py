"""Self-contained perf-bench + correctness-test for the triton-gemm PREP group
on the EP8 bs64 Kimi-K2.5-NVFP4 LoRA decode path. The 3 prep kernels run
back-to-back and are benched as ONE combined pipeline (not individually), so a
future fused replacement can be compared apples-to-apples:

    topk_ids --_fused_virtual_topk_ids_kernel--> virtual_topk_ids        (~1.5us)
             --moe_align_block_size (sgl_kernel)--> sorted/expert/post_pad
               (this one launches BOTH moe_align_block_size_kernel ~2.7us
                and count_and_sort_expert_tokens_kernel ~4.7us)

Production decode shapes (SHAPE_REPORT.md, decode bs64, per-rank EP8):
    bs=64, top_k=8, num_experts=384, max_loras=1, local_num_experts=48 (384/8),
    block_size=16 -> virtual_num_experts=384.

These kernels read only a few KB/call (topk_ids is 64x8 i32), i.e. they are
latency/launch bound and their working set is far below L2 -> L2 state does not
affect the number (common_bench will print WARN: footprint<L2). Timing + buffer
rotation come from common_bench (CUDA-graph via triton do_bench_cudagraph).

Usage (on the GPU pod):
    python3 bench_triton_gemm_prep.py --mode bench
    python3 bench_triton_gemm_prep.py --mode correctness
"""

from __future__ import annotations

import argparse

import torch
from common_bench import bench_kernel, pick_n_sets, report_sets, set_bytes

from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
    moe_align_block_size,
)
from sglang.srt.lora.triton_ops.virtual_experts import _fused_virtual_topk_ids


def make_input_set(bs, top_k, num_experts, device):
    """Routing inputs for one rotation slot. token_lora_mapping=0 (single adapter)."""
    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    token_lora_mapping = torch.zeros(bs, device=device, dtype=torch.int32)
    return {"topk_ids": topk_ids, "tlm": token_lora_mapping}


def prep_pipeline(s, num_experts, local_num_experts, block_size):
    """The full triton-gemm prep: virtual topk ids -> native align (+count_and_sort)."""
    vtopk, _, vne = _fused_virtual_topk_ids(
        s["topk_ids"],
        s["tlm"],
        num_experts,
        shared_outer=False,
        max_loras=1,
        local_expert_offset=0,
        local_num_experts=local_num_experts,
    )
    return moe_align_block_size(vtopk, block_size, vne)


def prep_pipeline_new(
    s, num_experts, local_num_experts, block_size, compact=False, fuse_scatter=False
):
    """Fused replacement (inline virtual + EP skip; compact histograms over local
    experts; fuse_scatter does the whole thing in one threadblock/launch).
    Returns the same 3 outputs as prep_pipeline for apples-to-apples."""
    from sglang.jit_kernel.moe_lora_merged_align import moe_lora_merged_align

    sorted_ids, expert_ids, post_pad, _mask, _vne = moe_lora_merged_align(
        s["topk_ids"],
        s["tlm"],
        num_experts,
        shared_outer=False,
        max_loras=1,
        block_size=block_size,
        local_expert_offset=0,
        local_num_experts=local_num_experts,
        do_skip=True,
        compact=compact,
        fuse_scatter=fuse_scatter,
    )
    return sorted_ids, expert_ids, post_pad


def ref_virtual_topk(topk_ids, tlm, num_experts, local_num_experts, local_offset=0):
    bs, top_k = topk_ids.shape
    out = torch.empty_like(topk_ids)
    for mrow in range(bs):
        lora = int(tlm[mrow].item())
        safe = max(lora, 0)
        for k in range(top_k):
            base = int(topk_ids[mrow, k].item())
            owned = local_offset <= base < local_offset + local_num_experts
            base = base if owned else -1
            res = base if base < 0 else base + safe * num_experts
            out[mrow, k] = res if lora >= 0 else -1
    return out


def ref_align(vtopk, block, vne):
    """torch reference for sgl_kernel::moe_align_block_size (native wrapper passes
    num_experts+1 and the kernel uses the +1 offset: id -> id+1, so -1 maps to the
    sentinel bucket 0 and expert e to bucket e+1; each bucket is padded up to a
    block multiple; expert_ids labels each block bucket-1). Returns
    (num_tokens_post_padded, expert_ids_multiset)."""
    from collections import Counter

    buckets = (vtopk.reshape(-1) + 1).clamp(min=0)  # -1 -> 0 (sentinel), e -> e+1
    counts = torch.bincount(buckets, minlength=vne + 1)
    blocks = (counts + block - 1) // block
    post = int((blocks * block).sum().item())
    eids = Counter()
    for b in range(vne + 1):
        nb = int(blocks[b].item())
        if nb:
            eids[b - 1] += nb  # bucket 0 -> -1 sentinel label
    return post, eids


def ref_align_skip(vtopk, block, vne):
    """torch reference for the EP-skip variant: dropped/masked slots (vid < 0 ->
    sentinel bucket 0) are NOT placed, so bucket 0 contributes no tokens/blocks
    (post_pad shrinks, no -1 sentinel block)."""
    from collections import Counter

    buckets = (vtopk.reshape(-1) + 1).clamp(min=0)
    counts = torch.bincount(buckets, minlength=vne + 1)
    counts[0] = 0  # EP skip: sentinel bucket gets nothing
    blocks = (counts + block - 1) // block
    post = int((blocks * block).sum().item())
    eids = Counter()
    for b in range(vne + 1):
        nb = int(blocks[b].item())
        if nb:
            eids[b - 1] += nb
    return post, eids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "correctness"], default="bench")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--num-experts", type=int, default=384)
    ap.add_argument("--local-num-experts", type=int, default=48)
    ap.add_argument("--local-expert-offset", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--budget-gb", type=float, default=16.0)
    ap.add_argument("--n-sets", type=int, default=0, help="0 = auto (fill --budget-gb)")
    args = ap.parse_args()
    dev = "cuda"
    ne, lne, blk = args.num_experts, args.local_num_experts, args.block_size
    loff = args.local_expert_offset
    mk = lambda: make_input_set(args.bs, args.top_k, ne, dev)

    if args.mode == "correctness":
        s = mk()
        vtopk, _, vne = _fused_virtual_topk_ids(
            s["topk_ids"],
            s["tlm"],
            ne,
            shared_outer=False,
            max_loras=1,
            local_expert_offset=loff,
            local_num_experts=lne,
        )
        ref = ref_virtual_topk(s["topk_ids"], s["tlm"], ne, lne, loff)
        verr = int((vtopk != ref).sum().item())  # bitwise exact (integer ids)
        print(
            f"{'PASS' if verr == 0 else 'FAIL'} virtual_topk_ids bitwise mismatches={verr}"
        )
        from collections import Counter

        sorted_ids, expert_ids, post_pad = moe_align_block_size(vtopk, blk, vne)
        post_ref, eids_ref = ref_align(vtopk, blk, vne)
        pp = int(post_pad.item())
        nblk = pp // blk
        eids_k = Counter(e for e in expert_ids[:nblk].tolist())
        post_ok = pp == post_ref
        eids_ok = eids_k == eids_ref
        print(
            f"{'PASS' if (post_ok and eids_ok) else 'FAIL'} align vs ref: "
            f"post_pad={pp} (ref {post_ref}) expert_ids_multiset_match={eids_ok}"
        )

        # count_and_sort placement: verify sorted_token_ids actually scatters every token to
        # the right expert. Per block j (label expert_ids[j]=bucket-1), each non-sentinel slot
        # must belong to bucket eid+1; and every slot 0..numel-1 must appear exactly once.
        numel = vtopk.numel()
        buckets = (vtopk.reshape(-1) + 1).clamp(min=0).tolist()  # bucket per slot
        st = sorted_ids[:pp].tolist()
        eids_list = expert_ids[:nblk].tolist()
        seen = []
        place_ok = True
        for j in range(nblk):
            eid = eids_list[j]
            for slot in st[j * blk : (j + 1) * blk]:
                if slot == numel:  # padding sentinel
                    continue
                if not (0 <= slot < numel) or buckets[slot] != eid + 1:
                    place_ok = False
                    break
                seen.append(slot)
            if not place_ok:
                break
        all_once = place_ok and (sorted(seen) == list(range(numel)))
        print(
            f"{'PASS' if all_once else 'FAIL'} count_and_sort placement: "
            f"every token in correct expert block + each of {numel} slots exactly once={all_once}"
        )

        # === NEW fused kernel: two modes ===
        # mode A (do_skip=False): bucket-for-bucket identical to the old path
        #   -> validated against the same ref (bitwise-equivalence guardrail).
        # mode B (do_skip=True): EP skip -> dropped/masked slots are not placed,
        #   sentinel bucket emptied -> validated against ref_align_skip + the
        #   placement set must equal exactly the OWNED slots.
        from sglang.jit_kernel.moe_lora_merged_align import moe_lora_merged_align

        ref_mask = (s["tlm"] >= 0).to(torch.bool)

        def _check_new(
            do_skip, ref_post, ref_eids, owned_only, compact=False, fuse_scatter=False
        ):
            ns, ne_, np_, nm_, _ = moe_lora_merged_align(
                s["topk_ids"],
                s["tlm"],
                ne,
                shared_outer=False,
                max_loras=1,
                block_size=blk,
                local_expert_offset=loff,
                local_num_experts=lne,
                do_skip=do_skip,
                compact=compact,
                fuse_scatter=fuse_scatter,
            )
            pp = int(np_.item())
            nblk = pp // blk
            eids_k = Counter(e for e in ne_[:nblk].tolist())
            post_ok = pp == ref_post
            eids_ok = eids_k == ref_eids
            st_ = ns[:pp].tolist()
            el_ = ne_[:nblk].tolist()
            seen_, ok_ = [], True
            for j in range(nblk):
                eid = el_[j]
                for slot in st_[j * blk : (j + 1) * blk]:
                    if slot == numel:
                        continue
                    if not (0 <= slot < numel) or buckets[slot] != eid + 1:
                        ok_ = False
                        break
                    seen_.append(slot)
                if not ok_:
                    break
            if owned_only:
                expected = sorted(i_ for i_ in range(numel) if buckets[i_] >= 1)
            else:
                expected = list(range(numel))
            place_ok = ok_ and (sorted(seen_) == expected)
            mask_ok = bool((nm_ == ref_mask).all().item())
            return post_ok, eids_ok, place_ok, mask_ok, pp

        a_post_ok, a_eids_ok, a_place_ok, a_mask_ok, a_pp = _check_new(
            False, post_ref, eids_ref, owned_only=False
        )
        a_ok = a_post_ok and a_eids_ok and a_place_ok and a_mask_ok
        print(
            f"{'PASS' if a_ok else 'FAIL'} NEW(no-skip) vs ref: post_pad={a_pp} "
            f"(ref {post_ref}) eids={a_eids_ok} placement={a_place_ok} mask={a_mask_ok}"
        )

        post_skip, eids_skip = ref_align_skip(vtopk, blk, vne)
        b_post_ok, b_eids_ok, b_place_ok, b_mask_ok, b_pp = _check_new(
            True, post_skip, eids_skip, owned_only=True
        )
        b_ok = b_post_ok and b_eids_ok and b_place_ok and b_mask_ok
        print(
            f"{'PASS' if b_ok else 'FAIL'} NEW(skip) vs ref_skip: post_pad={b_pp} "
            f"(ref {post_skip}) eids={b_eids_ok} placement(owned-only)={b_place_ok} "
            f"mask={b_mask_ok}"
        )

        # mode C (do_skip=True, compact=True): histogram over LOCAL experts, but
        # expert_ids are restored to GLOBAL -> same observable result as mode B
        # (same ref_align_skip post_pad / expert-id multiset / owned placement).
        c_post_ok, c_eids_ok, c_place_ok, c_mask_ok, c_pp = _check_new(
            True, post_skip, eids_skip, owned_only=True, compact=True
        )
        c_ok = c_post_ok and c_eids_ok and c_place_ok and c_mask_ok
        print(
            f"{'PASS' if c_ok else 'FAIL'} NEW(skip+compact) vs ref_skip: post_pad={c_pp} "
            f"(ref {post_skip}) eids={c_eids_ok} placement(owned-only)={c_place_ok} "
            f"mask={c_mask_ok}"
        )

        # mode D (skip + compact + fuse_scatter): single-block fused kernel (fill +
        # histogram + scan + expert_ids + scatter in one launch) -> same observable
        # result as mode C.
        d_post_ok, d_eids_ok, d_place_ok, d_mask_ok, d_pp = _check_new(
            True, post_skip, eids_skip, owned_only=True, compact=True, fuse_scatter=True
        )
        d_ok = d_post_ok and d_eids_ok and d_place_ok and d_mask_ok
        print(
            f"{'PASS' if d_ok else 'FAIL'} NEW(skip+compact+FUSE 1-kernel) vs ref_skip: "
            f"post_pad={d_pp} (ref {post_skip}) eids={d_eids_ok} "
            f"placement(owned-only)={d_place_ok} mask={d_mask_ok}"
        )
        new_ok = a_ok and b_ok and c_ok and d_ok

        raise SystemExit(
            0 if (verr == 0 and post_ok and eids_ok and all_once and new_ok) else 1
        )

    per = set_bytes(mk())
    n_sets = pick_n_sets(per, args.budget_gb, args.n_sets)
    S = [mk() for _ in range(n_sets)]
    call = lambda i: prep_pipeline(S[i], ne, lne, blk)
    us = bench_kernel(call, n_sets) * 1000
    call_new = lambda i: prep_pipeline_new(S[i], ne, lne, blk, compact=False)
    us_new = bench_kernel(call_new, n_sets) * 1000
    call_cmp = lambda i: prep_pipeline_new(S[i], ne, lne, blk, compact=True)
    us_cmp = bench_kernel(call_cmp, n_sets) * 1000
    call_fuse = lambda i: prep_pipeline_new(
        S[i], ne, lne, blk, compact=True, fuse_scatter=True
    )
    us_fuse = bench_kernel(call_fuse, n_sets) * 1000
    print(
        f"BENCH triton-gemm prep (COMBINED: virtual_topk_ids + moe_align + count_and_sort) "
        f"bs={args.bs} top_k={args.top_k} experts={ne} local_experts={lne} block={blk}"
    )
    print(f"  per_set={per/1e3:.1f}KB {report_sets(per, n_sets)}")
    print(f"  OLD combined prep pipeline       = {us:7.2f} us")
    print(f"  NEW 2-kernel (skip)              = {us_new:7.2f} us   ({us/us_new:.2f}x)")
    print(f"  NEW 2-kernel (skip + compact)    = {us_cmp:7.2f} us   ({us/us_cmp:.2f}x)")
    print(
        f"  NEW 1-kernel (skip+compact+FUSE) = {us_fuse:7.2f} us   ({us/us_fuse:.2f}x)"
    )


if __name__ == "__main__":
    main()
