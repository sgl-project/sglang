"""Phase 3+4: Benchmark deepep_permute_triton_kernel variants with correctness + PTX ablation.

Sweep: BLOCK_SIZE in {256, 512, 1024}, num_warps in {1, 2, 4, 8}, num_stages in {1, 2}
Benchmark sizes: num_tokens in {64, 256, 1024, 4096}, hidden_size in {2048, 4096, 7168}, topk in {2, 6, 8}
"""
import json
import os
import re
import shutil
from pathlib import Path

import torch
import triton
import triton.language as tl

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache_gpu_3"

OUT_DIR = Path(__file__).parent

# ============================================================
# Kernel variants
# ============================================================

@triton.jit
def kernel_bs256(
    input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
    topk, hidden_size, BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


@triton.jit
def kernel_bs512(
    input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
    topk, hidden_size, BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


@triton.jit
def kernel_bs1024(
    input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
    topk, hidden_size, BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


@triton.jit
def kernel_bs256_stages2(
    input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
    topk, hidden_size, BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE, num_stages=2):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


@triton.jit
def kernel_bs512_stages2(
    input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
    topk, hidden_size, BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE, num_stages=2):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


@triton.jit
def kernel_bs1024_stages2(
    input_ptr, gateup_input_ptr, src2dst_ptr, topk_ids_ptr, a1_scales_ptr,
    topk, hidden_size, BLOCK_SIZE: tl.constexpr,
):
    OutDtype = gateup_input_ptr.dtype.element_ty
    src_idx = tl.program_id(0)
    src2dst_ptr = src2dst_ptr + src_idx * topk
    topk_ids_ptr = topk_ids_ptr + src_idx * topk
    src_ptr = input_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE, num_stages=2):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size
        in_data = tl.load(src_ptr + offset, mask=mask).to(OutDtype)
        for idx in range(topk):
            dst_idx = tl.load(src2dst_ptr + idx)
            if dst_idx >= 0:
                dst_ptr = gateup_input_ptr + dst_idx * hidden_size
                tl.store(dst_ptr + offset, in_data, mask=mask)


# ============================================================
# Variant registry
# ============================================================

# (name, kernel_fn, BLOCK_SIZE, num_warps, num_stages_label)
VARIANTS = []
for bs, kern, kern_s2 in [
    (256, kernel_bs256, kernel_bs256_stages2),
    (512, kernel_bs512, kernel_bs512_stages2),
    (1024, kernel_bs1024, kernel_bs1024_stages2),
]:
    for nw in [1, 2, 4, 8]:
        VARIANTS.append((f"BS{bs}_W{nw}_S1", kern, bs, nw, 1))
        VARIANTS.append((f"BS{bs}_W{nw}_S2", kern_s2, bs, nw, 2))

BASELINE_NAME = "BS512_W4_S1"  # Current production config


# ============================================================
# PyTorch reference
# ============================================================

def pytorch_reference(input_tensor, src2dst, topk_ids, topk, hidden_size, out_dtype):
    """Reference: for each token, for each topk expert with dst_idx >= 0:
      gateup_input[dst_idx] = input[src_idx].to(output_dtype)
    """
    num_tokens = input_tensor.shape[0]
    max_dst = int(src2dst.max().item()) + 1 if (src2dst >= 0).any() else 1
    out = torch.zeros(max_dst, hidden_size, device=input_tensor.device, dtype=out_dtype)
    for t in range(num_tokens):
        for k in range(topk):
            dst_idx = src2dst[t, k].item()
            if dst_idx >= 0:
                out[dst_idx] = input_tensor[t].to(out_dtype)
    return out


# ============================================================
# Setup inputs
# ============================================================

def make_inputs(num_tokens, hidden_size, topk, device="cuda:0"):
    input_tensor = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    # Create realistic src2dst: sequential mapping with some -1s
    src2dst = torch.arange(num_tokens * topk, device=device, dtype=torch.int64).reshape(num_tokens, topk)
    # Mark ~10% as invalid (dst_idx = -1)
    invalid_mask = torch.rand(num_tokens, topk, device=device) < 0.1
    src2dst[invalid_mask] = -1
    max_dst = int(src2dst.max().item()) + 1
    gateup_input = torch.empty(max_dst, hidden_size, device=device, dtype=torch.bfloat16)
    topk_ids = torch.randint(0, 64, (num_tokens, topk), device=device, dtype=torch.int64)
    return input_tensor, gateup_input, src2dst, topk_ids, max_dst


# ============================================================
# Correctness check
# ============================================================

def check_correctness(kernel_fn, block_size, num_warps, num_tokens, hidden_size, topk, device="cuda:0"):
    input_tensor, gateup_input, src2dst, topk_ids, max_dst = make_inputs(num_tokens, hidden_size, topk, device)

    # Reference
    ref = pytorch_reference(input_tensor, src2dst, topk_ids, topk, hidden_size, torch.bfloat16)

    # Triton
    gateup_input.zero_()
    grid = (num_tokens,)
    kernel_fn[grid](
        input_tensor, gateup_input, src2dst, topk_ids, None,
        topk, hidden_size, BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()

    # Compare only valid entries
    valid_dst_indices = src2dst[src2dst >= 0].long()
    if len(valid_dst_indices) == 0:
        return True, 1.0, 0.0

    triton_out = gateup_input[valid_dst_indices]
    ref_out = ref[valid_dst_indices]

    exact = (triton_out.view(torch.uint8) == ref_out.view(torch.uint8)).float().mean().item()
    max_diff = (triton_out.float() - ref_out.float()).abs().max().item()
    return exact > 0.99, exact, max_diff


# ============================================================
# Benchmark
# ============================================================

def bench_variant(kernel_fn, block_size, num_warps, num_tokens, hidden_size, topk, device="cuda:0"):
    input_tensor, gateup_input, src2dst, topk_ids, max_dst = make_inputs(num_tokens, hidden_size, topk, device)
    grid = (num_tokens,)

    def run():
        kernel_fn[grid](
            input_tensor, gateup_input, src2dst, topk_ids, None,
            topk, hidden_size, BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )

    # warmup
    for _ in range(10):
        run()
    torch.cuda.synchronize()

    ms = triton.testing.do_bench(run, warmup=50, rep=200, return_mode="median")
    return ms


# ============================================================
# PTX extraction
# ============================================================

def extract_ptx_for_variant(kernel_fn, block_size, num_warps, num_tokens, hidden_size, topk, device="cuda:0"):
    """Compile variant and extract PTX."""
    cache_dir = Path("/tmp/triton_cache_gpu_3")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    input_tensor, gateup_input, src2dst, topk_ids, _ = make_inputs(num_tokens, hidden_size, topk, device)
    grid = (num_tokens,)
    kernel_fn[grid](
        input_tensor, gateup_input, src2dst, topk_ids, None,
        topk, hidden_size, BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    torch.cuda.synchronize()

    ptx_files = sorted(cache_dir.rglob("*.ptx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if ptx_files:
        return ptx_files[0].read_text()
    return ""


def count_ptx_instructions(ptx_text):
    cats = {
        "ld.global": r"ld\.global",
        "st.global": r"st\.global",
        "ld.shared": r"ld\.shared",
        "st.shared": r"st\.shared",
        "cp.async": r"cp\.async",
        "mul": r"\tmul\.",
        "add": r"\tadd\.",
        "fma": r"\tfma\.",
        "cvt": r"\tcvt\.",
        "setp": r"\tsetp\.",
        "mov": r"\tmov\.",
        "bar": r"bar\.",
        "bra": r"\tbra\b",
    }
    return {k: len(re.findall(v, ptx_text)) for k, v in cats.items()}


# ============================================================
# Main
# ============================================================

def main():
    device = "cuda:0"

    # ---- Correctness check ----
    print("=" * 70)
    print("CORRECTNESS CHECK")
    print("=" * 70)

    test_configs = [(128, 2048, 2), (256, 4096, 6)]
    all_correct = True
    for vname, kern, bs, nw, ns in VARIANTS:
        for nt, hs, tk in test_configs:
            ok, exact, maxd = check_correctness(kern, bs, nw, nt, hs, tk, device)
            if not ok:
                print(f"  FAIL {vname} @ nt={nt},hs={hs},tk={tk}: exact={exact:.4f}, max_diff={maxd}")
                all_correct = False
        print(f"  OK: {vname}")

    if not all_correct:
        print("\nSome variants FAILED correctness. Aborting.")
        return

    print("\nAll variants passed correctness.\n")

    # ---- Benchmark ----
    print("=" * 70)
    print("BENCHMARK")
    print("=" * 70)

    sizes = []
    for nt in [64, 256, 1024, 4096]:
        for hs in [2048, 4096, 7168]:
            for tk in [2, 6, 8]:
                sizes.append((nt, hs, tk))

    results = {}
    for vname, kern, bs, nw, ns in VARIANTS:
        results[vname] = {}
        for (nt, hs, tk) in sizes:
            ms = bench_variant(kern, bs, nw, nt, hs, tk, device)
            key = f"nt{nt}_hs{hs}_tk{tk}"
            results[vname][key] = ms
        # Print progress
        sample_key = "nt1024_hs4096_tk6"
        print(f"  {vname:25s}: {results[vname][sample_key]:.4f} ms (nt=1024,hs=4096,tk=6)")

    # ---- Find best ----
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    baseline_name = BASELINE_NAME
    baseline_results = results[baseline_name]

    # Geometric mean speedup across all sizes
    best_variant = None
    best_geomean = 0.0

    for vname in results:
        speedups = []
        for key in baseline_results:
            bl = baseline_results[key]
            vr = results[vname][key]
            if bl > 0 and vr > 0:
                speedups.append(bl / vr)
        if speedups:
            import math
            geomean = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
            if geomean > best_geomean:
                best_geomean = geomean
                best_variant = vname
            if vname == baseline_name:
                print(f"  {vname:25s}: geomean speedup = 1.0000x (baseline)")
            else:
                print(f"  {vname:25s}: geomean speedup = {geomean:.4f}x vs baseline")

    print(f"\nBest variant: {best_variant} with {best_geomean:.4f}x geomean speedup")

    # Per-size breakdown for top variants
    print("\n--- Per-size comparison: baseline vs best ---")
    for key in sorted(baseline_results.keys()):
        bl = baseline_results[key]
        bv = results[best_variant][key]
        speedup = bl / bv if bv > 0 else 0
        print(f"  {key:30s}: baseline={bl:.4f}ms, best={bv:.4f}ms, speedup={speedup:.3f}x")

    # ---- PTX Ablation ----
    print("\n" + "=" * 70)
    print("PTX ABLATION")
    print("=" * 70)

    # Extract baseline and best variant PTX
    bl_kern = [v for v in VARIANTS if v[0] == baseline_name][0]
    best_kern = [v for v in VARIANTS if v[0] == best_variant][0]

    ptx_baseline = extract_ptx_for_variant(bl_kern[1], bl_kern[2], bl_kern[3], 1024, 4096, 6, device)
    (OUT_DIR / "ptx_baseline.txt").write_text(ptx_baseline)

    ptx_best = extract_ptx_for_variant(best_kern[1], best_kern[2], best_kern[3], 1024, 4096, 6, device)
    (OUT_DIR / "ptx_best.txt").write_text(ptx_best)

    bl_counts = count_ptx_instructions(ptx_baseline)
    best_counts = count_ptx_instructions(ptx_best)

    print(f"\n{'Category':15s} {'Baseline':>10s} {'Best':>10s} {'Delta':>10s}")
    print("-" * 50)
    for cat in bl_counts:
        bl_c = bl_counts.get(cat, 0)
        bst_c = best_counts.get(cat, 0)
        delta = bst_c - bl_c
        print(f"  {cat:15s} {bl_c:10d} {bst_c:10d} {delta:+10d}")

    # ---- Save results ----
    output = {
        "baseline": baseline_name,
        "best_variant": best_variant,
        "best_geomean_speedup": best_geomean,
        "best_config": {
            "BLOCK_SIZE": best_kern[2],
            "num_warps": best_kern[3],
            "num_stages": best_kern[4],
        },
        "baseline_config": {
            "BLOCK_SIZE": bl_kern[2],
            "num_warps": bl_kern[3],
            "num_stages": bl_kern[4],
        },
        "results": results,
        "ptx_baseline_counts": bl_counts,
        "ptx_best_counts": best_counts,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {OUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
