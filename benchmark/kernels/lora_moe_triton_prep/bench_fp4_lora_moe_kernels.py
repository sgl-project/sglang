"""Self-contained perf-bench + correctness-test for the fp4-LoRA MoE compute
kernels that live inside FP4BlockScaleLoraLauncher::run (EP8 bs64 Kimi decode):

    moe::dev::permute::permuteKernel                       (~7us)
    tensorrt_llm::kernels::nvfp4QuantAndPerTokenScaleKernel (~14us, x2: gate_up + down)
    moe::dev::activation::activationKernel                  (~14-16us)

These have NO standalone python binding, so the overlay module exports three
single-kernel runners (bench_permute / bench_nvfp4_quant / bench_activation);
this script pre-allocates the in/out tensors (decode-bs64 shapes, hard-coded as the
defaults below) and calls them directly. Timing + cold-L2 buffer rotation: see common_bench.

Decode bs64 (per-rank EP8): num_tokens=64 top_k=8 hidden=7168 inter=2048
gate_up_n=4096 num_experts=384 local_experts=48 tile=8 max_num_padded_tokens=3200.

Usage (on the GPU pod):
  python3 bench_fp4_lora_moe_kernels.py --mode bench
  python3 bench_fp4_lora_moe_kernels.py --mode correctness
"""

from __future__ import annotations

import argparse

import torch
from common_bench import bench_kernel, pick_n_sets, report_sets, set_bytes

from sglang.jit_kernel.flashinfer_trtllm_moe.core import (
    get_sgl_trtllm_moe_sm100_raw_module,
)


def swizzled_sf_size(m, n, tile):
    """SF buffer size, mirroring the launcher's computeSwizzledLayoutSFSize(m, n/16) calls,
    which use the DEFAULT rowSize=128 regardless of tile -> round m to 128, n/16 to 4. (Always
    128-rounding never under-allocates vs the 8x4 read index used in dequant.)"""
    del tile  # launcher always rounds rows to 128 for the buffer size
    m_round = ((m + 127) // 128) * 128
    nsf = ((n // 16 + 3) // 4) * 4
    return m_round * nsf


def make_one_set(num_tokens, top_k, hidden, inter, gate_up_n, maxpad, tile, dev):
    e = num_tokens * top_k  # real permuted rows
    # idx_map: expanded slot e -> permuted row e (injective; rows [0,e) used, rest padding).
    idx_map = torch.arange(e, dtype=torch.int32, device=dev)
    total_pad = torch.tensor([maxpad], dtype=torch.int32, device=dev)
    return dict(
        hidden_in=torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=dev),
        # permuted / activated are randn (NOT zeros): in production the [e:maxpad] padding
        # rows are left UNINITIALIZED (non-zero garbage), and quant #1 processes all maxpad
        # rows. Zero-filling them would hit the quant's zero-row fast path and undercount the
        # quant timing. permute/activation overwrite the real [0:e) rows; padding stays randn.
        permuted=torch.randn(maxpad, hidden, dtype=torch.bfloat16, device=dev),
        idx_map=idx_map,
        total_pad=total_pad,
        q1_fp4=torch.empty(maxpad, hidden // 2, dtype=torch.uint8, device=dev),
        q1_sf=torch.empty(
            swizzled_sf_size(maxpad, hidden, tile), dtype=torch.uint8, device=dev
        ),
        q1_ptsf=torch.empty(maxpad, dtype=torch.float32, device=dev),
        gate_up=torch.randn(maxpad, gate_up_n, dtype=torch.bfloat16, device=dev),
        lora_delta=torch.randn(
            num_tokens, top_k, gate_up_n, dtype=torch.bfloat16, device=dev
        )
        * 0.1,
        activated=torch.randn(maxpad, inter, dtype=torch.bfloat16, device=dev),
        lora_input=torch.zeros(
            num_tokens, top_k, inter, dtype=torch.bfloat16, device=dev
        ),
        q2_fp4=torch.empty(maxpad, inter // 2, dtype=torch.uint8, device=dev),
        q2_sf=torch.empty(
            swizzled_sf_size(maxpad, inter, tile), dtype=torch.uint8, device=dev
        ),
        q2_ptsf=torch.empty(maxpad, dtype=torch.float32, device=dev),
    )


_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
_E2M1 = _E2M1 + [-v for v in _E2M1]  # codes 8..15 are the negatives


def dequant_rel_err(input_bf16, fp4_u8, sf_u8, ptsf, m, n, tile, e):
    """Dequantize the kernel's NVFP4 output and return ||deq-input||/||input|| over the
    e real rows. Per the quant kernel (quantization.cuh:nvfp4QuantAndPerTokenScaleKernel):
    x ~= e2m1(fp4) * e4m3(block_sf) * per_token_sf, where per_token_sf[row]=rowamax/(448*6),
    block_sf is the per-16-elt e4m3 scale stored in the SWIZZLED_8x4 layout
    (get_sf_out_offset_8x4), and fp4 is row-major (2 codes/byte, even col = low nibble).
    """
    assert tile < 128, "this dequant decodes SWIZZLED_8x4 (tile<128)"
    dev = input_bf16.device
    lut = torch.tensor(_E2M1, dtype=torch.float32, device=dev)
    lo = (fp4_u8 & 0xF).long()
    hi = (fp4_u8 >> 4).long()
    vals = lut[torch.stack([lo, hi], -1).reshape(m, n)]  # [m, n], even col=low nibble

    num_vecs = n // 16
    rows = torch.arange(m, device=dev)[:, None]
    vecs = torch.arange(num_vecs, device=dev)[None, :]
    num_k_tiles = (num_vecs + 3) // 4
    off = (
        (rows // 8) * (num_k_tiles * 32)
        + (vecs // 4) * 32
        + (rows % 8) * 4
        + (vecs % 4)
    )  # get_sf_out_offset_8x4
    block_sf = sf_u8.view(torch.float8_e4m3fn).float()[off]  # [m, num_vecs]
    block_sf = block_sf.repeat_interleave(16, dim=1)[:, :n]

    deq = vals * block_sf * ptsf[:, None]
    inp = input_bf16.float()
    num = (deq[:e] - inp[:e]).norm()
    den = inp[:e].norm() + 1e-9
    return float((num / den).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bench", "correctness"], default="bench")
    ap.add_argument("--num-tokens", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--inter", type=int, default=2048)
    ap.add_argument("--maxpad", type=int, default=3200)
    ap.add_argument("--tile", type=int, default=8)
    ap.add_argument("--budget-gb", type=float, default=16.0)
    ap.add_argument("--n-sets", type=int, default=0, help="0 = auto (fill --budget-gb)")
    args = ap.parse_args()
    dev = "cuda"
    nt, tk, H, I = args.num_tokens, args.top_k, args.hidden, args.inter
    gun, mp = 2 * I, args.maxpad
    m = get_sgl_trtllm_moe_sm100_raw_module()
    mk = lambda: make_one_set(nt, tk, H, I, gun, mp, args.tile, dev)

    def runners(S):
        return {
            "permuteKernel": lambda i: m.bench_permute(
                S[i]["hidden_in"],
                S[i]["idx_map"],
                S[i]["total_pad"],
                S[i]["permuted"],
                nt,
                tk,
                H,
            ),
            f"nvfp4 quant #1 (gate_up, m={mp})": lambda i: m.bench_nvfp4_quant(
                S[i]["permuted"],
                None,
                S[i]["q1_fp4"],
                S[i]["q1_sf"],
                S[i]["q1_ptsf"],
                mp,
                H,
                args.tile,
            ),
            "activationKernel": lambda i: m.bench_activation(
                S[i]["gate_up"],
                S[i]["lora_delta"],
                S[i]["idx_map"],
                S[i]["total_pad"],
                S[i]["activated"],
                S[i]["lora_input"],
                gun,
                nt,
                tk,
                0,  # grid_x_override=0 -> default grid.x = innerDim/128
                0,  # opt_mode=0 -> scalar activationKernel
            ),
            f"nvfp4 quant #2 (down, m={nt*tk})": lambda i: m.bench_nvfp4_quant(
                S[i]["activated"],
                S[i]["idx_map"],
                S[i]["q2_fp4"],
                S[i]["q2_sf"],
                S[i]["q2_ptsf"],
                nt * tk,
                I,
                args.tile,
            ),
            # fused permute+quant (replaces permuteKernel + quant#1): reads unpermuted hidden_in,
            # scatter-writes to the permuted q1 buffers. Compare its time to permute+quant#1 summed.
            "FUSED permute+quant (no-dedup)": lambda i: m.bench_fused_permute_quant(
                S[i]["hidden_in"],
                S[i]["idx_map"],
                S[i]["q1_fp4"],
                S[i]["q1_sf"],
                S[i]["q1_ptsf"],
                nt,
                tk,
                H,
                mp,
                args.tile,
                0,  # dedup=0
            ),
            "FUSED permute+quant (dedup)": lambda i: m.bench_fused_permute_quant(
                S[i]["hidden_in"],
                S[i]["idx_map"],
                S[i]["q1_fp4"],
                S[i]["q1_sf"],
                S[i]["q1_ptsf"],
                nt,
                tk,
                H,
                mp,
                args.tile,
                1,  # dedup=1
            ),
        }

    if args.mode == "correctness":
        import torch.nn.functional as F

        S = [mk()]
        s0 = S[0]
        r = list(runners(S).values())
        e = nt * tk  # idx_map = arange(e): permuted row s <- token s//tk, k s%tk

        # permute: bitwise-exact gather.
        r[0](0)
        torch.cuda.synchronize()
        ref_perm = s0["hidden_in"][torch.arange(e, device=dev) // tk]
        perr = int((s0["permuted"][:e] != ref_perm).sum().item())
        print(
            f"{'PASS' if perr == 0 else 'FAIL'} permute bitwise gather mismatches={perr}"
        )

        # activation: torch ref of activationKernel (dev_kernel.cu). gate_up is interleaved;
        # per the kernel: up = even col + lora_delta[h+inner]; gate = odd col + lora_delta[h];
        # out = silu(gate) * up. (lora_delta indexed contiguously, NOT interleaved.) bf16 -> num tol.
        r[2](0)
        torch.cuda.synchronize()
        gu = s0["gate_up"][:e].float()
        ld = s0["lora_delta"].reshape(e, gun).float()
        inner = gun // 2
        up = gu[:, 0::2] + ld[:, inner:]
        ga = gu[:, 1::2] + ld[:, :inner]
        ref_act = F.silu(ga) * up
        got_act = s0["activated"][:e].float()
        aerr = float((got_act - ref_act).abs().max().item())
        arel = aerr / float(ref_act.abs().max().item() + 1e-9)
        act_ok = arel <= 2e-2
        print(
            f"{'PASS' if act_ok else 'FAIL'} activation vs SwiGLU+lora ref: "
            f"max_abs_err={aerr:.4e} rel={arel:.2e} (tol 2e-2, bf16)"
        )

        # activation grid-invariance: a grid.x override must be bitwise-identical to the
        # default grid (the kernel's hidden-dim grid-stride loop writes each output once,
        # so output is independent of grid.x). This is the guard for the grid-tuning opt.
        act_ref = s0["activated"].clone()
        li_ref = s0["lora_input"].clone()
        m.bench_activation(
            s0["gate_up"], s0["lora_delta"], s0["idx_map"], s0["total_pad"],
            s0["activated"], s0["lora_input"], gun, nt, tk, 2, 0,  # grid_x_override=2, opt_mode=0
        )
        torch.cuda.synchronize()
        gx_eq = bool(
            torch.equal(s0["activated"], act_ref) and torch.equal(s0["lora_input"], li_ref)
        )
        print(f"{'PASS' if gx_eq else 'FAIL'} activation grid.x=2 bitwise == default grid.x")

        # activation opt (vectorized activationKernelOpt) must be BITWISE-identical to the scalar
        # kernel: same per-element float math (bf16->float silu), only loads/stores are vectorized.
        # act_ref/li_ref are the scalar golden above.
        m.bench_activation(
            s0["gate_up"], s0["lora_delta"], s0["idx_map"], s0["total_pad"],
            s0["activated"], s0["lora_input"], gun, nt, tk, 0, 1,  # grid_x_override=0, opt_mode=1
        )
        torch.cuda.synchronize()
        opt_eq = bool(
            torch.equal(s0["activated"], act_ref) and torch.equal(s0["lora_input"], li_ref)
        )
        print(f"{'PASS' if opt_eq else 'FAIL'} activation opt (vectorized) bitwise == scalar")

        # opt vs scalar with permutedIdx=-1 padding slots (real EP8 drops ~7/8 of slots to -1;
        # idx_map=arange above never exercises that branch). Inject -1s, run both kernels from the
        # SAME buffer state, assert bitwise-equal (padding writes 0 to lora_input, skips activated).
        idx_pad = s0["idx_map"].clone()
        idx_pad[1::3] = -1
        a0, l0 = s0["activated"].clone(), s0["lora_input"].clone()
        a_sc, l_sc = a0.clone(), l0.clone()
        a_op, l_op = a0.clone(), l0.clone()
        args_pad = (s0["gate_up"], s0["lora_delta"], idx_pad, s0["total_pad"])
        m.bench_activation(*args_pad, a_sc, l_sc, gun, nt, tk, 0, 0)  # scalar
        m.bench_activation(*args_pad, a_op, l_op, gun, nt, tk, 0, 1)  # opt
        torch.cuda.synchronize()
        pad_eq = bool(torch.equal(a_sc, a_op) and torch.equal(l_sc, l_op))
        print(f"{'PASS' if pad_eq else 'FAIL'} activation opt bitwise == scalar with -1 padding")

        # quant: dequantize the kernel output and compare to the bf16 input (fp4 has no
        # bitwise ref; assert the round-trip error is within fp4 precision).
        r[1](0)
        torch.cuda.synchronize()
        qrel = dequant_rel_err(
            s0["permuted"],
            s0["q1_fp4"],
            s0["q1_sf"],
            s0["q1_ptsf"],
            mp,
            H,
            args.tile,
            e,
        )
        q_ok = qrel <= 0.20
        print(
            f"{'PASS' if q_ok else 'FAIL'} nvfp4 quant#1 dequant vs input: "
            f"rel_err={qrel:.3e} (tol 0.20, e2m1 ~2^-1 mantissa)"
        )

        # quant #2 (down input): exercises the MAPPED path (m=num_tokens*top_k with
        # expanded_idx_to_permuted_idx). Reads `activated` rows via the map; same dequant check.
        r[3](0)
        torch.cuda.synchronize()
        q2rel = dequant_rel_err(
            s0["activated"], s0["q2_fp4"], s0["q2_sf"], s0["q2_ptsf"], mp, I, args.tile, e
        )
        q2_ok = q2rel <= 0.20
        print(
            f"{'PASS' if q2_ok else 'FAIL'} nvfp4 quant#2 (mapped) dequant vs input: "
            f"rel_err={q2rel:.3e} (tol 0.20)"
        )

        # fused permute+quant: NEW kernel vs the OLD permute->quant chain (= golden). The fused
        # kernel reads UN-permuted hidden_in and scatter-writes fp4+swizzled-sf+per-token-sf to the
        # permuted positions; quant#1 above read the permuted buffer (which permute filled from the
        # SAME hidden_in). It is a lossless refactor, so for the e valid rows the output must be
        # BITWISE-identical to quant#1's. Both dedup variants must match. Padding rows [e:maxpad)
        # are intentionally not written by the fused kernel, so compare only [0:e).
        num_vecs = H // 16
        rws = torch.arange(e, device=dev)[:, None]
        vcs = torch.arange(num_vecs, device=dev)[None, :]
        n_k_tiles = (num_vecs + 3) // 4
        sf_off = (rws // 8) * (n_k_tiles * 32) + (vcs // 4) * 32 + (rws % 8) * 4 + (vcs % 4)
        fused_ok = True
        for dedup in (0, 1):
            f_fp4 = torch.empty_like(s0["q1_fp4"])
            f_sf = torch.empty_like(s0["q1_sf"])
            f_ptsf = torch.empty_like(s0["q1_ptsf"])
            m.bench_fused_permute_quant(
                s0["hidden_in"], s0["idx_map"], f_fp4, f_sf, f_ptsf, nt, tk, H, mp, args.tile, dedup
            )
            torch.cuda.synchronize()
            fp4_eq = torch.equal(f_fp4[:e], s0["q1_fp4"][:e])
            ptsf_eq = torch.equal(f_ptsf[:e], s0["q1_ptsf"][:e])
            sf_eq = torch.equal(f_sf[sf_off], s0["q1_sf"][sf_off])
            ok = fp4_eq and ptsf_eq and sf_eq
            fused_ok = fused_ok and ok
            tag = "dedup" if dedup else "no-dedup"
            print(
                f"{'PASS' if ok else 'FAIL'} fused permute+quant ({tag}) vs old chain "
                f"[0:e) bitwise: fp4={fp4_eq} sf={sf_eq} ptsf={ptsf_eq}"
            )

        # fused act+quant: validity vs the separate (scalar activation -> quant#2) chain. The fused
        # kernel rounds activated to bf16 before quantizing (same as the separate path).
        # golden: scalar activation -> s0["activated"]/lora_input, then quant#2 on activated.
        m.bench_activation(
            s0["gate_up"], s0["lora_delta"], s0["idx_map"], s0["total_pad"],
            s0["activated"], s0["lora_input"], gun, nt, tk, 0, 0,
        )
        g_li = s0["lora_input"].clone()
        r[3](0)  # quant#2 on s0["activated"] -> s0["q2_fp4"]/q2_sf/q2_ptsf (golden)
        torch.cuda.synchronize()
        g_fp4, g_sf, g_ptsf = s0["q2_fp4"].clone(), s0["q2_sf"].clone(), s0["q2_ptsf"].clone()
        af_fp4 = torch.empty_like(s0["q2_fp4"])
        af_sf = torch.empty_like(s0["q2_sf"])
        af_ptsf = torch.empty_like(s0["q2_ptsf"])
        af_li = torch.empty_like(s0["lora_input"])
        m.bench_fused_act_quant(
            s0["gate_up"], s0["lora_delta"], s0["idx_map"],
            af_fp4, af_sf, af_ptsf, af_li, I, gun, nt, tk, args.tile,
        )
        torch.cuda.synchronize()
        # Validity gate (the installed flashinfer quant#2 uses a different internal cvt path, so
        # the fp4/SF/ptsf are NOT bit-identical to it): (1) activation_lora_input must be BITWISE
        # == the scalar activation (pure SwiGLU+LoRA, no quant), (2) the fused fp4 must dequantize
        # to the activated values within fp4 precision, like quant#2's own round-trip.
        li_eq = bool(torch.equal(af_li, g_li))
        afrel = dequant_rel_err(s0["activated"], af_fp4, af_sf, af_ptsf, mp, I, args.tile, e)
        grel = dequant_rel_err(s0["activated"], g_fp4, g_sf, g_ptsf, mp, I, args.tile, e)
        fused_aq_ok = li_eq and afrel <= 0.20
        print(
            f"{'PASS' if fused_aq_ok else 'FAIL'} fused act+quant valid: lora_input bitwise={li_eq}, "
            f"dequant-vs-activated rel={afrel:.3e} (quant#2 ref rel={grel:.3e}, tol 0.20)"
        )

        raise SystemExit(
            0
            if (
                perr == 0
                and act_ok
                and q_ok
                and q2_ok
                and gx_eq
                and opt_eq
                and pad_eq
                and fused_ok
                and fused_aq_ok
            )
            else 1
        )

    per = set_bytes(mk())
    n_sets = pick_n_sets(per, args.budget_gb, args.n_sets)
    print(
        f"BENCH fp4_lora_moe_kernels decode num_tokens={nt} maxpad={mp} hidden={H} inter={I}"
    )
    print(f"  per_set={per/1e6:.1f}MB {report_sets(per, n_sets)}")
    S = [mk() for _ in range(n_sets)]
    for name, call in runners(S).items():
        us = bench_kernel(call, n_sets) * 1000
        print(f"  {name:30s} = {us:7.2f} us")

    # activation sweeps (all bitwise-identical to the default scalar kernel; asserted in
    # --mode correctness). scalar: grid.x sweep (config-bound test, empty-block removal +
    # longer per-thread strip). opt: vectorized activationKernelOpt (128-bit gate/up + 64-bit
    # delta/store, 4 pairs/thread) which additionally raises memory-level parallelism.
    def act_call(gx, opt):
        return lambda i, gx=gx, opt=opt: m.bench_activation(
            S[i]["gate_up"],
            S[i]["lora_delta"],
            S[i]["idx_map"],
            S[i]["total_pad"],
            S[i]["activated"],
            S[i]["lora_input"],
            gun,
            nt,
            tk,
            gx,
            opt,
        )

    default_gx = gun // 128
    print("  -- activationKernel scalar grid.x sweep --")
    for gx in [default_gx, 16, 8, 4, 2, 1]:
        us = bench_kernel(act_call(gx, 0), n_sets) * 1000
        strip = (gun // 2 + 256 * gx - 1) // (256 * gx)
        tag = " (default)" if gx == default_gx else ""
        print(f"    grid.x={gx:3d}  (~{strip} elt/thread) = {us:7.2f} us{tag}")

    print("  -- activationKernelOpt vectorized (4 pairs/thread) grid.x sweep --")
    for gx in [0, 8, 4, 2, 1]:  # 0 -> opt default grid.x = ceil(innerHalf/4/256)
        us = bench_kernel(act_call(gx, 1), n_sets) * 1000
        gxs = "auto" if gx == 0 else f"{gx:4d}"
        print(f"    grid.x={gxs:>4s} = {us:7.2f} us")

    # fused act+quant vs the separate (opt activation + quant#2) pair it replaces.
    print("  -- fused act+quant (aggressive fusion) vs separate pair --")
    us_actopt = bench_kernel(act_call(0, 1), n_sets) * 1000  # opt activation alone
    us_q2 = (
        bench_kernel(
            lambda i: m.bench_nvfp4_quant(
                S[i]["activated"],
                S[i]["idx_map"],
                S[i]["q2_fp4"],
                S[i]["q2_sf"],
                S[i]["q2_ptsf"],
                nt * tk,
                I,
                args.tile,
            ),
            n_sets,
        )
        * 1000
    )
    us_fused = (
        bench_kernel(
            lambda i: m.bench_fused_act_quant(
                S[i]["gate_up"],
                S[i]["lora_delta"],
                S[i]["idx_map"],
                S[i]["q2_fp4"],
                S[i]["q2_sf"],
                S[i]["q2_ptsf"],
                S[i]["lora_input"],
                I,
                gun,
                nt,
                tk,
                args.tile,
            ),
            n_sets,
        )
        * 1000
    )
    print(f"    separate: opt activation {us_actopt:6.2f} + quant#2 {us_q2:6.2f} = {us_actopt + us_q2:6.2f} us")
    print(f"    fused act+quant                                   = {us_fused:6.2f} us")


if __name__ == "__main__":
    main()
