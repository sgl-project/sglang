"""Drift guard of the DeepSeek-V4.1 mono K1 launch on gfx950: K1 must stay bit-exact with SGLang's own decode
chain for the attention seam and the SWA KV write, so a change on either side (SGLang's boundary / norm / wqkv_a
kernels or their tuned configs, or K1) fails here rather than in served accuracy.

On fixed wide-range data (varied block scales, non-unit norms, outlier channels), per width:
- SGLang's wqkv_a route is the one K1 follows (``mxfp8_gemv_gfx95_configs.json`` entries for N 1792, K 5120);
- residual_out, X8 and its E8M0 scales, and every SWA record (NOPE codes, scales, rope) are bitwise equal
  to ``hc_boundary_fused_deferred`` -> ``rmsnorm_with_sinkhorn`` -> ``mxfp8_native_blockscaled_linear`` ->
  ``fused_k_norm_rope_flashmla`` on the same inputs;
- pre / post / comb are within ``MIX_TOL`` (K1 sums the mixes in another order).

Run: torchrun --nproc-per-node 4 test_dsv41_mono_k1_drift.py --widths 4   (TP2: --nproc-per-node 2 --widths 48)
"""

import argparse
import math
import os

import torch
import torch.distributed as dist
from test_dsv41_mono_ffn import (
    HC,
    HC_EPS,
    HIDDEN,
    RMS_EPS,
    make_weights,
    mono_weights,
    seam_inputs,
)

RECORD, BLOCK, PAGES, POSITIONS = 584, 64, 64, 8192
Q_RANK = 1280
MIX_TOL = 1e-6  # measured <= 5.4e-7 (TP2 / TP4 synthetic, TP4 server)
# SGLang's wqkv_a configs K1 is matched to: the GEMV per M bucket up to 32 rows, the split-K tile above
GEMV = {1: "w16s1r16t16k", 2: "w16s1r16t16k", 4: "w16s1r16t16k", 8: "w8s2r16t16k", 16: "w8s2r16t16k",
        32: "w8s1r16t32k"}  # fmt: skip
LARGE_M = (64, 64, 256, 4, 5)


def attn_weights(heads, dev, g):
    """K1's weights of one layer, wide-range: e4m3 [N, K] with varied E8M0 32 x 32 block codes."""

    def fp8(n, k, lo, hi):
        w = (torch.randn(n, k, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
        return w, torch.randint(lo, hi, (n // 32, k // 32), generator=g, device=dev).to(
            torch.uint8
        )

    wqkv, wqkv_s = fp8(1792, HIDDEN, 116, 126)
    wqb, wqb_s = fp8(heads * 512, 1280, 121, 122)
    q_norm = torch.ones(1280, dtype=torch.bfloat16, device=dev)
    kv_norm = (torch.randn(512, generator=g, device=dev).abs() * 2 + 0.05).to(
        torch.bfloat16
    )
    return dict(
        wqkv=wqkv, wqkv_s=wqkv_s, q_norm=q_norm, kv_norm=kv_norm, wqb=wqb, wqb_s=wqb_s
    )


def rope_table(dev):
    j = torch.arange(32, device=dev, dtype=torch.float64)
    ang = torch.arange(POSITIONS, device=dev, dtype=torch.float64)[
        :, None
    ] * 10000.0 ** (-j / 32)
    return torch.cat([ang.cos(), ang.sin()], -1).float().contiguous()


def check_wqkv_route(M):
    """SGLang's selected wqkv_a config at M is the one K1 implements (``plan.wqkv_order``)."""
    from sglang.kernels.ops.quantization import mxfp8_native_amd_gfx95 as mx
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.attention.plan import (
        wqkv_order,
    )

    if M <= 32:
        b = mx._m_bucket(M)
        got, want = mx._select_config(b, 1792, HIDDEN), mx._GemvConfig.parse(GEMV[b])
        return got == want and wqkv_order(M)[0] == want.waves, f"GEMV {got}"
    got = tuple(mx._large_m_tile(M, 1792, HIDDEN))
    return got == LARGE_M and wqkv_order(M)[0] == LARGE_M[4], f"tile {got}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", default="4,48")
    ap.add_argument("--trials", type=int, default=4)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.kernels.ops.attention.dsv4.elementwise import fused_k_norm_rope_flashmla
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        hc_boundary_fused_deferred,
        rmsnorm_with_sinkhorn,
    )
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
        prepare_mxfp8_native_weight,
    )
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.layer import (
        MonoBuild,
        build_mono_k1,
        scratch_layout,
    )
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import (
        MAX_TOKENS,
        DSV41MonoLayer,
    )

    w, _, _ = make_weights(tp, rank, dev, seed=0)
    m = mono_weights(w)
    del w
    r = DSV41MonoLayer(tp, rank, cpu, dev)
    g = torch.Generator(device=dev).manual_seed(11)
    a = attn_weights(r.q.shape[1], dev, g)
    m.ffn_norm.copy_(
        (torch.randn(HIDDEN, generator=g, device=dev) * 1.5).exp().to(m.ffn_norm.dtype)
    )
    cos_sin = rope_table(dev)
    freqs = torch.complex(cos_sin[:, :32], cos_sin[:, 32:])  # SGLang's freqs_cis
    w_sh, w_su = prepare_mxfp8_native_weight(
        a["wqkv"], torch.exp2(a["wqkv_s"].float() - 127), (32, 32)
    )
    stride = math.ceil(BLOCK * RECORD / 576) * 576
    dummy = torch.zeros(MAX_TOKENS * 128, dtype=torch.int32, device=dev)

    def records(buf, s):
        """The 584 bytes of slots ``s``: 576 data bytes (448 NOPE e4m3, 64 bf16 rope), 8 scale bytes (7, pad)."""
        page, off = s // BLOCK, s % BLOCK
        col = torch.cat([off[:, None] * 576 + torch.arange(576, device=dev),
                         BLOCK * 576 + off[:, None] * 8 + torch.arange(8, device=dev)], 1)  # fmt: skip
        return buf[page[:, None].expand_as(col), col]

    fails = []
    for M in [int(v) for v in args.widths.split(",")]:
        ok, route = check_wqkv_route(M)
        k1 = build_mono_k1(MonoBuild(M, tp, 0))
        lay = scratch_layout(M, tp)
        diff = dict(res=0, x8=0, x8s=0, nope=0, scale=0, rope=0)
        mix = 0.0
        for _ in range(args.trials):
            _, res, post, comb, pre = seam_inputs(M, tp, rank, dev)
            res = (res * (torch.randn(HIDDEN, generator=g, device=dev) * 1.5).exp()).to(
                res.dtype
            )
            x = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.1).to(
                torch.bfloat16
            )
            pos = torch.randint(0, POSITIONS, (M,), generator=g, device=dev)
            slot = torch.randperm(PAGES * BLOCK, generator=g, device=dev)[:M]
            swa = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)
            sc = r.scratch(M)
            r.epoch[0] += (
                1  # the FFN launch moves the epoch; K1 alone must not see the last trial's tags
            )
            k1(
                res.data_ptr(), x.data_ptr(), post.data_ptr(), comb.data_ptr(), pre.data_ptr(),
                m.hc_ffn_fn.data_ptr(), m.hc_ffn_scale.data_ptr(), m.hc_ffn_base.data_ptr(),
                m.ffn_norm.data_ptr(),
                r.res_mid.data_ptr(), r.post_a.data_ptr(), r.comb_a.data_ptr(), r.pre_a.data_ptr(),
                a["wqkv"].data_ptr(), a["wqkv_s"].data_ptr(), a["q_norm"].data_ptr(), a["kv_norm"].data_ptr(),
                a["wqb"].data_ptr(), a["wqb_s"].data_ptr(), cos_sin.data_ptr(), pos.data_ptr(),
                slot.data_ptr(), swa.data_ptr(), swa.stride(0), BLOCK, r.q.data_ptr(),
                dummy.data_ptr(), dummy.data_ptr(), 1, r._dummy.data_ptr(), 0, r._dummy.data_ptr(),
                dummy.data_ptr(), r.kt.data_ptr(), r.klen.data_ptr(), sc.data_ptr(),
                r.epoch.data_ptr(), 0, stream=torch.cuda.current_stream(),
            )  # fmt: skip
            torch.cuda.synchronize()
            o = lay["x8"][0]
            x8 = sc[o : o + M * HIDDEN]
            o = lay["x8s"][0]
            x8s = (
                sc[o : o + M * HIDDEN // 32 * 4]
                .view(torch.int32)
                .view(M, HIDDEN // 32)
                .to(torch.uint8)
            )
            got = records(swa, slot.long())

            ro, y, co = hc_boundary_fused_deferred(
                x,
                res,
                post,
                comb,
                pre,
                m.hc_ffn_fn,
                m.hc_ffn_scale,
                m.hc_ffn_base,
                HC,
                20,
                RMS_EPS,
                HC_EPS,
            )
            xq, _ = rmsnorm_with_sinkhorn(y, m.ffn_norm, RMS_EPS, co, emit_fp8=True)
            pre_r, post_r, comb_r = co.tensors()
            qkv = mxfp8_native_blockscaled_linear(
                xq.q[:M], w_sh, w_su, input_scale=xq.scale[:M]
            )
            ref = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)
            fused_k_norm_rope_flashmla(
                qkv[:, Q_RANK:].contiguous(),
                a["kv_norm"],
                RMS_EPS,
                freqs,
                pos,
                slot.int(),
                ref,
                BLOCK,
            )
            want = records(ref, slot.long())

            diff["res"] += int(
                (r.res_mid[:M].view(torch.int16) != ro.view(torch.int16)).sum()
            )
            diff["x8"] += int(
                (x8 != xq.q[:M].contiguous().view(torch.uint8).flatten()).sum()
            )
            diff["x8s"] += int(
                (
                    x8s != xq.scale[:M].contiguous().view(torch.uint8).reshape(M, -1)
                ).sum()
            )
            # byte 583 pads the 7 scales: no reader, and SGLang's store leaves it as it was
            for name, cols in (
                ("nope", slice(0, 448)),
                ("rope", slice(448, 576)),
                ("scale", slice(576, 583)),
            ):
                diff[name] += int((got[:, cols] != want[:, cols]).sum())
            for k1_t, ref_t in (
                (r.pre_a, pre_r),
                (r.post_a, post_r),
                (r.comb_a, comb_r),
            ):
                mix = max(
                    mix, (k1_t[:M].reshape(ref_t.shape) - ref_t).abs().max().item()
                )
        bad = (
            ([] if ok else ["wqkv route"])
            + [k for k, v in diff.items() if v]
            + (["mix"] if mix > MIX_TOL else [])
        )
        if rank == 0:
            print(f"M={M:2d} {'PASS' if not bad else 'FAIL ' + ','.join(bad)} | wqkv_a {route} | differing: "
                  f"{diff} | pre/post/comb max|d| {mix:.2e} (bound {MIX_TOL:.0e})", flush=True)  # fmt: skip
        fails += [f"M={M} {b}" for b in bad]
    if rank == 0:
        print("ALL PASS" if not fails else f"FAILED: {fails}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
