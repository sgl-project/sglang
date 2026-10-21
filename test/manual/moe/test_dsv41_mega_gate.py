import json
import os
import sys

import deep_gemm
import torch
from safetensors import safe_open

from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32
from sglang.kernels.ops.layernorm.mhc_mega import mhc_mega_boundary
from sglang.kernels.ops.moe.dsv4 import mega_moe_pre_dispatch
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.layers.moe.topk import _simulate_balanced_routing

torch.manual_seed(0)
dev = "cuda"
CKPT = sys.argv[1] if len(sys.argv) > 1 else None
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 10
E, K, H, RSF, IMAGE_ID = 384, 6, 5120, 1.5, 129264


def load_gate():
    names = [f"layers.{LAYER}.ffn.gate.{n}" for n in ("weight", "bias", "bias_vl")]
    try:
        index = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))
        out = []
        for n in names:
            with safe_open(
                os.path.join(CKPT, index["weight_map"][n]), "pt", device="cuda"
            ) as f:
                out.append(f.get_tensor(n))
        w, b, bvl = out
        print(f"real gate of layer {LAYER}: {tuple(w.shape)} {w.dtype}")
        return (
            w.to(torch.bfloat16).contiguous(),
            b.float().contiguous(),
            bvl.float().contiguous(),
        )
    except Exception as e:  # noqa: BLE001
        print(f"checkpoint not readable ({e!r}); random gate")
        return (
            (torch.randn(E, H, device=dev) * 0.02).to(torch.bfloat16),
            torch.randn(E, device=dev) * 0.1,
            torch.randn(E, device=dev) * 0.1,
        )


def check_gate(w, b, bvl, T, n_real, n_image):
    x = (torch.randn(T, H, device=dev) * 1.0).to(torch.bfloat16)
    ids = torch.randint(0, 100000, (T,), device=dev)
    ids[:n_image] = IMAGE_ID
    bound = torch.tensor([n_real], device=dev, dtype=torch.int32)
    logits = linear_bf16_fp32(x, w)
    ref_w, ref_i = moe_fused_gate(
        logits,
        b,
        topk=K,
        scoring_func="sqrtsoftplus",
        num_fused_shared_experts=0,
        bias_alt=bvl,
        input_ids=ids,
        bias_alt_token_id=IMAGE_ID,
        renormalize=True,
        renormalize_epsilon=1e-20,
        routed_scaling_factor=RSF,
        apply_routed_scaling_factor_on_output=False,
        num_token_non_padded=bound,
        packed_out=None,
        sqrtsoftplus_log1p=True,
    )
    ref_w = ref_w * RSF
    valid = torch.arange(T, device=dev, dtype=torch.int32) < bound
    image = ids == IMAGE_ID
    idx = torch.empty(T, K, device=dev, dtype=torch.int64)
    wts = torch.empty(T, K, device=dev, dtype=torch.float32)
    deep_gemm.bf16_mega_gate(
        x,
        w,
        K,
        False,
        0,
        RSF,
        0,
        "sqrtsoftplus",
        mask=valid,
        bias=b,
        image_bias=bvl,
        image_token_mask=image,
        out=(idx, wts),
    )
    real = slice(0, n_real)
    gi, go = idx[real].sort(dim=1)
    ri, ro = ref_i[real].long().sort(dim=1)
    same = (gi == ri).all(dim=1)
    gw, rw = wts[real].gather(1, go), ref_w[real].gather(1, ro)
    wdiff = (gw - rw).abs()[same].max().item() if same.any() else float("nan")
    pad_ok = bool((idx[n_real:] == -1).all() and (wts[n_real:] == 0).all())
    scores = torch.nn.functional.softplus(logits.float()).sqrt() + torch.where(
        image[:, None], bvl, b
    )
    top = scores.topk(K + 1, dim=1).values
    gap = (top[:, K - 1] - top[:, K])[real]
    worst_gap = gap[~same].max().item() if (~same).any() else 0.0
    print(
        f"gate T={T} real={n_real} image={n_image}: same top-k set {same.float().mean().item():.4f} "
        f"({int((~same).sum())} rows differ, largest k/k+1 score gap among them {worst_gap:.2e}); "
        f"max |dw| on same rows {wdiff:.2e}; padded rows (-1, 0): {pad_ok}"
    )
    return same.float().mean().item(), wdiff, pad_ok, worst_gap


def check_moe_input(T):
    hc = 4
    x = torch.randn(T, H, device=dev, dtype=torch.bfloat16)
    residual = torch.randn(T, hc, H, device=dev, dtype=torch.bfloat16)
    pre, post = torch.rand(T, hc, device=dev) * 0.5, torch.rand(T, hc, device=dev)
    comb = torch.rand(T, hc, hc, device=dev).softmax(-1)
    fn = torch.randn((2 + hc) * hc, hc * H, device=dev) * 0.01
    scale, base = (
        torch.rand(3, device=dev),
        torch.randn((2 + hc) * hc, device=dev) * 0.1,
    )
    weight = (torch.rand(H, device=dev) + 0.5).to(torch.bfloat16)
    args = (
        x,
        residual,
        pre,
        post,
        comb,
        fn,
        scale,
        base,
        weight,
        1e-6,
        1e-6,
        1e-20,
        20,
    )
    P, G4 = 4096, H // 128
    bm = deep_gemm.get_block_m_for_mega_moe(8, E, 16384, T, K, "fp8xfp4")
    S = (P + bm - 1) // bm * ((bm + 127) // 128 * 128)

    def bufs():
        return (
            torch.zeros(P, H, device=dev, dtype=torch.float8_e4m3fn),
            torch.zeros(P, G4, device=dev, dtype=torch.int32),
            torch.zeros(G4, S, device=dev, dtype=torch.int32).mT,
            torch.full((P, K), 7, device=dev, dtype=torch.int64),
            torch.full((P, K), 0.5, device=dev, dtype=torch.float32),
        )

    _, y, _ = mhc_mega_boundary(*args)
    ax, asf, ash, aidx, aw = bufs()
    mega_moe_pre_dispatch(
        y,
        aidx,
        aw,
        ax,
        asf,
        aidx,
        aw,
        quant_group_size=32,
        shared_x_sf=ash,
        shared_block_m=bm,
        copy_topk=False,
    )
    bx, bsf, bsh, bidx, bw = bufs()
    _, y2, _ = mhc_mega_boundary(*args, moe_input=(bx[:T], bsf[:T], bsh[:T], bm))
    rows = (T + bm - 1) // bm * ((bm + 127) // 128 * 128)
    ok = (
        torch.equal(y, y2)
        and torch.equal(ax.view(torch.uint8), bx.view(torch.uint8))
        and torch.equal(asf, bsf)
        and torch.equal(ash[:rows], bsh[:rows])
    )
    untouched = bool((aidx == 7).all() and (aw == 0.5).all())
    print(
        f"moe input T={T} block_m={bm}: Mega-mHC bytes == quantize-only pre-dispatch bytes: {ok}; "
        f"quantize-only left the routing untouched: {untouched}"
    )
    return ok and untouched


def check_simulation(T, n_real):
    idx = torch.randint(0, E, (T, K), device=dev, dtype=torch.int64)
    w = torch.rand(T, K, device=dev)
    bound = torch.tensor([n_real], device=dev, dtype=torch.int32)
    i32, w32 = idx.int().clone(), w.clone()
    _simulate_balanced_routing(i32, w32, E, random=False, num_ranks=8, layer_id=3)
    _simulate_balanced_routing(
        idx,
        w,
        E,
        random=False,
        num_ranks=8,
        layer_id=3,
        num_token_non_padded=bound,
        weight_scale=RSF,
    )
    ok = (
        torch.equal(idx[:n_real], i32[:n_real].long())
        and torch.allclose(w[:n_real], w32[:n_real] * RSF)
        and bool((idx[n_real:] == -1).all() and (w[n_real:] == 0).all())
    )
    print(f"simulation T={T} real={n_real}: int64 + mask + scale matches: {ok}")
    return ok


w, b, bvl = load_gate()
results = [
    check_gate(w, b, bvl, T, n, img)
    for T, n, img in [
        (1, 1, 0),
        (256, 250, 3),
        (1024, 1000, 0),
        (1024, 1024, 17),
        (2048, 2040, 0),
    ]
]
moe_ok = all(check_moe_input(T) for T in (1, 37, 256, 1024, 2048))
sim_ok = all(check_simulation(T, n) for T, n in ((256, 250), (1024, 1024)))
assert all(p for _, _, p, _ in results)
assert min(r for r, _, _, _ in results) > 0.995, "top-k sets differ beyond near ties"
assert max(g for _, _, _, g in results) < 1e-3, "a differing row is not a near tie"
assert max(d for _, d, _, _ in results if d == d) < 1e-4
assert moe_ok and sim_ok
print("PASS")
