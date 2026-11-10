"""SM70 MoE glue: the clamped SwiGLU must equal swiglu_limit_func bitwise over
every fp16 input; the router logits must be as accurate as the cuBLAS fp32
path they replace (FP64 reference) and route the same experts.

Not a CI test:

  CUDA_VISIBLE_DEVICES=0 python test/manual/layers/test_sm70_moe_glue.py [ckpt_dir]
"""

import json
import sys

import torch

from sglang.kernels.ops.moe import sm70_moe_glue as glue
from sglang.kernels.ops.moe.sm70_router_topk import sm70_router_topk
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import swiglu_limit_func

DEV = "cuda"


def all_halves() -> torch.Tensor:
    return torch.arange(-(2**15), 2**15, dtype=torch.int32).to(torch.int16).view(torch.float16).to(DEV)


def swiglu_case(name, gate, up, limit):
    d = 256
    gate_up = torch.cat([gate.view(-1, d), up.view(-1, d)], dim=1).contiguous()
    ref = torch.empty(gate_up.size(0), d, dtype=torch.float16, device=DEV)
    got = torch.empty_like(ref)
    swiglu_limit_func(ref, gate_up, limit)
    glue.swiglu_clamp(got, gate_up, limit)
    rb, gb = ref.view(torch.int16), got.view(torch.int16)
    same = rb == gb
    both_nan = ref.isnan() & got.isnan()
    bad = ~(same | both_nan)
    print(
        f"swiglu {name:36s} limit={limit:5.1f}  {'equal' if not bad.any() else 'DIFFER'}"
        f"  (nan payload diffs {int((both_nan & ~same).sum())})"
    )
    if bad.any():
        i = bad.nonzero()[0].tolist()
        print("  gate", gate_up[i[0], i[1]].item(), "up", gate_up[i[0], d + i[1]].item(), "ref", ref[tuple(i)].item(),
              "got", got[tuple(i)].item())
    return not bad.any()


def test_swiglu() -> bool:
    g = torch.Generator(device=DEV).manual_seed(0)
    h = all_halves()
    ok = True
    for limit in (10.0, 0.0, 10.3, 7.0):
        ok &= swiglu_case("every gate, up=1", h, torch.ones_like(h), limit)
        ok &= swiglu_case("every gate, up=-3.5", h, torch.full_like(h, -3.5), limit)
        ok &= swiglu_case("every up, gate=1.25", torch.full_like(h, 1.25), h, limit)
        ok &= swiglu_case("every up, gate=every gate reversed", h.flip(0), h, limit)
        bits = torch.randint(-(2**15), 2**15, (2, 4096 * 256), device=DEV, generator=g, dtype=torch.int32)
        halves = bits.to(torch.int16).view(torch.float16)
        ok &= swiglu_case("random bit patterns", halves[0], halves[1], limit)
        x = (torch.randn(2, 64 * 256, device=DEV, generator=g) * 6).half()
        ok &= swiglu_case("randn*6 (decode-like, 64 rows)", x[0], x[1], limit)
    return ok


def router_case(name, x, w, bias):
    ref64 = x.double() @ w.double().t()
    cublas = torch.mm(x.float(), w.float().t())
    got = glue.router_logits(x, w)
    again = glue.router_logits(x, w)
    err_k = (got.double() - ref64).abs()
    err_c = (cublas.double() - ref64).abs()
    scale = ref64.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    ok_det = torch.equal(got.view(torch.int32), again.view(torch.int32))
    _, ids_k = sm70_router_topk(got, bias, True, 2.5, False)
    _, ids_c = sm70_router_topk(cublas, bias, True, 2.5, False)
    _, ids_r = sm70_router_topk(ref64.float(), bias, True, 2.5, False)
    ids_k, ids_c, ids_r = ids_k.sort(-1)[0], ids_c.sort(-1)[0], ids_r.sort(-1)[0]
    print(
        f"logits {name:34s} max|err|/max|y| kernel {float((err_k / scale).max()):.2e} cublas {float((err_c / scale).max()):.2e}"
        f"  mean|err| kernel {float(err_k.mean()):.2e} cublas {float(err_c.mean()):.2e}"
        f"  det {'yes' if ok_det else 'NO'}"
        f"  route!=fp64: kernel {int((ids_k != ids_r).any(-1).sum())} cublas {int((ids_c != ids_r).any(-1).sum())} of {x.size(0)}"
    )
    return ok_det and float(err_k.mean()) <= 2 * float(err_c.mean()) + 1e-12


def load_router(ckpt: str, layer: int):
    from safetensors import safe_open

    idx = json.load(open(f"{ckpt}/model.safetensors.index.json"))["weight_map"]
    tensors = []
    for suffix in ("weight", "e_score_correction_bias"):
        key = f"model.language_model.layers.{layer}.mlp.gate.{suffix}"
        with safe_open(f"{ckpt}/{idx[key]}", "pt") as f:
            tensors.append(f.get_tensor(key).to(DEV))
    return tensors[0].half().contiguous(), tensors[1].float().contiguous()


def test_router(ckpt: str | None) -> bool:
    g = torch.Generator(device=DEV).manual_seed(1)
    ok = True
    for n, k in ((288, 4096), (320, 7168), (304, 1024)):
        w = (torch.randn(n, k, device=DEV, generator=g) * 0.02).half()
        bias = torch.randn(n, device=DEV, generator=g) * 0.01
        for m in (1, 2, 3, 4):
            x = (torch.randn(m, k, device=DEV, generator=g) * 3).half()
            ok &= router_case(f"randn N={n} K={k} M={m}", x, w, bias)
    if ckpt:
        for layer in (3, 20, 44):
            w, bias = load_router(ckpt, layer)
            xs = (torch.randn(4000, 4, w.size(1), device=DEV, generator=g) * 2).half()
            outs_k, outs_c, ref = [], [], []
            for x in xs:
                outs_k.append(glue.router_logits(x, w))
                outs_c.append(torch.mm(x.float(), w.float().t()))
            got, cub = torch.cat(outs_k), torch.cat(outs_c)
            ref64 = xs.reshape(-1, w.size(1)).double() @ w.double().t()
            _, ik = sm70_router_topk(got, bias, True, 2.5, False)
            _, ic = sm70_router_topk(cub, bias, True, 2.5, False)
            _, ir = sm70_router_topk(ref64.float(), bias, True, 2.5, False)
            ik, ic, ir = ik.sort(-1)[0], ic.sort(-1)[0], ir.sort(-1)[0]
            ek = (got.double() - ref64).abs().mean()
            ec = (cub.double() - ref64).abs().mean()
            print(
                f"logits ckpt layer {layer:2d} 16000 tokens: mean|err| kernel {float(ek):.2e} cublas {float(ec):.2e}"
                f"  route!=fp64 kernel {int((ik != ir).any(-1).sum())} cublas {int((ic != ir).any(-1).sum())}"
                f"  kernel!=cublas {int((ik != ic).any(-1).sum())}"
            )
            ok &= float(ek) <= 2 * float(ec)
    return ok


def graph_us(fn, reps=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(reps):
            fn()
    best = 1e9
    for _ in range(20):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        graph.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / reps * 1e3)
    return best


def timing() -> None:
    x = torch.randn(1, 4096, device=DEV).half()
    w = (torch.randn(288, 4096, device=DEV) * 0.02).half()
    print(f"router logits: torch {graph_us(lambda: torch.mm(x.float(), w.float().t())):.1f} us"
          f"  kernel {graph_us(lambda: glue.router_logits(x, w)):.1f} us")
    gate_up = torch.randn(8, 512, device=DEV).half()
    out = torch.empty(8, 256, device=DEV, dtype=torch.float16)
    print(f"swiglu clamp: torch {graph_us(lambda: swiglu_limit_func(out, gate_up, 10.0)):.1f} us"
          f"  kernel {graph_us(lambda: glue.swiglu_clamp(out, gate_up, 10.0)):.1f} us")


def main() -> None:
    ok = test_swiglu() if "--skip-swiglu" not in sys.argv else True
    ok &= test_router(next((a for a in sys.argv[1:] if not a.startswith("--")), None))
    timing()
    assert ok, "sm70 MoE glue differs"
    print("ALL OK")


if __name__ == "__main__":
    main()
