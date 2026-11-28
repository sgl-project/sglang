"""Inkling2 routed-expert LoRA kernels against their torch references, at Inkling 2 Mini's sizes.

Rows are routed as Inkling2Experts routes them: sorted by expert, one per (token, choice);
an EP rank owns its experts' rows and sees the other choices outside its row range.
"""

import pytest
import torch

from sglang.kernels.ops.lora.moe import inkling2_routed
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels import inkling2_lora_reference as ref

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Inkling2 LoRA kernels need CUDA"
)

_D, _F, _E, _TOP_K, _R = 2304, 1152, 128, 8, 16  # Inkling 2 Mini, LoRA rank 16
_CAPS = (4.0, 25.0)  # arbitrary; nonzero caps exercise the capped SwiGLU's tanh


def _routing(tokens: int, *, ep_size: int, ep_rank: int, seed: int):
    """dst2src [rows], src2dst [tokens * top_k], offsets [local experts + 1] for one EP rank."""
    g = torch.Generator().manual_seed(seed)
    choices = torch.stack(
        [torch.randperm(_E, generator=g)[:_TOP_K] for _ in range(tokens)]
    )
    local = _E // ep_size
    first = ep_rank * local
    src = torch.arange(tokens * _TOP_K)
    expert = choices.flatten()
    mine = (expert >= first) & (expert < first + local)
    order = torch.argsort(expert[mine] * (tokens * _TOP_K) + src[mine])
    dst2src = src[mine][order]
    counts = torch.bincount(expert[mine] - first, minlength=local)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    # choices on other ranks point past this rank's rows
    src2dst = torch.full((tokens * _TOP_K,), dst2src.numel() + 7, dtype=torch.long)
    src2dst[dst2src] = torch.arange(dst2src.numel())
    return dst2src.int(), src2dst.int(), offsets.int()


def _nvfp4_scales(rows: int, local: int, g: torch.Generator):
    x_amax = torch.rand(rows, generator=g) * 4 + 0.5
    w_amax = torch.rand(local, generator=g) * 0.2 + 0.01
    return x_amax, w_amax, torch.tensor([6.0])


def _cuda(*tensors):
    return [t.cuda() for t in tensors]


CASES = [
    # tokens, ep_size, nvfp4 (fp32 unscaled accumulator) or not (bf16 scaled, MXFP8 / bf16 paths)
    pytest.param(1, 1, True, id="decode-1tok-ep1-nvfp4"),
    pytest.param(37, 1, True, id="37tok-ep1-nvfp4"),
    pytest.param(37, 1, False, id="37tok-ep1-bf16acc"),
    pytest.param(256, 1, True, id="256tok-ep1-nvfp4"),
    pytest.param(64, 2, True, id="64tok-ep2-nvfp4"),
    pytest.param(64, 2, False, id="64tok-ep2-bf16acc"),
]


def _inputs(tokens, ep_size, nvfp4, seed=0):
    dst2src, src2dst, offsets = _routing(
        tokens, ep_size=ep_size, ep_rank=ep_size - 1, seed=seed
    )
    local, rows = _E // ep_size, dst2src.numel()
    g = torch.Generator().manual_seed(seed + 1)
    acc_dtype = torch.float32 if nvfp4 else torch.bfloat16
    # the raw NVFP4 GEMM output is the scaled one over s; keep both near unit size after scaling
    acc = torch.randn(rows, 2 * _F, generator=g)
    scales = _nvfp4_scales(rows, local, g) if nvfp4 else None
    if nvfp4:
        acc = acc / ref._row_scale(scales, ref._row_experts(offsets, rows))[:, None]
    return dict(
        h=(torch.randn(tokens, 2 * _R, generator=g)).bfloat16(),
        b=(torch.randn(local, 2 * _F, _R, generator=g) * 0.1).bfloat16(),
        a=(torch.randn(local, _R, _F, generator=g) * 0.05).bfloat16(),
        acc=acc.to(acc_dtype),
        gammas=torch.rand(rows, generator=g),
        mask=(torch.rand(tokens, generator=g) > 0.2).float(),
        dst2src=dst2src,
        src2dst=src2dst,
        offsets=offsets,
        scales=scales,
    )


def _run(module, x, *, device):
    to = (lambda t: t.to(device)) if device == "cuda" else (lambda t: t)
    scales = tuple(to(s) for s in x["scales"]) if x["scales"] is not None else None
    acc = to(x["acc"].clone())
    module.routed_gate_up_delta(
        h=to(x["h"]),
        b=to(x["b"]),
        acc=acc,
        dst2src=to(x["dst2src"]),
        offsets=to(x["offsets"]),
        top_k=_TOP_K,
        global_experts=_E,
        nvfp4_scales=scales,
    )
    inner = module.routed_down_inner(
        acc=acc,
        gammas=to(x["gammas"]),
        a=to(x["a"]),
        offsets=to(x["offsets"]),
        gate_cap=_CAPS[0],
        up_cap=_CAPS[1],
        global_experts=_E,
        nvfp4_scales=scales,
    )
    ranks = module.routed_token_rank(
        inner=inner,
        src2dst=to(x["src2dst"]),
        offsets=to(x["offsets"]),
        mask=to(x["mask"]),
        top_k=_TOP_K,
    )
    return acc.cpu(), inner.cpu(), ranks.cpu()


def _rel(got, want):
    return (
        (got.float() - want.float()).norm() / want.float().norm().clamp_min(1e-30)
    ).item()


@pytest.mark.parametrize("tokens,ep_size,nvfp4", CASES)
def test_kernels_match_torch_reference(tokens, ep_size, nvfp4):
    x = _inputs(tokens, ep_size, nvfp4)
    acc, inner, ranks = _run(inkling2_routed, x, device="cuda")
    acc_ref, inner_ref, ranks_ref = _run(ref, x, device="cpu")
    rows = x["dst2src"].numel()
    # gate/up: bf16 products, fp32 sums in both; the delta must move the accumulator
    assert not torch.equal(acc[:rows], x["acc"][:rows])
    assert _rel(acc - x["acc"].float(), acc_ref - x["acc"].float()) < 1e-3
    # A2 projection: the kernel's capped SwiGLU uses approximate exp / tanh, so h can differ by one bf16 step
    assert _rel(inner[:rows], inner_ref[:rows]) < 1e-2
    assert _rel(ranks, ranks_ref) < 1e-2
    # tokens without an adapter get an exactly zero rank-space partial
    assert torch.all(ranks[x["mask"] == 0] == 0)


def test_rank_partials_sum_across_ep_ranks():
    """Each EP rank sums only its own rows; the all-reduce of the [T, r] partials is the EP=1 result."""
    tokens = 48
    full = _inputs(tokens, 1, nvfp4=False, seed=3)
    _, _, ranks_full = _run(inkling2_routed, full, device="cuda")
    local = _E // 2
    total = torch.zeros_like(ranks_full)
    for ep_rank in range(2):
        dst2src, src2dst, offsets = _routing(tokens, ep_size=2, ep_rank=ep_rank, seed=3)
        part = dict(full)
        part.update(
            acc=_by_source(full, "acc", dst2src),
            gammas=_by_source(full, "gammas", dst2src),
            b=full["b"][ep_rank * local : (ep_rank + 1) * local],
            a=full["a"][ep_rank * local : (ep_rank + 1) * local],
            dst2src=dst2src,
            src2dst=src2dst,
            offsets=offsets,
        )
        total += _run(inkling2_routed, part, device="cuda")[2]
    assert _rel(total, ranks_full) < 1e-5


def _by_source(full, name, dst2src):
    """The EP=1 rows of `name` for the given (token, choice) sources, in this rank's row order."""
    by_src = torch.empty_like(full[name])
    by_src[full["dst2src"].long()] = full[name]
    return by_src[dst2src.long()]


def test_cuda_graph_replay_matches_eager():
    """A captured LoRA step replays new inputs written into its static buffers."""
    first, second = _inputs(16, 1, True, seed=5), _inputs(16, 1, True, seed=6)
    second.update(
        dst2src=first["dst2src"], src2dst=first["src2dst"], offsets=first["offsets"]
    )
    static = {
        k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in first.items()
    }
    static["scales"] = tuple(s.cuda() for s in first["scales"])
    acc = static["acc"].clone()

    def step():
        inkling2_routed.routed_gate_up_delta(
            h=static["h"],
            b=static["b"],
            acc=acc,
            dst2src=static["dst2src"],
            offsets=static["offsets"],
            top_k=_TOP_K,
            global_experts=_E,
            nvfp4_scales=static["scales"],
        )
        inner = inkling2_routed.routed_down_inner(
            acc=acc,
            gammas=static["gammas"],
            a=static["a"],
            offsets=static["offsets"],
            gate_cap=_CAPS[0],
            up_cap=_CAPS[1],
            global_experts=_E,
            nvfp4_scales=static["scales"],
        )
        return inkling2_routed.routed_token_rank(
            inner=inner,
            src2dst=static["src2dst"],
            offsets=static["offsets"],
            mask=static["mask"],
            top_k=_TOP_K,
        )

    step()  # compile outside capture
    acc.copy_(static["acc"])
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step()
    for name in ("h", "b", "a", "gammas", "mask"):
        static[name].copy_(second[name].cuda())
    for mine, theirs in zip(static["scales"], second["scales"]):
        mine.copy_(theirs.cuda())
    acc.copy_(second["acc"].cuda())
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.cpu().clone()
    _, _, eager = _run(inkling2_routed, second, device="cuda")
    assert torch.equal(replayed, eager)
