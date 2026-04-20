"""B=1 kernel-level sweep for the head-aware extend-attention dispatch.

Measures pure GPU time via CUDA events across ShareGPT-style B=1
prefill shapes under the prior (non-head-aware) and the new head-aware
dispatch rules. Pins both runs to the basic kernel path
(``_force_use_splitk=_force_use_persistent=False``) so the comparison
isolates config selection from kernel-selection noise.

Columns: Triton reference, OLD (forced to pre-change config),
NEW (auto dispatch, cache-warm), gain% = (OLD - NEW) / OLD.
"""
import torch
from sglang.srt.layers.attention.gluon_ops.cdna4.extend_attention.extend_attention_gfx950 import (
    gluon_extend_attention_fwd,
    _get_basic_dispatch_config,
)
from sglang.srt.layers.attention.triton_ops.extend_attention import (
    extend_attention_fwd as triton_fwd,
)

BF16 = torch.bfloat16; DEV='cuda:0'

def bench_gpu(fn, warmup=50, iters=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record(); fn(); ends[i].record()
    torch.cuda.synchronize()
    total_ms = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
    return total_ms * 1000 / iters

def mk(B, S, H_q, H_kv, D):
    torch.manual_seed(0); total = B*S
    qo = torch.arange(0, total+1, S, device=DEV, dtype=torch.int32)
    kvp = torch.arange(0, total+1, S, device=DEV, dtype=torch.int32)
    kvi = torch.arange(0, total, device=DEV, dtype=torch.int32)
    q = torch.randn(total, H_q, D, device=DEV, dtype=BF16)/8
    k = torch.randn(total, H_kv, D, device=DEV, dtype=BF16)/8
    v = torch.randn(total, H_kv, D, device=DEV, dtype=BF16)/8
    o = torch.empty_like(q); psz = total+16
    kb = torch.zeros(psz, H_kv, D, device=DEV, dtype=BF16); kb[:total]=k
    vb = torch.zeros(psz, H_kv, D, device=DEV, dtype=BF16); vb[:total]=v
    mi = torch.zeros(B+1, device=DEV, dtype=torch.int32)
    return (q,k,v,o,kb,vb,qo,kvp,kvi,mi,S)


def call_forced(inp, BM, NW, NS):
    q,k,v,o,kb,vb,qo,kvp,kvi,mi,S = inp
    gluon_extend_attention_fwd(q,k,v,o,kb,vb,qo,kvp,kvi,None,True,mi,S,1.0,1.0,
        _force_block_m=BM, _force_num_warps=NW, _force_num_stages=NS,
        _force_use_splitk=False, _force_use_persistent=False)

def call_auto(inp):
    q,k,v,o,kb,vb,qo,kvp,kvi,mi,S = inp
    gluon_extend_attention_fwd(q,k,v,o,kb,vb,qo,kvp,kvi,None,True,mi,S,1.0,1.0)

def call_triton(inp):
    q,k,v,o,kb,vb,qo,kvp,kvi,mi,S = inp
    triton_fwd(q,k,v,o,kb,vb,qo,kvp,kvi,None,True,mi,S,1.0,1.0)


def _old_cfg_b1(S, H_q, D):
    """Pre-change ``_get_basic_dispatch_config`` output for B=1."""
    if D == 128:
        return (64, 4, 2)
    elif D == 64:
        if S >= 2048:
            return (256, 8, 4) if S <= 8192 else (256, 8, 2)
        else:
            return (128, 8, 4)
    return None


SHAPES = [
    ("D128 H=64 S=199",  1, 199,  64, 8, 128),
    ("D128 H=64 S=512",  1, 512,  64, 8, 128),
    ("D128 H=64 S=1024", 1, 1024, 64, 8, 128),
    ("D128 H=64 S=1500", 1, 1500, 64, 8, 128),
    ("D128 H=64 S=1750", 1, 1750, 64, 8, 128),
    ("D128 H=64 S=2000", 1, 2000, 64, 8, 128),
    ("D128 H=64 S=2500", 1, 2500, 64, 8, 128),
    ("D128 H=64 S=3000", 1, 3000, 64, 8, 128),
    ("D128 H=64 S=4000", 1, 4000, 64, 8, 128),
    ("D128 H=64 S=8000", 1, 8000, 64, 8, 128),
    ("D128 H=32 S=1500", 1, 1500, 32, 4, 128),
    ("D128 H=32 S=2000", 1, 2000, 32, 4, 128),
    ("D128 H=32 S=4000", 1, 4000, 32, 4, 128),
    ("D64  H=64 S=199",  1, 199,  64, 8, 64),
    ("D64  H=64 S=512",  1, 512,  64, 8, 64),
    ("D64  H=64 S=1024", 1, 1024, 64, 8, 64),
    ("D64  H=64 S=1500", 1, 1500, 64, 8, 64),
    ("D64  H=64 S=1600", 1, 1600, 64, 8, 64),
    ("D64  H=64 S=1800", 1, 1800, 64, 8, 64),
    ("D64  H=64 S=2000", 1, 2000, 64, 8, 64),
    ("D64  H=64 S=3000", 1, 3000, 64, 8, 64),
    ("D64  H=64 S=4000", 1, 4000, 64, 8, 64),
    ("D64  H=32 S=1500", 1, 1500, 32, 4, 64),
    ("D64  H=32 S=2000", 1, 2000, 32, 4, 64),
    ("D64  H=32 S=3000", 1, 3000, 32, 4, 64),
]

print(f"{'shape':<22s}  {'Triton':>8s}  {'OLD(forced)':>11s} {'NEW(auto)':>10s}  {'delta':>8s}  {'gain%':>7s}  OLD/NEW cfg")
for label, B, S, H_q, H_kv, D in SHAPES:
    inp = mk(B, S, H_q, H_kv, D)
    old_cfg = _old_cfg_b1(S, H_q, D)
    new_cfg_full = _get_basic_dispatch_config(D, B, S, 0, False, -1, head_num=H_q)
    new_cfg = (new_cfg_full[0], new_cfg_full[2], new_cfg_full[3])
    t_triton = bench_gpu(lambda: call_triton(inp))
    t_old = bench_gpu(lambda: call_forced(inp, *old_cfg))
    t_new = bench_gpu(lambda: call_auto(inp))
    delta = t_new - t_old
    gain = (t_old - t_new) / t_old * 100 if t_old > 0 else 0
    flag = "" if abs(gain) < 1 else ("WIN" if gain > 0 else "REGR")
    print(f"{label:<22s}  {t_triton:>6.1f}us  {t_old:>9.1f}us {t_new:>8.1f}us  {delta:>+6.1f}us  {gain:>+6.2f}%  {str(old_cfg)} → {str(new_cfg)}  {flag}")
