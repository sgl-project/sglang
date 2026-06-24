"""Operator-level benchmark: seg_la kv vs vk state layout.

Compares ``seg_la_fwd(state_layout="kv")`` vs ``"vk"`` for PREFILL and
DECODE paths on Ling-2.6-flash dims (H=HV=8, D=128).  Sweeps (B, S)
configs and prints a per-path, per-config timing table.

Columns:  path, B, S, H, D, layout, ms, ratio
"""

import argparse

import torch

from sglang.srt.layers.attention.linear.seg_la import SegLaMeta, seg_la_fwd


def _bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/iter


def make_inputs(B, S, H, D):
    total = B * S
    q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16)
    decay = (0.3 * torch.arange(1, H + 1, device="cuda", dtype=torch.float32) / H).view(
        H, 1, 1
    )
    return q, k, v, decay


def make_closure(B, S, H, D, q, k, v, decay, scale, layout):
    s = torch.zeros(B, H, D, D, device="cuda", dtype=torch.float32)
    if layout == "vk":
        s = s.transpose(-1, -2).contiguous()
    q_offsets = torch.arange(0, B * S + 1, S, device="cuda", dtype=torch.int32)
    meta = SegLaMeta(
        batch_size=B,
        max_q_length=S,
        q_offsets=q_offsets,
        s_offsets=torch.arange(B, device="cuda", dtype=torch.int32),
        q_lengths=torch.full((B,), S, device="cuda", dtype=torch.int32),
        s_scales=torch.zeros(B, device="cuda", dtype=torch.int32),
    )

    def run():
        seg_la_fwd(q, k, v, s, decay, meta, softmax_scale=scale, state_layout=layout)

    return run


def make_decode_closure(B, H, D, q, k, v, decay, scale, layout):
    s = torch.randn(B, H, D, D, device="cuda", dtype=torch.float32)
    if layout == "vk":
        s = s.transpose(-1, -2).contiguous()
    q_offsets = torch.arange(0, B + 1, 1, device="cuda", dtype=torch.int32)
    meta = SegLaMeta(
        batch_size=B,
        max_q_length=1,
        q_offsets=q_offsets,
        s_offsets=torch.arange(B, device="cuda", dtype=torch.int32),
        q_lengths=torch.ones(B, device="cuda", dtype=torch.int32),
        s_scales=torch.ones(B, device="cuda", dtype=torch.int32),
    )

    def run():
        seg_la_fwd(q, k, v, s, decay, meta, softmax_scale=scale, state_layout=layout)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    H, D = args.H, args.D
    scale = D**-0.5

    header = f"{'path':>8} {'B':>4} {'S':>6} {'tokens':>8} {'H':>3} {'D':>4} {'layout':>5} {'ms':>11} {'ratio':>8}"
    print(
        f"seg_la state_layout bench | H={H} D={D} | "
        f"{torch.cuda.get_device_name(0)} "
        f"sm_{torch.cuda.get_device_properties(0).major}"
        f"{torch.cuda.get_device_properties(0).minor}"
    )
    print(header)
    print("-" * len(header))

    # ------- PREFILL -------
    for B, S in [
        (1, 256),
        (1, 1024),
        (1, 4096),
        (4, 512),
        (8, 1024),
        (16, 512),
        (1, 8192),
    ]:
        torch.manual_seed(0)
        q, k, v, decay = make_inputs(B, S, H, D)
        t_kv = _bench(
            make_closure(B, S, H, D, q, k, v, decay, scale, "kv"), iters=args.iters
        )
        t_vk = _bench(
            make_closure(B, S, H, D, q, k, v, decay, scale, "vk"), iters=args.iters
        )
        ratio = t_vk / t_kv if t_kv > 0 else float("nan")
        print(
            f"{'PREFILL':>8} {B:>4} {S:>6} {B * S:>8} {H:>3} {D:>4} {'kv':>5} {t_kv:>11.4f} {'--':>8}"
        )
        print(
            f"{'PREFILL':>8} {B:>4} {S:>6} {B * S:>8} {H:>3} {D:>4} {'vk':>5} {t_vk:>11.4f} {ratio:>7.2f}x"
        )

    # ------- DECODE -------
    for B in [1, 32, 128, 256]:
        torch.manual_seed(1)
        q, k, v, decay = make_inputs(B, 1, H, D)
        t_kv = _bench(
            make_decode_closure(B, H, D, q, k, v, decay, scale, "kv"), iters=args.iters
        )
        t_vk = _bench(
            make_decode_closure(B, H, D, q, k, v, decay, scale, "vk"), iters=args.iters
        )
        ratio = t_vk / t_kv if t_kv > 0 else float("nan")
        print(
            f"{'DECODE':>8} {B:>4} {1:>6} {B:>8} {H:>3} {D:>4} {'kv':>5} {t_kv:>11.4f} {'--':>8}"
        )
        print(
            f"{'DECODE':>8} {B:>4} {1:>6} {B:>8} {H:>3} {D:>4} {'vk':>5} {t_vk:>11.4f} {ratio:>7.2f}x"
        )


if __name__ == "__main__":
    main()
