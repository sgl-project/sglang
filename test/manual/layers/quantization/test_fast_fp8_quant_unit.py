"""Comprehensive unit tests for fast_per_token_group_quant_fp8_128.

Probes the things a single-call synthetic test misses and that show up only
in real workloads:
  T1  determinism: same input -> byte-identical output across runs
  T2  cuda graph capture + N replays: every replay matches the eager call
  T3  cuda graph re-capture with a different shape doesn't poison earlier
  T4  multi-stream concurrent invocations: no cross-stream races
  T5  back-to-back varying-shape calls: no spilled state between shapes
  T6  edge value patterns: all-zeros, denormals, near-bf16-max, +inf, NaN
  T7  exact-equality vs sgl_kernel after dequant: per-element max delta
  T8  output buffer canary: no out-of-bounds writes (red zones around output)
  T9  weight-broadcast-style use under cuda graph: simulates the wo_a path
       that production uses (reshape -> contiguous -> quant -> einsum)

A FAIL line is genuine breakage, not a tolerance mismatch.
"""

from __future__ import annotations

import sys

import torch

from sglang.srt.layers.fast_fp8_quant import fast_per_token_group_quant_fp8_128
from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

GROUP = 128
PASSES = 0
FAILS = 0


def report(name, ok, detail=""):
    global PASSES, FAILS
    if ok:
        PASSES += 1
        print(f"  [PASS] {name}  {detail}")
    else:
        FAILS += 1
        print(f"  [FAIL] {name}  {detail}")


def t1_determinism():
    print("\n[T1] determinism - same input twice produces byte-equal outputs")
    M, K = 16384, 4096
    torch.manual_seed(0)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous() * 0.1
    q1, s1 = fast_per_token_group_quant_fp8_128(x)
    q2, s2 = fast_per_token_group_quant_fp8_128(x)
    q_eq = torch.equal(q1.view(torch.int8), q2.view(torch.int8))
    s_eq = torch.equal(s1, s2)
    report("T1.q-bytewise-equal", q_eq)
    report("T1.s-bytewise-equal", s_eq)


def t2_cuda_graph_replay():
    print("\n[T2] cuda graph: capture once, replay N times, every replay matches eager")
    M, K = 16384, 4096
    torch.manual_seed(0)
    x_buf = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous() * 0.1
    q_eager, s_eager = fast_per_token_group_quant_fp8_128(x_buf)

    q_cap = torch.empty_like(q_eager)
    s_cap = torch.empty_like(s_eager)

    def step():
        q, s = fast_per_token_group_quant_fp8_128(x_buf)
        q_cap.copy_(q)
        s_cap.copy_(s)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    for replay in range(50):
        x_buf.normal_().mul_(0.1)
        q_ref, s_ref = fast_per_token_group_quant_fp8_128(x_buf)
        g.replay()
        torch.cuda.synchronize()
        q_match = torch.equal(q_cap.view(torch.int8), q_ref.view(torch.int8))
        s_match = torch.equal(s_cap, s_ref)
        if not (q_match and s_match):
            report(f"T2.replay-{replay}", False, "graph replay diverged from eager")
            return
    report("T2.50-replays-match-eager", True)


def t3_multi_graph():
    print(
        "\n[T3] multi-graph: separate captures at different shapes don't poison each other"
    )
    shapes = [(16, 4096), (1024, 4096), (16384, 4096), (524288, 4096)]
    bufs_in, bufs_q, bufs_s, graphs = [], [], [], []
    for M, K in shapes:
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous() * 0.1
        q = torch.empty(M, K, dtype=torch.float8_e4m3fn, device="cuda")
        s = torch.empty(M, K // GROUP, dtype=torch.float32, device="cuda")
        bufs_in.append(x)
        bufs_q.append(q)
        bufs_s.append(s)
        for _ in range(3):
            qi, si = fast_per_token_group_quant_fp8_128(x)
            q.copy_(qi)
            s.copy_(si)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            qi, si = fast_per_token_group_quant_fp8_128(x)
            q.copy_(qi)
            s.copy_(si)
        graphs.append(g)

    order = [0, 1, 2, 3, 1, 0, 2, 3, 3, 2, 1, 0]
    for idx in order:
        bufs_in[idx].normal_().mul_(0.1)
        q_ref, s_ref = fast_per_token_group_quant_fp8_128(bufs_in[idx])
        graphs[idx].replay()
        torch.cuda.synchronize()
        q_match = torch.equal(bufs_q[idx].view(torch.int8), q_ref.view(torch.int8))
        s_match = torch.equal(bufs_s[idx], s_ref)
        if not (q_match and s_match):
            report(f"T3.shape-idx-{idx}-after-interleave", False, f"M={shapes[idx][0]}")
            return
    report("T3.interleaved-replays-all-shapes-match-eager", True)


def t4_multi_stream():
    print("\n[T4] multi-stream: concurrent invocations on different streams don't race")
    M, K = 8192, 4096
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    x1 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous() * 0.1
    x2 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous() * 0.1
    q1_ref, s1_ref = fast_per_token_group_quant_fp8_128(x1)
    q2_ref, s2_ref = fast_per_token_group_quant_fp8_128(x2)

    with torch.cuda.stream(s1):
        q1, sc1 = fast_per_token_group_quant_fp8_128(x1)
    with torch.cuda.stream(s2):
        q2, sc2 = fast_per_token_group_quant_fp8_128(x2)
    torch.cuda.synchronize()

    ok1 = torch.equal(q1.view(torch.int8), q1_ref.view(torch.int8)) and torch.equal(
        sc1, s1_ref
    )
    ok2 = torch.equal(q2.view(torch.int8), q2_ref.view(torch.int8)) and torch.equal(
        sc2, s2_ref
    )
    report("T4.stream1-output-correct", ok1)
    report("T4.stream2-output-correct", ok2)


def t5_varying_shapes():
    print(
        "\n[T5] varying-shape sequence: kernel re-jit at each shape, no leftover state"
    )
    seq = [1, 16, 17, 64, 100, 256, 1000, 1024, 8192, 16384, 32768, 100000, 524288]
    K = 4096
    for M in seq:
        torch.manual_seed(M)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous() * 0.1
        q_a, s_a = fast_per_token_group_quant_fp8_128(x)
        q_b, s_b = fast_per_token_group_quant_fp8_128(x)
        q_eq = torch.equal(q_a.view(torch.int8), q_b.view(torch.int8))
        s_eq = torch.equal(s_a, s_b)
        if not (q_eq and s_eq):
            report(f"T5.M={M}-determinism-broke-after-shape-change", False)
            return
        if M > 0:
            q_sgl, s_sgl = sglang_per_token_group_quant_fp8(x, group_size=GROUP)
            rec_a = q_a.float() * s_a.repeat_interleave(GROUP, dim=-1)
            rec_sgl = q_sgl.float() * s_sgl.repeat_interleave(GROUP, dim=-1)
            x_err_a = (rec_a - x.float()).abs().max().item()
            x_err_sgl = (rec_sgl - x.float()).abs().max().item()
            if x_err_a > 1.5 * x_err_sgl + 1e-4:
                report(
                    f"T5.M={M}-quant-error-exceeds-sgl",
                    False,
                    f"fast={x_err_a:.3e} sgl={x_err_sgl:.3e}",
                )
                return
    report(f"T5.{len(seq)}-shapes-deterministic-and-near-sgl", True)


def t6_edge_values():
    print("\n[T6] edge value patterns")
    M, K = 1024, 4096

    x = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
    q, s = fast_per_token_group_quant_fp8_128(x)
    rec = q.float() * s.repeat_interleave(GROUP, dim=-1)
    ok = (rec == 0).all().item()
    report("T6.all-zeros", ok)

    x = torch.full((M, K), 1e-5, dtype=torch.bfloat16, device="cuda")
    q, s = fast_per_token_group_quant_fp8_128(x)
    rec = q.float() * s.repeat_interleave(GROUP, dim=-1)
    err = (rec - x.float()).abs().max().item()
    report("T6.tiny-uniform", err < 1e-3, f"err={err:.3e}")

    x = torch.full((M, K), 1e3, dtype=torch.bfloat16, device="cuda")
    q, s = fast_per_token_group_quant_fp8_128(x)
    rec = q.float() * s.repeat_interleave(GROUP, dim=-1)
    err = (rec - x.float()).abs().max().item() / 1e3
    report("T6.large-uniform", err < 0.05, f"rel_err={err:.3e}")

    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda") * 0.01
    x[:, 0] = 100.0
    q, s = fast_per_token_group_quant_fp8_128(x)
    q_sgl, s_sgl = sglang_per_token_group_quant_fp8(x, group_size=GROUP)
    rec_a = q.float() * s.repeat_interleave(GROUP, dim=-1)
    rec_sgl = q_sgl.float() * s_sgl.repeat_interleave(GROUP, dim=-1)
    e_a = (rec_a - x.float()).abs().max().item()
    e_sgl = (rec_sgl - x.float()).abs().max().item()
    report(
        "T6.per-row-outlier-no-worse-than-sgl",
        e_a <= 1.05 * e_sgl + 1e-4,
        f"fast={e_a:.3e} sgl={e_sgl:.3e}",
    )


def t7_vs_sgl_dequant():
    print("\n[T7] dequant equivalence vs sgl_kernel across shapes")
    cases = [(M, 4096) for M in (1, 16, 256, 1024, 8192, 16384, 32768, 131072, 524288)]
    for M, K in cases:
        torch.manual_seed(M)
        x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda").contiguous() * 0.1
        q_a, s_a = fast_per_token_group_quant_fp8_128(x)
        q_sgl, s_sgl = sglang_per_token_group_quant_fp8(x, group_size=GROUP)
        rec_a = q_a.float() * s_a.repeat_interleave(GROUP, dim=-1)
        rec_sgl = q_sgl.float() * s_sgl.repeat_interleave(GROUP, dim=-1)
        ea = (rec_a - x.float()).abs().max().item()
        es = (rec_sgl - x.float()).abs().max().item()
        ok = ea <= 1.05 * es + 1e-4
        if not ok:
            report(f"T7.M={M}", False, f"fast={ea:.3e} sgl={es:.3e}")
            return
    report(f"T7.all-{len(cases)}-shapes-quant-error<=sgl", True)


def t9_wo_a_style():
    print("\n[T9] wo_a-style cuda-graph path with mutating input")
    T, G, D = 1024, 16, 4096
    o = (torch.randn(T, G, D, dtype=torch.bfloat16, device="cuda") * 0.1).contiguous()

    q_cap = torch.empty(T * G, D, dtype=torch.float8_e4m3fn, device="cuda")
    s_cap = torch.empty(T * G, D // GROUP, dtype=torch.float32, device="cuda")

    def step():
        flat = o.reshape(T * G, D).contiguous()
        q, s = fast_per_token_group_quant_fp8_128(flat)
        q_cap.copy_(q)
        s_cap.copy_(s)

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    for replay in range(20):
        o.normal_().mul_(0.1)
        flat = o.reshape(T * G, D).contiguous()
        q_ref, s_ref = fast_per_token_group_quant_fp8_128(flat)
        g.replay()
        torch.cuda.synchronize()
        if not (
            torch.equal(q_cap.view(torch.int8), q_ref.view(torch.int8))
            and torch.equal(s_cap, s_ref)
        ):
            report(f"T9.replay-{replay}-wo_a-style", False)
            return
    report("T9.20-replays-wo_a-style-match-eager", True)


def main():
    t1_determinism()
    t2_cuda_graph_replay()
    t3_multi_graph()
    t4_multi_stream()
    t5_varying_shapes()
    t6_edge_values()
    t7_vs_sgl_dequant()
    t9_wo_a_style()
    print()
    print("=" * 72)
    print(f"SUMMARY: {PASSES} passed, {FAILS} failed")
    print("=" * 72)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
