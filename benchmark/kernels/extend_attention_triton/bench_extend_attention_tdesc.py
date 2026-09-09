"""Benchmark extend attention with host-side tensor descriptors on/off (Intel XPU).

``extend_attention_fwd`` describes the Q / extend-K / extend-V tiles with host-side
tensor descriptors when running on XPU, so the kernel issues 2D block I/O instead
of tensor-of-pointer loads. This bench forces the path on and off via
``SGLANG_USE_TRITON_ATTN_TENSOR_DESC`` and reports per-call kernel time plus
numerical agreement (which must be exact -- both paths read the same tiles in the
same order).

Shapes are given as ``(batch, prefix_len, extend_len, ...)`` to mirror what a
serving step actually issues: the first chunk of a prompt is ``prefix_len=0``,
later chunks carry the previous chunks as prefix, and decode-adjacent steps are a
tiny extend over a long prefix.

Timing is one synchronized call at a time (median), because that is how the
runtime invokes the kernel: once per layer, between other work.

Note these ratios understate the serving gain. In a running server the same
kernel with the same arguments costs 19.7 ms/call on pointers vs 4.22 ms/call on
descriptors for the ``serving 1st chunk`` shape -- 2.4x on end-to-end prefill --
which this harness does not reproduce.

Run on an XPU host:
    python benchmark/kernels/extend_attention_triton/bench_extend_attention_tdesc.py
"""

import argparse
import statistics
import time

import torch

from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd
from sglang.srt.environ import envs

# (batch, prefix_len, extend_len, H_Q, H_KV, D, label)
CONFIGS = [
    (1, 0, 2048, 32, 8, 128, "serving 1st chunk"),
    (1, 2048, 2048, 32, 8, 128, "serving later chunk"),
    (8, 0, 2048, 32, 8, 128, "batched prefill"),
    (4, 1024, 1024, 16, 16, 128, "MHA, mixed"),
    (2, 0, 8192, 32, 8, 128, "long cold prefill"),
    (4, 1024, 1024, 16, 16, 64, "small head dim"),
]


def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_inputs(B, prefix_len, extend_len, H_Q, H_KV, D, dtype, device, seed=0):
    """Build one extend-attention step with fixed prefix / extend lengths.

    ``k_buffer`` / ``v_buffer`` are sized like a real KV pool rather than tightly
    around this step, so strides and gather offsets match production.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(shape, heads, head_dim):
        return torch.empty(
            (shape, heads, head_dim), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2, generator=gen)

    pool_tokens = max(B * (prefix_len + extend_len), 43_008)
    k_buffer = randn(pool_tokens, H_KV, D)
    v_buffer = randn(pool_tokens, H_KV, D)

    b_prefix = torch.full((B,), prefix_len, dtype=torch.int32, device=device)
    b_extend = torch.full((B,), extend_len, dtype=torch.int32, device=device)
    seq_start = torch.zeros((B,), dtype=torch.int32, device=device)
    seq_start[1:] = torch.cumsum((b_prefix + b_extend)[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(b_prefix, 0)
    # Empty when there is no prefix, matching what a server's first chunk passes.
    kv_indices = torch.zeros((int(b_prefix.sum()),), dtype=torch.int32, device=device)
    for i in range(B):
        if prefix_len:
            kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
                seq_start[i], seq_start[i] + prefix_len, device=device
            )

    extend_tokens = B * extend_len
    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1:] = torch.cumsum(b_extend, 0)

    return dict(
        q_extend=randn(extend_tokens, H_Q, D),
        k_extend=randn(extend_tokens, H_KV, D),
        v_extend=randn(extend_tokens, H_KV, D),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        max_len_extend=extend_len,
        extend_tokens=extend_tokens,
        H_Q=H_Q,
        D=D,
        dtype=dtype,
        device=device,
    )


def _run_once(inp, use_desc: bool, out=None):
    """Run extend_attention_fwd with the descriptor path forced on/off via the
    SGLANG_USE_TRITON_ATTN_TENSOR_DESC tri-state override."""
    o = (
        out
        if out is not None
        else torch.empty(
            (inp["extend_tokens"], inp["H_Q"], inp["D"]),
            dtype=inp["dtype"],
            device=inp["device"],
        )
    )
    with envs.SGLANG_USE_TRITON_ATTN_TENSOR_DESC.override(use_desc):
        extend_attention_fwd(
            inp["q_extend"],
            inp["k_extend"],
            inp["v_extend"],
            o,
            inp["k_buffer"],
            inp["v_buffer"],
            inp["qo_indptr"],
            inp["kv_indptr"],
            inp["kv_indices"],
            None,  # custom_mask
            True,  # is_causal
            None,  # mask_indptr
            inp["max_len_extend"],
            1.0,  # k_scale
            1.0,  # v_scale
        )
    return o


def _time_isolated(inp, use_desc: bool, repeat: int, device: str) -> float:
    """Median of single synchronized calls -- one call per launch, like serving."""
    out = torch.empty(
        (inp["extend_tokens"], inp["H_Q"], inp["D"]),
        dtype=inp["dtype"],
        device=inp["device"],
    )
    sync = torch.xpu.synchronize if device == "xpu" else torch.cuda.synchronize
    latencies = []
    for _ in range(repeat):
        sync()
        start = time.perf_counter()
        _run_once(inp, use_desc, out=out)
        sync()
        latencies.append((time.perf_counter() - start) * 1e3)
    return statistics.median(latencies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repeat", type=int, default=7, help="timed calls per variant (median)"
    )
    args = parser.parse_args()

    device = get_device()
    dtype = torch.bfloat16
    print(f"device={device} dtype={dtype}")
    if device != "xpu":
        print(
            "WARNING: descriptors are only auto-enabled on XPU; elsewhere this just "
            "exercises correctness of the toggle."
        )

    header = (
        f"{'config':<20}{'B':>2} {'prefix':>7} {'extend':>7} {'H_Q':>4} {'H_KV':>5} "
        f"{'D':>4} {'off ms':>9} {'on ms':>8} {'speedup':>8} {'max err':>9}"
    )
    print(header)
    print("-" * len(header))

    for B, prefix_len, extend_len, H_Q, H_KV, D, label in CONFIGS:
        inp = _build_inputs(B, prefix_len, extend_len, H_Q, H_KV, D, dtype, device)

        o_off = _run_once(inp, use_desc=False)
        o_on = _run_once(inp, use_desc=True)
        max_err = (o_on.float() - o_off.float()).abs().max().item()
        # Both paths read the same tiles in the same order, so equality is exact;
        # any difference means a descriptor read the wrong head or rows.
        flag = "OK" if max_err == 0.0 else "MISMATCH"

        t_off = _time_isolated(inp, False, args.repeat, device)
        t_on = _time_isolated(inp, True, args.repeat, device)

        print(
            f"{label:<20}{B:>2} {prefix_len:>7} {extend_len:>7} {H_Q:>4} {H_KV:>5} "
            f"{D:>4} {t_off:>9.2f} {t_on:>8.2f} {t_off / t_on:>7.2f}x "
            f"{max_err:>9.1e} {flag}"
        )

    print("\nms are per call.")


if __name__ == "__main__":
    main()
