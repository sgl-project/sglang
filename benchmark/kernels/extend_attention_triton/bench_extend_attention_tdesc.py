"""Benchmark extend attention with device-side tensor descriptors on/off (Intel XPU).

`_fwd_kernel` in extend_attention.py builds device-side tensor descriptors for
the Q / extend-K / extend-V / O tiles when ``MAKE_DEVICE_DESC`` is set, which the
launcher wires to ``_is_xpu``. This bench toggles that module-level flag to
compare the descriptor ("TMA on") path against the legacy tensor-of-pointer
("TMA off") path -- both for numerical agreement and for kernel time.

Run on an XPU host:
    python benchmark/kernels/extend_attention_triton/bench_extend_attention_tdesc.py
"""

import argparse

import torch
import triton.testing as tt

import sglang.srt.layers.attention.triton_ops.extend_attention as ea
from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd


def get_device():
    if torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _build_inputs(B, N_CTX, H_Q, H_KV, D, dtype, device, seed=0):
    """Mirror the input construction in test_triton_attention_kernels.py."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def randint(low, high, shape):
        return (
            torch.randint(low, high, shape, generator=gen, device=device)
            .to(torch.int32)
        )

    b_seq_len_prefix = randint(1, N_CTX // 2, (B,))
    b_seq_len_extend = randint(1, N_CTX // 2, (B,))
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = torch.zeros(
        (b_seq_len_prefix.sum().item(),), dtype=torch.int32, device=device
    )
    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i], device=device
        )

    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device
    ).normal_(mean=0.1, std=0.2, generator=gen)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device
    ).normal_(mean=0.1, std=0.2, generator=gen)

    k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    v_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype, device=device)
    q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = torch.empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2, generator=gen)

    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

    return dict(
        q_extend=q_extend,
        k_extend=k_extend,
        v_extend=v_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        max_len_extend=max_len_extend,
        extend_token_num=extend_token_num,
        H_Q=H_Q,
        D=D,
        dtype=dtype,
        device=device,
    )


def _run_once(inp, device_desc: bool):
    """Run extend_attention_fwd with the device-descriptor path forced on/off."""
    o = torch.empty(
        (inp["extend_token_num"], inp["H_Q"], inp["D"]),
        dtype=inp["dtype"],
        device=inp["device"],
    )
    saved = ea._is_xpu
    ea._is_xpu = device_desc  # launcher reads this for MAKE_DEVICE_DESC=_is_xpu
    try:
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
    finally:
        ea._is_xpu = saved
    return o


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-3)
    args = parser.parse_args()

    device = get_device()
    dtype = torch.bfloat16
    print(f"device={device} dtype={dtype}")
    if device != "xpu":
        print(
            "WARNING: device-side descriptors only take the XPU fast path on XPU; "
            "results elsewhere just exercise correctness of the toggle."
        )

    # (B, N_CTX, H_Q, H_KV, D)
    configs = [
        (12, 2048, 16, 16, 128),  # MHA
        (12, 2048, 28, 4, 128),  # GQA
        (8, 4096, 32, 8, 128),  # GQA, longer ctx
        (4, 8192, 16, 16, 64),  # MHA, small head dim
    ]

    header = (
        f"{'B':>3} {'N_CTX':>6} {'H_Q':>4} {'H_KV':>4} {'D':>4} "
        f"{'off (ms)':>10} {'on (ms)':>10} {'speedup':>8} {'max_abs_err':>12}"
    )
    print(header)
    print("-" * len(header))

    for B, N_CTX, H_Q, H_KV, D in configs:
        inp = _build_inputs(B, N_CTX, H_Q, H_KV, D, dtype, device)

        o_off = _run_once(inp, device_desc=False)
        o_on = _run_once(inp, device_desc=True)

        max_err = (o_on.float() - o_off.float()).abs().max().item()
        ok = torch.allclose(o_on, o_off, rtol=args.rtol, atol=args.atol)

        t_off = tt.do_bench(lambda: _run_once(inp, device_desc=False))
        t_on = tt.do_bench(lambda: _run_once(inp, device_desc=True))

        flag = "OK" if ok else "MISMATCH"
        print(
            f"{B:>3} {N_CTX:>6} {H_Q:>4} {H_KV:>4} {D:>4} "
            f"{t_off:>10.4f} {t_on:>10.4f} {t_off / t_on:>7.2f}x "
            f"{max_err:>12.2e} {flag}"
        )


if __name__ == "__main__":
    main()
