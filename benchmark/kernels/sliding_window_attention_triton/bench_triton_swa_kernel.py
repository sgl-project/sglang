import itertools

import torch
import torch.nn.functional as F
import triton.testing as tt

from sglang.srt.layers.attention.triton_ops.extend_attention import extend_attention_fwd


def extend_attention_fwd_torch(
    q: torch.Tensor,  # [extend_tokens, H_Q, D]
    k: torch.Tensor,  # [extend_tokens, H_KV, D]
    v: torch.Tensor,  # [extend_tokens, H_KV, D]
    o: torch.Tensor,  # [extend_tokens, H_Q, D]
    k_cache: torch.Tensor,  # [total_tokens, H_KV, D]
    v_cache: torch.Tensor,  # [total_tokens, H_KV, D]
    qo_indptr: torch.Tensor,  # [B+1]
    kv_indptr: torch.Tensor,  # [B+1]
    kv_indices: torch.Tensor,  # [prefix_tokens]
    sliding_window_size: int,
):
    B = qo_indptr.size(0) - 1
    _, H_Q, D = q.shape
    _, H_KV, _ = k.shape

    group_size = H_Q // H_KV
    scale = 1.0 / D**0.5

    for i in range(B):
        q_start = int(qo_indptr[i].item())
        q_end = int(qo_indptr[i + 1].item())
        kv_start = int(kv_indptr[i].item())
        kv_end = int(kv_indptr[i + 1].item())

        prefix_indices = kv_indices[kv_start:kv_end]
        k_prefix = k_cache[prefix_indices]  # [prefix_len, H_KV, D]
        v_prefix = v_cache[prefix_indices]  # [prefix_len, H_KV, D]

        k_extend = k[q_start:q_end]  # [extend_len, H_KV, D]
        v_extend = v[q_start:q_end]  # [extend_len, H_KV, D]
        q_extend = q[q_start:q_end]  # [extend_len, H_Q,  D]

        k_full = torch.cat([k_prefix, k_extend], dim=0)  # [total_len, H_KV, D]
        v_full = torch.cat([v_prefix, v_extend], dim=0)  # [total_len, H_KV, D]

        if group_size != 1:
            k_full_hq = k_full.repeat_interleave(
                group_size, dim=1
            )  # [total_len, H_Q, D]
            v_full_hq = v_full.repeat_interleave(
                group_size, dim=1
            )  # [total_len, H_Q, D]
        else:
            k_full_hq = k_full
            v_full_hq = v_full

        prefix_len = k_prefix.size(0)
        extend_len = k_extend.size(0)
        total_len = prefix_len + extend_len

        # causal
        pos_keys = torch.arange(total_len, device=q.device)
        t = prefix_len + torch.arange(extend_len, device=q.device)  # [extend_len]
        causal_mask = pos_keys.unsqueeze(0) <= t.unsqueeze(1)

        # sliding window
        if sliding_window_size is not None and sliding_window_size > 0:
            start = (t - (sliding_window_size)).clamp_min(0)  # [extend_len]
        else:
            start = torch.zeros_like(t)
        window_mask = pos_keys.unsqueeze(0) >= start.unsqueeze(1)

        final_mask = causal_mask & window_mask

        attn_scores = (
            torch.einsum("qhd,khd->qhk", q_extend, k_full_hq) * scale
        )  # [extend_len, H_Q, total_len]
        attn_scores = attn_scores.masked_fill(~final_mask.unsqueeze(1), float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        o[q_start:q_end] = torch.einsum("qhk,khd->qhd", attn_weights, v_full_hq)


def _build_batch(
    B, N_CTX, H_Q, H_KV, D, WINDOW_SIZE, dtype=torch.bfloat16, device="cuda"
):
    b_seq_len_prefix = torch.randint(
        1, max(2, N_CTX // 2), (B,), dtype=torch.int32, device=device
    )
    b_seq_len_extend = torch.randint(
        1, max(2, N_CTX // 2), (B,), dtype=torch.int32, device=device
    )
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_start_loc = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = torch.zeros((B,), dtype=torch.int32, device=device)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)

    kv_indices = torch.zeros(
        (int(b_seq_len_prefix.sum().item()),), dtype=torch.int32, device=device
    )
    for i in range(B):
        s = kv_indptr[i].item()
        e = kv_indptr[i + 1].item()
        kv_indices[s:e] = torch.arange(
            b_start_loc[i],
            b_start_loc[i] + b_seq_len_prefix[i],
            dtype=torch.int32,
            device=device,
        )

    total_token_num = int(torch.sum(b_seq_len).item())
    extend_token_num = int(torch.sum(b_seq_len_extend).item())

    k_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device
    ).normal_(mean=0.1, std=0.2)
    v_buffer = torch.empty(
        (total_token_num, H_KV, D), dtype=dtype, device=device
    ).normal_(mean=0.1, std=0.2)

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
            (int(b_seq_len_extend[i].item()), H_Q, D), dtype=dtype, device=device
        ).normal_(mean=0.1, std=0.2)

    o_extend_triton = torch.empty(
        (extend_token_num, H_Q, D), dtype=dtype, device=device
    )
    o_extend_torch = torch.empty((extend_token_num, H_Q, D), dtype=dtype, device=device)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    max_len_extend = int(torch.max(b_seq_len_extend, 0)[0].item())
    qo_indptr = torch.zeros((B + 1,), dtype=torch.int32, device=device)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)

    inputs = dict(
        q_extend=q_extend,
        k_extend=k_extend,
        v_extend=v_extend,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        o_extend_triton=o_extend_triton,
        o_extend_torch=o_extend_torch,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        max_len_extend=max_len_extend,
        WINDOW_SIZE=WINDOW_SIZE,
    )
    meta = dict(
        B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D, extend_token_num=extend_token_num
    )
    return inputs, meta


def _run_triton(inputs):
    extend_attention_fwd(
        inputs["q_extend"],
        inputs["k_extend"],
        inputs["v_extend"],
        inputs["o_extend_triton"],
        inputs["k_buffer"],
        inputs["v_buffer"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        custom_mask=None,
        is_causal=True,
        mask_indptr=None,
        max_len_extend=inputs["max_len_extend"],
        sliding_window_size=inputs["WINDOW_SIZE"],
    )


def _run_torch_ref(inputs):
    extend_attention_fwd_torch(
        inputs["q_extend"],
        inputs["k_extend"],
        inputs["v_extend"],
        inputs["o_extend_torch"],
        inputs["k_buffer"],
        inputs["v_buffer"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_indices"],
        inputs["WINDOW_SIZE"],
    )


N_CTXS = [1024, 2048, 4096, 8192]
WINDOW_SIZES = [-1, 127, 256, 512]

CONFIGS = list(itertools.product(N_CTXS, WINDOW_SIZES))

PROVIDERS = ["torch", "triton"]


@tt.perf_report(
    tt.Benchmark(
        x_names=["N_CTX", "WINDOW_SIZE"],
        x_vals=CONFIGS,
        line_arg="provider",
        line_vals=PROVIDERS,
        line_names=PROVIDERS,
        ylabel="Runtime (ms)",
        plot_name="extend_attention_triton_vs_torch",
        args={
            "B": 32,
            "H_Q": 64,
            "H_KV": 8,
            "D": 128,
            "dtype": "bf16",
            "device": "cuda",
            "check_correctness": False,
            "warmup": 25,
            "rep": 100,
        },
    )
)
def bench(
    N_CTX,
    provider,
    B,
    H_Q,
    H_KV,
    D,
    dtype,
    device,
    WINDOW_SIZE,
    check_correctness,
    warmup,
    rep,
):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dt = dtype_map[dtype]

    inputs, _ = _build_batch(
        B, N_CTX, H_Q, H_KV, D, WINDOW_SIZE, dtype=dt, device=device
    )

    if check_correctness and provider == "triton":
        _run_triton(inputs)
        _run_torch_ref(inputs)
        torch.cuda.synchronize()
        if not torch.allclose(
            inputs["o_extend_triton"], inputs["o_extend_torch"], rtol=1e-3, atol=1e-3
        ):
            raise AssertionError("Mismatch between triton and torch reference.")

    if provider == "triton":
        ms = tt.do_bench(lambda: _run_triton(inputs), warmup=warmup, rep=rep)
    elif provider == "torch":
        ms = tt.do_bench(lambda: _run_torch_ref(inputs), warmup=warmup, rep=rep)
    else:
        raise ValueError(provider)

    return ms


if __name__ == "__main__":
    bench.run(print_data=True, show_plots=False)
