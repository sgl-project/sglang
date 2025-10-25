import os
from collections import defaultdict

import matplotlib.pyplot as plt
import torch
from st_attn import sliding_tile_attention
from triton.testing import do_bench


def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)


def compute_TFLOPS(flops, ms):
    flops = flops / 1e12
    ms = ms / 1e3
    return flops / ms


def benchmark_attention(configurations):
    results = {"fwd": defaultdict(list), "bwd": defaultdict(list)}

    for B, H, N, D, causal, dit_seq_shape, window_size in configurations:
        print("=" * 60)
        print(
            f"Timing forward and backward pass for B={B}, H={H}, N={N}, D={D}, causal={causal}"
        )

        q = torch.randn(
            B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=False
        ).contiguous()
        k = torch.randn(
            B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=False
        ).contiguous()
        v = torch.randn(
            B, H, N, D, dtype=torch.bfloat16, device="cuda", requires_grad=False
        ).contiguous()

        # grad_output = torch.randn_like(q, requires_grad=False).contiguous()
        # qg = torch.zeros_like(q, requires_grad=False, dtype=torch.float).contiguous()
        # kg = torch.zeros_like(k, requires_grad=False, dtype=torch.float).contiguous()
        # vg = torch.zeros_like(v, requires_grad=False, dtype=torch.float).contiguous()

        # # Warmup for forward pass
        # for _ in range(10):
        #     o = sliding_tile_attention(q, k, v, [[3, 6, 10]] * 24, 0, False, dit_seq_shape)

        # # Time the forward pass
        # for i in range(10):
        #     start_events_fwd[i].record()
        #     o = sliding_tile_attention(q, k, v, [[3, 6, 10]] * 24, 0, False, dit_seq_shape)
        #     end_events_fwd[i].record()
        ms = do_bench(
            lambda: sliding_tile_attention(
                q, k, v, [window_size] * 24, 0, False, dit_seq_shape
            )
        )

        # times_fwd = [s.elapsed_time(e) for s, e in zip(start_events_fwd, end_events_fwd)]
        # time_us_fwd = np.mean(times_fwd) * 1000

        tflops_fwd = compute_TFLOPS(flops(B, N, H, D, causal, "fwd"), ms)
        results["fwd"][(D, causal)].append((N, tflops_fwd))

        print(f"Average time for forward pass (ms): {ms:.2f}")
        print(f"Average TFLOPS: {tflops_fwd}")
        print("-" * 60)

        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()

        # # Prepare for timing backward pass
        # start_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]
        # end_events_bwd = [torch.cuda.Event(enable_timing=True) for _ in range(10)]

        # # Warmup for backward pass
        # for _ in range(10):
        #     qg, kg, vg = tk.mha_backward(q, k, v, o, l_vec, grad_output, causal)

        # # Time the backward pass
        # for i in range(10):
        #     start_events_bwd[i].record()
        #     qg, kg, vg = tk.mha_backward(q, k, v, o, l_vec, grad_output, causal)
        #     end_events_bwd[i].record()

        # torch.cuda.synchronize()
        # times_bwd = [s.elapsed_time(e) for s, e in zip(start_events_bwd, end_events_bwd)]
        # time_us_bwd = np.mean(times_bwd) * 1000

        # tflops_bwd = compute_TFLOPS(flops(B, N, H, D, causal, 'bwd'), ms)
        # results['bwd'][(D, causal)].append((N, tflops_bwd))

        # print(f"Average time for backward pass(ms): {ms:.2f}")
        # print(f"Average TFLOPS: {tflops_bwd}")
        # print("=" * 60)

    return results


def plot_results(results):
    os.makedirs("benchmark_results", exist_ok=True)
    for mode in ["fwd", "bwd"]:
        for (D, causal), values in results[mode].items():
            seq_lens = [x[0] for x in values]
            tflops = [x[1] for x in values]

            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(seq_lens)), tflops, tick_label=seq_lens)
            plt.xlabel("Sequence Length")
            plt.ylabel("TFLOPS")
            plt.title(f"{mode.upper()} Pass - Head Dim: {D}, Causal: {causal}")
            plt.grid(True)

            # Adding the numerical y value on top of each bar
            for bar in bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval,
                    round(yval, 2),
                    ha="center",
                    va="bottom",
                )

            filename = f"benchmark_results/{mode}_D{D}_causal{causal}.png"
            plt.savefig(filename)
            plt.close()


# Example list of configurations to test
configurations = [
    (2, 24, 69120, 128, False, "18x48x80", [3, 6, 10]),
    (2, 24, 69120, 128, True, "18x48x80", [3, 6, 10]),
    (2, 24, 82944, 128, False, "36x48x48", [3, 3, 6]),  # Stepvideo
    (2, 24, 82944, 128, True, "36x48x48", [3, 3, 6]),
    # (16, 16, 768*16,  128, False),
    # (16, 16, 768*2,  128, False),
    # (16, 16, 768*4,  128, False),
    # (16, 16, 768*8,  128, False),
    # (16, 16, 768*16, 128, False),
    # (16, 16, 768,    128, True),
    # (16, 16, 768*2,  128, True),
    # (16, 16, 768*4,  128, True),
    # (16, 16, 768*8,  128, True),
    # (16, 16, 768*16, 128, True),
    # (16, 32, 768,    64,  False),
    # (16, 32, 768*2,  64,  False),
    # (16, 32, 768*4,  64,  False),
    # (16, 32, 768*8,  64,  False),
    # (16, 32, 768*16, 64,  False),
    # (16, 32, 768,    64,  True),
    # (16, 32, 768*2,  64,  True),
    # (16, 32, 768*4,  64,  True),
    # (16, 32, 768*8,  64,  True),
    # (16, 32, 768*16, 64,  True),
]

results = benchmark_attention(configurations)
# plot_results(results)
