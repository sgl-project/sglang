import torch

from sglang.kernels.ops.attention.dsv4.topk import top_k_per_row_prefill_bf16

TOPK = 2048


def run_case(name: str, B: int, K: int, starts, lens):
    torch.manual_seed(0)
    scores = torch.randn(B, K, dtype=torch.bfloat16, device="cuda")
    row_starts = torch.tensor(starts, dtype=torch.int32, device="cuda")
    row_ends = row_starts + torch.tensor(lens, dtype=torch.int32, device="cuda")
    page_table = (
        torch.arange(K, dtype=torch.int32, device="cuda").repeat(B, 1).contiguous()
    )
    out = torch.full((B, TOPK), -1, dtype=torch.int32, device="cuda")

    top_k_per_row_prefill_bf16(scores, row_starts, row_ends, page_table, out, 1, None)
    torch.cuda.synchronize()

    for i in range(B):
        s, e = int(row_starts[i]), int(row_ends[i])
        n = e - s
        idx = out[i].long()
        idx = idx[idx >= 0]
        got = scores[i, s:e].float()[idx].sort(descending=True).values
        ref = (
            scores[i, s:e]
            .float()
            .topk(min(TOPK, n))
            .values.sort(descending=True)
            .values
        )
        assert torch.equal(got, ref), f"{name}: row {i} mismatch (n={n})"
    print(f"PASS {name}")


def main():
    # Stream kernel (score_stride > 32K). Odd row_starts -> misaligned row
    # bases: this is the in_64k_co_3 crash repro (pre-fix kernel faulted on
    # unaligned cp.async / uint4 accesses).
    run_case("stream_misaligned_odd_starts", 4, 40000, [1, 3, 5, 7], [39992] * 4)
    # Odd score_stride: row bases misaligned even with row_start == 0.
    run_case("stream_misaligned_odd_stride", 3, 40001, [0, 0, 0], [40001, 39999, 33000])
    # Aligned rows: the perf-critical fast path.
    run_case("stream_aligned", 3, 40000, [0, 0, 0], [40000, 32768, 32769])
    # Lengths crossing 8K stage / 256-elem tile boundaries + short row routed
    # to the naive path inside the stream kernel.
    run_case(
        "stream_boundary_lengths", 4, 40000, [0, 1, 2, 3], [8192, 8193, 16385, 2048]
    )
    # Register kernel (score_stride <= 32K): 2-pass misaligned src (exercises
    # the scalar-copy fallback), 2-pass aligned, 1-pass, and naive.
    run_case("register_2pass_misaligned", 2, 20000, [1, 3], [19998, 19998])
    run_case("register_2pass_aligned", 2, 20000, [0, 0], [20000, 16500])
    run_case("register_1pass_misaligned", 2, 10000, [1, 2], [9998, 5000])
    run_case("register_naive", 2, 1000, [1, 0], [999, 999])
    print("ALL PASS")


if __name__ == "__main__":
    main()
