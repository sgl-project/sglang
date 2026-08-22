import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.attention.fixup_zero_kv import fixup_zero_kv_rows
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def _make_out(total_tokens: int, num_heads: int, v_head_dim: int, aligned: bool):
    shape = (total_tokens, num_heads, v_head_dim)
    if aligned:
        return torch.empty(shape, dtype=torch.float16, device="cuda")

    base = torch.empty(
        total_tokens * num_heads * v_head_dim + 1,
        dtype=torch.float16,
        device="cuda",
    )
    return base[1:].view(shape)


def _make_lse(total_tokens: int, num_heads: int, aligned: bool):
    shape = (total_tokens, num_heads)
    if aligned:
        return torch.empty(shape, dtype=torch.float32, device="cuda")

    base = torch.empty(total_tokens * num_heads + 1, dtype=torch.float32, device="cuda")
    return base[1:].view(shape)


def _build(max_seq_len: int, batch_size: int, mode: str):
    out_aligned, lse_aligned = {
        "vec_vec": (True, True),
        "vec_scalar": (True, False),
        "scalar_vec": (False, True),
        "scalar_scalar": (False, False),
    }[mode]

    num_heads = 4 if lse_aligned else 3
    v_head_dim = 128
    total_tokens = batch_size * max_seq_len

    out = _make_out(total_tokens, num_heads, v_head_dim, out_aligned)
    lse = _make_lse(total_tokens, num_heads, lse_aligned)
    kv_lens = torch.zeros(batch_size, dtype=torch.int32, device="cuda")
    cum_seq_lens = torch.arange(
        0,
        total_tokens + 1,
        max_seq_len,
        dtype=torch.int32,
        device="cuda",
    )
    return out, lse, kv_lens, cum_seq_lens


@marker.parametrize("max_seq_len,batch_size", [(32, 32), (128, 32), (512, 16)])
@marker.benchmark("mode", ["vec_vec", "vec_scalar", "scalar_vec", "scalar_scalar"])
def benchmark(max_seq_len: int, batch_size: int, mode: str):
    out, lse, kv_lens, cum_seq_lens = _build(max_seq_len, batch_size, mode)
    return marker.do_bench(
        fixup_zero_kv_rows,
        input_args=(out, lse, kv_lens, cum_seq_lens, max_seq_len),
        graph_clone_args=None,
        graph_clone_kwargs=None,
        memory_args=(kv_lens, cum_seq_lens),
        memory_output=(out, lse),
    )


if __name__ == "__main__":
    benchmark.run()
