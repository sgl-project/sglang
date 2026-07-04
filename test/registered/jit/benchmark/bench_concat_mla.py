import torch
from sgl_kernel import concat_mla_absorb_q as aot_absorb_q
from sgl_kernel import concat_mla_k as aot_k

from sglang.jit_kernel.benchmark import marker
from sglang.jit_kernel.concat_mla import concat_mla_absorb_q as jit_absorb_q
from sglang.jit_kernel.concat_mla import concat_mla_k as jit_k
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NUM_LOCAL_HEADS = 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM

A_LAST_DIM = 512
B_LAST_DIM = 64

DTYPE = torch.bfloat16
DEVICE = "cuda"


def aot_concat_mla_k(k, k_nope, k_rope):
    aot_k(k, k_nope, k_rope)


def jit_concat_mla_k(k, k_nope, k_rope):
    jit_k(k, k_nope, k_rope)


def torch_concat_mla_k(k, k_nope, k_rope):
    nope_head_dim = k_nope.shape[-1]
    k[:, :, :nope_head_dim] = k_nope
    k[:, :, nope_head_dim:] = k_rope.expand(-1, k.shape[1], -1)


def aot_concat_mla_absorb_q(a, b):
    return aot_absorb_q(a, b)


def jit_concat_mla_absorb_q(a, b):
    return jit_absorb_q(a, b)


def torch_concat_mla_absorb_q(a, b, out):
    a_last_dim = a.shape[-1]
    out[:, :, :a_last_dim] = a
    out[:, :, a_last_dim:] = b


K_FN_MAP = {
    "aot": aot_concat_mla_k,
    "jit": jit_concat_mla_k,
    "torch": torch_concat_mla_k,
}


def _create_concat_mla_k_data(num_tokens):
    """Allocate oversized containers and slice to produce non-contiguous tensors."""
    k_nope_container = torch.randn(
        (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + 128),
        dtype=DTYPE,
        device=DEVICE,
    )
    k_nope = k_nope_container[:, :, :QK_NOPE_HEAD_DIM]

    k_rope_container = torch.randn(
        (num_tokens, 1, 128 + QK_ROPE_HEAD_DIM),
        dtype=DTYPE,
        device=DEVICE,
    )
    k_rope = k_rope_container[:, :, -QK_ROPE_HEAD_DIM:]

    k = torch.empty(
        (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM),
        dtype=DTYPE,
        device=DEVICE,
    )
    return k, k_nope, k_rope


@marker.parametrize(
    "num_tokens",
    [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    [256, 1024],
)
@marker.benchmark("provider", ["aot", "jit", "torch"])
def bench_concat_mla_k(num_tokens: int, provider: str):
    k, k_nope, k_rope = _create_concat_mla_k_data(num_tokens)
    return marker.do_bench(
        K_FN_MAP[provider],
        input_args=(k, k_nope, k_rope),
        graph_clone_args=(0, 1, 2),
        memory_args=(k_nope, k_rope),
        memory_output=(k,),  # inplace write to k
    )


@marker.parametrize("dim_0", [1, 4, 8, 16, 32], [4, 16])
@marker.parametrize("dim_1", [1, 8, 32, 128], [16])
@marker.benchmark("provider", ["aot", "jit", "torch"])
def bench_concat_mla_absorb_q(dim_0: int, dim_1: int, provider: str):
    a = torch.randn(dim_0, dim_1, A_LAST_DIM, dtype=DTYPE, device=DEVICE)
    b = torch.randn(dim_0, dim_1, B_LAST_DIM, dtype=DTYPE, device=DEVICE)

    if provider == "torch":
        out = torch.empty(
            dim_0, dim_1, A_LAST_DIM + B_LAST_DIM, dtype=DTYPE, device=DEVICE
        )
        return marker.do_bench(
            torch_concat_mla_absorb_q,
            input_args=(a, b, out),
            graph_clone_args=(0, 1, 2),
            memory_args=(a, b),
            memory_output=(out,),  # inplace write to out
        )

    fn = aot_concat_mla_absorb_q if provider == "aot" else jit_concat_mla_absorb_q
    return marker.do_bench(
        fn,
        input_args=(a, b),
        graph_clone_args=(0, 1),
        memory_args=(a, b),
    )


if __name__ == "__main__":
    bench_concat_mla_k.run()
    bench_concat_mla_absorb_q.run()
