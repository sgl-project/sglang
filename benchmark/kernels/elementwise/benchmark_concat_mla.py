import torch
import triton
import triton.language as tl
from sgl_kernel import concat_mla_k as concat_mla_k_cuda

DEVICE = triton.runtime.driver.active.get_active_torch_device()

num_local_heads = 128
qk_nope_head_dim = 128
qk_rope_head_dim = 64


def create_data(num_tokens):
    k_nope_container = torch.randn(
        (num_tokens, num_local_heads, qk_nope_head_dim + 128),
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_nope = k_nope_container[:, :, :qk_nope_head_dim]

    k_rope_container = torch.randn(
        (num_tokens, 1, 128 + qk_rope_head_dim), dtype=torch.bfloat16, device="cuda"
    )
    k_rope = k_rope_container[:, :, -qk_rope_head_dim:]

    k = torch.empty(
        (num_tokens, num_local_heads, qk_nope_head_dim + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return dict(k=k, k_nope=k_nope, k_rope=k_rope)


def fn_torch(k, k_nope, k_rope):
    k[..., :qk_nope_head_dim] = k_nope
    k[..., qk_nope_head_dim:] = k_rope


def fn_hack_non_strided(k, k_nope, k_rope):
    k_flatten_view = k.flatten()
    k_flatten_view[: k_nope.numel()] = k_nope.flatten()

    k2 = k_flatten_view[k_nope.numel() :].view(k_rope.numel(), -1)
    k2 = k_rope.flatten()[:, None]


@torch.compile(dynamic=True)
def fn_torch_compiled(k, k_nope, k_rope):
    return fn_torch(k, k_nope, k_rope)


def fn_cuda(k, k_nope, k_rope):
    concat_mla_k_cuda(k, k_nope, k_rope)


@triton.jit
def fn_triton_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    num_tokens,
    QK_NOPE_HEAD_DIM: tl.constexpr,
    QK_ROPE_HEAD_DIM: tl.constexpr,
    NUM_LOCAL_HEADS: tl.constexpr,
    K_NOPE_STRIDE_0: tl.constexpr,
    K_NOPE_STRIDE_1: tl.constexpr,
    K_STRIDE_0: tl.constexpr,
    K_STRIDE_1: tl.constexpr,
    K_ROPE_STRIDE_0: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    token_id = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    token_mask = token_id < num_tokens

    head_id = tl.arange(0, NUM_LOCAL_HEADS)

    # nope
    nope_sub_id = tl.arange(0, QK_NOPE_HEAD_DIM)
    offs_nope = (
        token_id[:, None, None] * K_NOPE_STRIDE_0
        + head_id[None, :, None] * K_NOPE_STRIDE_1
        + nope_sub_id[None, None, :]
    )
    offs_k = (
        token_id[:, None, None] * K_STRIDE_0
        + head_id[None, :, None] * K_STRIDE_1
        + nope_sub_id[None, None, :]
    )
    vals_nope = tl.load(k_nope_ptr + offs_nope, mask=token_mask[:, None, None])
    tl.store(k_ptr + offs_k, vals_nope, mask=token_mask[:, None, None])

    # rope
    rope_sub_id = tl.arange(0, QK_ROPE_HEAD_DIM)
    offs_rope = token_id[:, None, None] * K_ROPE_STRIDE_0 + rope_sub_id[None, None, :]
    offs_k = (
        token_id[:, None, None] * K_STRIDE_0
        + head_id[None, :, None] * K_STRIDE_1
        + rope_sub_id[None, None, :]
        + QK_NOPE_HEAD_DIM
    )
    vals_rope = tl.load(k_rope_ptr + offs_rope, mask=token_mask[:, None, None])
    tl.store(k_ptr + offs_k, vals_rope, mask=token_mask[:, None, None])


def fn_triton(k, k_nope, k_rope):
    assert k.device == DEVICE and k_nope.device == DEVICE and k_rope.device == DEVICE
    num_tokens, _, _ = k.shape
    grid = lambda meta: (triton.cdiv(num_tokens, meta["BLOCK_ROWS"]),)
    fn_triton_kernel[grid](
        k,
        k_nope,
        k_rope,
        num_tokens,
        QK_NOPE_HEAD_DIM=qk_nope_head_dim,
        QK_ROPE_HEAD_DIM=qk_rope_head_dim,
        NUM_LOCAL_HEADS=num_local_heads,
        K_NOPE_STRIDE_0=k_nope.stride(0),
        K_NOPE_STRIDE_1=k_nope.stride(1),
        K_STRIDE_0=k.stride(0),
        K_STRIDE_1=k.stride(1),
        K_ROPE_STRIDE_0=k_rope.stride(0),
        BLOCK_ROWS=16,
    )


def execute_and_get_output(f, data):
    data["k"].zero_()
    f(**data)
    assert data["k"].sum().item() != 0
    return data["k"].clone()


torch.manual_seed(0)
data = create_data(num_tokens=32768)
output_ref = execute_and_get_output(fn_torch, data)
output_exp = execute_and_get_output(fn_cuda, data)
# print(output_ref)
# print(output_exp)
if not torch.all(output_ref == output_exp):
    abs_delta = torch.abs(output_ref - output_exp)
    raise AssertionError(
        f"{output_ref=} {output_exp=} "
        f"{abs_delta=} "
        f"{torch.argwhere(abs_delta != 0.0)=} "
    )


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["num_tokens"],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2048,
            4096,
            8192,
            16384,
            32768,
        ],  # Different possible values for `x_name`.
        x_log=False,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=[
            "torch",
            "torch_compiled",
            "triton",
            "hack_non_strided",
            "cuda",
        ],  # Possible values for `line_arg`.
        line_names=[
            "torch",
            "torch_compiled",
            "triton",
            "hack_non_strided",
            "cuda",
        ],  # Label name for the lines.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(num_tokens, provider):
    data = create_data(num_tokens=num_tokens)
    quantiles = [0.5, 0.2, 0.8]
    fn = {
        "torch": fn_torch,
        "torch_compiled": fn_torch_compiled,
        "triton": fn_triton,
        "hack_non_strided": fn_hack_non_strided,
        "cuda": fn_cuda,
    }[provider]
    ms, min_ms, max_ms = triton.testing.do_bench(
        lambda: fn(**data), quantiles=quantiles
    )
    return ms, min_ms, max_ms


torch.cuda.cudart().cudaProfilerStart()
benchmark.run(print_data=True, show_plots=True)
torch.cuda.cudart().cudaProfilerStop()
