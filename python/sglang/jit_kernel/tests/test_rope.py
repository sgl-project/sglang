import pytest
import torch

torch.ops.load_library(
    "/usr/local/lib/python3.12/dist-packages/sgl_kernel/sm100/common_ops.abi3.so"
)
from sglang.jit_kernel.rope import apply_rope_pos_ids_cos_sin_cache

DEVICE = "cuda"


def create_cos_sin_cache(rotary_dim, max_position_embeddings, base, dtype):
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=DEVICE)
            / rotary_dim
        )
    )

    t = torch.arange(max_position_embeddings, dtype=torch.float32, device=DEVICE)
    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache


def _view_3d(x, head_size):
    return x.view(x.shape[0], -1, head_size)


@pytest.mark.parametrize("bs", [1, 8])
@pytest.mark.parametrize("seq_len", [1, 256, 512])
@pytest.mark.parametrize("num_qo_heads", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 8, 16])
@pytest.mark.parametrize("head_dim", [64, 128, 256, 512])
@pytest.mark.parametrize("rotary_dim", [64, 128])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("save_kv_cache", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_rope(
    bs,
    seq_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    interleave: bool,
    enable_pdl: bool,
    save_kv_cache: bool,
    dtype: torch.dtype,
) -> None:
    if head_dim < rotary_dim:
        pytest.skip(f"{head_dim=} < {rotary_dim=}")
    if not save_kv_cache and enable_pdl:
        pytest.skip(f"({save_kv_cache=}, {enable_pdl=}) is not allowed")

    q = torch.randn(bs * seq_len, num_qo_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(bs * seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(bs * seq_len, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)

    KV_POOL_SIZE = bs * seq_len * 2
    k_buffer = torch.zeros(
        KV_POOL_SIZE, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    v_buffer = torch.zeros(
        KV_POOL_SIZE, num_kv_heads, head_dim, device=DEVICE, dtype=dtype
    )
    out_cache_loc = torch.randperm(KV_POOL_SIZE, dtype=torch.int64, device=DEVICE)[
        : bs * seq_len
    ].clone()

    pos_ids = torch.arange(seq_len, device=DEVICE).repeat(bs)

    max_seq_len = seq_len
    base = 10000
    cos_sin_cache = create_cos_sin_cache(rotary_dim, max_seq_len, base, dtype)

    q_jit = q.clone()
    k_jit = k.clone()
    v_jit = v.clone()
    k_buffer_jit = k_buffer.clone()
    v_buffer_jit = v_buffer.clone()
    out_cache_loc_jit = out_cache_loc.clone()

    q_kernel = q.clone()
    k_kernel = k.clone()
    v_kernel = v.clone()
    k_buffer_kernel = k_buffer.clone()
    v_buffer_kernel = v_buffer.clone()
    out_cache_loc_kernel = out_cache_loc.clone()

    apply_rope_pos_ids_cos_sin_cache(
        q_jit,
        k_jit,
        q_jit,
        k_jit,
        cos_sin_cache,
        pos_ids,
        interleave,
        enable_pdl,
        v_jit,
        k_buffer_jit,
        v_buffer_jit,
        out_cache_loc_jit,
    )

    torch.ops.sgl_kernel.apply_rope_pos_ids_cos_sin_cache.default(
        q_kernel,
        k_kernel,
        q_kernel,
        k_kernel,
        cos_sin_cache,
        pos_ids,
        interleave,
        enable_pdl,
        v_kernel,
        k_buffer_kernel,
        v_buffer_kernel,
        out_cache_loc_kernel,
    )

    atol = 1e-3 if dtype != torch.float32 else 1e-6
    rtol = 1e-3 if dtype != torch.float32 else 1e-6
    torch.testing.assert_close(q_jit, q_kernel, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_jit, k_kernel, atol=atol, rtol=rtol)
    torch.testing.assert_close(k_buffer_jit, k_buffer_kernel, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_buffer_jit, v_buffer_kernel, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main([__file__])
