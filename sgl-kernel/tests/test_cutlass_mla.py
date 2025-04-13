import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import cutlass_mla_decode, cutlass_mla_get_workspace_size
from torch import Tensor

if torch.cuda.get_device_capability() < (10, 0):
    pytest.skip(
        reason="Cutlass MLA Requires compute capability of 10 or above.",
        allow_module_level=True,
    )


def ref_mla(
    out: Tensor,  # (bs, num_heads, v_head_dim)
    query: Tensor,  # (bs, num_heads, head_dim)
    kv_cache: Tensor,  # (num_blocks, block_size, head_dim)
    scale: float,
    block_tables: Tensor,  # (bs, max_num_blocks)
    seq_lens: Tensor,  # (bs,)
):
    bs, num_heads, v_head_dim = out.shape
    head_dim = query.shape[2]

    for i in range(bs):
        # gather and flatten KV-cache
        kv = kv_cache[block_tables[i]]  # (max_num_blocks, block_size, head_dim)
        kv = kv.view(1, -1, head_dim)[:, : seq_lens[i]]  # (1, seq_len, head_dim)
        v = kv[:, :, :v_head_dim]

        q = query[i].view(num_heads, 1, head_dim)
        o = F.scaled_dot_product_attention(q, kv, v, scale=scale, enable_gqa=True)
        out[i] = o.view(num_heads, v_head_dim)

    return out


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mean_seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("bs", [1, 2, 4])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("block_size", [128])
def test_cutlass_mla_decode(
    dtype: torch.dtype, mean_seq_len: int, bs: int, varlen: bool, block_size: int
):
    torch.set_default_dtype(dtype)
    torch.set_default_device("cuda")
    torch.manual_seed(42)

    d = 576
    h_q = 128
    dv = 512

    q_nope_dim = 128
    q_pe_dim = 64
    scale = (q_nope_dim + q_pe_dim) ** (-0.5)
    if varlen:
        seq_lens = torch.empty(bs).normal_(mean_seq_len, mean_seq_len / 2)
        seq_lens = seq_lens.clip(2).to(torch.int32)
    else:
        seq_lens = torch.full((bs,), mean_seq_len, dtype=torch.int32)
    max_seq_len = seq_lens.max().item()
    block_num = (max_seq_len + block_size - 1) // block_size

    q = torch.randn(bs, h_q, d)
    block_table = torch.randint(0, bs * block_num, (bs, block_num), dtype=torch.int32)

    kv_cache = torch.randn(block_table.numel(), block_size, d)

    workspace_size = cutlass_mla_get_workspace_size(block_num * block_size, bs)
    workspace = torch.empty(workspace_size, device="cuda", dtype=torch.uint8)

    out_ref = q.new_zeros(bs, h_q, dv)
    ref_mla(out_ref, q, kv_cache, scale, block_table, seq_lens)
    out = cutlass_mla_decode(q, kv_cache, seq_lens, block_table, workspace)

    torch.testing.assert_close(out, out_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
