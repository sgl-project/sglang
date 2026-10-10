import pytest
import torch

from sglang.kernels.ops.kvcache.trtllm_mha_v_tail import zero_v_page_tails
from sglang.srt.mem_cache.layout.paged_view import paged_kv_view
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEADS = 2
LAYERS = 3
NUM_PAGES = 12

# (seq_len, q_len, last page id, first zeroed row: only new pages, every forward).
# seq_len is the KV length after this forward and q_len the tokens it writes.
CASES = {
    16: [
        (33, 1, 3, 1, 1),  # decode whose token starts page 2
        (40, 1, 4, None, 8),  # decode inside page 2
        (50, 10, 6, 2, 2),  # extend that starts page 3
        (45, 5, 7, None, 13),  # extend whose last page began in an earlier forward
        (48, 4, 8, None, None),  # full last page
        (1, 1, 0, None, None),  # CUDA-graph padding row on page 0
    ],
    64: [
        (65, 1, 3, 1, 1),
        (100, 1, 4, None, 36),
        (140, 20, 6, 12, 12),
        (150, 5, 7, None, 22),
        (128, 4, 8, None, None),
        (1, 1, 0, None, None),
    ],
}


def _zero_tails(buffers, cases, *, page, dim, every_forward):
    page_table = torch.zeros(len(cases), 8, dtype=torch.int32, device="cuda")
    for i, (seq_len, _, last_page, _, _) in enumerate(cases):
        page_table[i, (seq_len - 1) // page] = last_page
    q_lens = torch.tensor([c[1] for c in cases], dtype=torch.int32, device="cuda")
    page_stride, row_stride, head_stride, dim_stride = paged_kv_view(
        buffers[0], page, HEADS, dim
    ).stride()
    zero_v_page_tails(
        v_ptrs=torch.tensor(
            [b.data_ptr() for b in buffers], dtype=torch.uint64, device="cuda"
        ),
        page_table=page_table,
        seq_lens=torch.tensor([c[0] for c in cases], dtype=torch.int32, device="cuda"),
        cu_seqlens_q=torch.nn.functional.pad(
            torch.cumsum(q_lens, 0, dtype=torch.int32), (1, 0)
        ),
        v_paged_strides=(page_stride, head_stride, row_stride, dim_stride),
        num_heads=HEADS,
        page_size=page,
        head_dim=dim,
        every_forward=every_forward,
    )


@pytest.mark.parametrize("every_forward", [False, True])
@pytest.mark.parametrize("page,dim", [(16, 64), (64, 128), (64, 96)])
@pytest.mark.parametrize("layout", ["hnd", "nhd"])
def test_zeroes_exactly_the_tail_rows_of_each_last_page(
    layout, page, dim, every_forward
):
    """Only last-page V rows past seq_len change; mid-page pages only with every_forward."""
    torch.manual_seed(0)
    shape = (
        (NUM_PAGES, HEADS, page, dim)
        if layout == "hnd"
        else (NUM_PAGES * page, HEADS, dim)
    )
    buffers = [
        torch.randn(shape, dtype=torch.bfloat16, device="cuda") for _ in range(LAYERS)
    ]
    expected = [paged_kv_view(b.clone(), page, HEADS, dim) for b in buffers]
    for _, _, last_page, new_only_row, every_row in CASES[page]:
        first_row = every_row if every_forward else new_only_row
        if first_row is not None:
            for view in expected:
                view[last_page, first_row:] = 0

    _zero_tails(buffers, CASES[page], page=page, dim=dim, every_forward=every_forward)

    for buffer, want in zip(buffers, expected):
        got = paged_kv_view(buffer, page, HEADS, dim)
        torch.testing.assert_close(got, want, rtol=0, atol=0)


def _trtllm_gen_available() -> bool:
    if torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import flashinfer.decode  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.skipif(not _trtllm_gen_available(), reason="needs SM100 TRT-LLM-gen")
def test_trtllm_gen_decode_ignores_stale_rows_after_zeroing():
    """A NaN an earlier request left past seq_len on a reused page must not reach decode output."""
    import flashinfer.decode

    torch.manual_seed(0)
    page, dim, q_heads = 16, 64, 8
    seq_len = 33  # this decode token starts page 2, held by page id 3
    cache = torch.randn(
        NUM_PAGES, 2, HEADS, page, dim, dtype=torch.bfloat16, device="cuda"
    )
    block_tables = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    query = torch.randn(1, q_heads, dim, dtype=torch.bfloat16, device="cuda")
    workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")

    def decode():
        out = flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            query=query,
            kv_cache=cache,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=4096,
            bmm1_scale=dim**-0.5,
            bmm2_scale=1.0,
        )
        torch.cuda.synchronize()
        return out

    with_finite_tail = decode()
    cache[3, 1, :, seq_len % page :] = float("nan")
    assert not torch.isfinite(decode()).all(), "precondition: the kernel reads the tail"

    v = cache[:, 1]  # [pages, heads, page, dim]
    zero_v_page_tails(
        v_ptrs=torch.tensor([v.data_ptr()], dtype=torch.uint64, device="cuda"),
        page_table=block_tables,
        seq_lens=seq_lens,
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        v_paged_strides=v.stride(),
        num_heads=HEADS,
        page_size=page,
        head_dim=dim,
        every_forward=False,
    )
    torch.testing.assert_close(decode(), with_finite_tail, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
