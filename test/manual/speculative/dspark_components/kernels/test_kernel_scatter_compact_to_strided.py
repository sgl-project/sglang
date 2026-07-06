import pytest
import torch

from sglang.srt.speculative.dspark_components.kernels.scatter_compact_to_strided import (
    scatter_compact_to_strided,
    scatter_compact_to_strided_triton,
)
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton kernel needs CUDA"
)

T = 6


@pytest.mark.parametrize("bs", [1, 2, 3, 8])
@pytest.mark.parametrize("dim", [16, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("pad", ["exact", "bucket"])
def test_triton_matches_torch_scatter(bs, dim, dtype, pad):
    device = torch.device("cuda")
    verify_lens = torch.randint(1, T + 1, (bs,), dtype=torch.int32, device=device)
    total = int(verify_lens.sum().item())
    graph_num_tokens = total if pad == "exact" else bs * T
    layout = RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=verify_lens, graph_num_tokens=graph_num_tokens
    )
    compact = torch.randn(graph_num_tokens, dim, dtype=dtype, device=device)
    ref = scatter_compact_to_strided(
        compact=compact, layout=layout, fill_value=0.0, verify_num_draft_tokens=T
    )
    got = scatter_compact_to_strided_triton(
        compact=compact, layout=layout, fill_value=0.0, verify_num_draft_tokens=T
    )
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    assert torch.equal(got, ref)
