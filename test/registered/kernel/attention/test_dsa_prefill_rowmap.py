from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.kernels.ops.attention.dsa.transform_index as transform_index_module
from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_row_map,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_topk_backend import (
    DSATopKBackend,
    TopkTransformMethod,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="A GPU is required for this test.",
)


@pytest.mark.parametrize(
    "num_requests,num_rows,width,topk",
    [
        (3, 17, 257, 64),
        (16, 32, 60003, 2048),
    ],
)
def test_row_map_matches_pytorch(
    num_requests: int,
    num_rows: int,
    width: int,
    topk: int,
) -> None:
    torch.manual_seed(7)
    page_table = torch.randint(
        0,
        2**30,
        (num_requests, width),
        dtype=torch.int32,
        device="cuda",
    )
    row_to_batch = torch.randint(
        0,
        num_requests,
        (num_rows,),
        dtype=torch.int32,
        device="cuda",
    )
    logical = torch.randint(
        0,
        width,
        (num_rows, topk),
        dtype=torch.int32,
        device="cuda",
    )
    logical[::3, -7:] = -1

    actual = transform_index_page_table_row_map(page_table, logical, row_to_batch)
    reference = page_table[
        row_to_batch[:, None].to(torch.int64),
        logical.clamp_min(0).to(torch.int64),
    ]
    reference = torch.where(logical >= 0, reference, -1)

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_chunked_paged_topk_uses_row_map() -> None:
    page_table = torch.arange(8, dtype=torch.int32, device="cuda").reshape(2, 4)
    logical = torch.tensor([[3, 1], [0, 2]], dtype=torch.int32, device="cuda")
    row_to_batch = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
    logits = torch.zeros((2, 4), dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4, dtype=torch.int32, device="cuda")
    expected = torch.empty_like(logical)

    with (
        envs.SGLANG_DSA_FUSE_TOPK.override(True),
        patch.object(DSATopKBackend, "topk_func", return_value=logical),
        patch.object(
            transform_index_module,
            "transform_index_page_table_row_map",
            return_value=expected,
        ) as row_map,
    ):
        actual = DSATopKBackend.SGL_KERNEL.topk_transform(
            logits=logits,
            lengths=lengths,
            topk=2,
            topk_transform_method=TopkTransformMethod.PAGED,
            attn_metadata=SimpleNamespace(page_table_1=page_table),
            batch_idx_list=row_to_batch,
        )

    assert actual is expected
    row_map.assert_called_once_with(page_table, logical, row_to_batch)
