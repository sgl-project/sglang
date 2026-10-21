import torch

from sglang.kernels.ops.speculative import cache_locs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_assign_extend_cache_locs_fallback_preserves_variable_ranges(monkeypatch):
    for platform_flag in (
        "_is_cuda",
        "_is_hip",
        "_is_musa",
        "_is_xpu",
        "_is_npu",
        "_is_cpu",
    ):
        monkeypatch.setattr(cache_locs, platform_flag, False)

    req_to_token = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
        ],
        dtype=torch.int32,
    )
    output = cache_locs.assign_extend_cache_locs_func(
        req_pool_indices=torch.tensor([2, 0, 1]),
        req_to_token=req_to_token,
        start_offset=torch.tensor([1, 2, 0]),
        end_offset=torch.tensor([3, 3, 3]),
        batch_size=3,
        draft_token_num=2,
        device="cpu",
    )

    torch.testing.assert_close(
        output,
        torch.tensor([31, 32, 12, 20, 21, 22], dtype=torch.int64),
    )
