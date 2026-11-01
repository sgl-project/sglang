"""Regression vector for uint32 rotation with the most significant bit set."""

import pytest
import torch

pytest.importorskip("torch_npu")
from sglang.kernels.ops.sampling.murmur_hash import murmur_hash32


@pytest.mark.skipif(not torch.npu.is_available(), reason="Requires an Ascend NPU")
def test_murmur_high_bit_columns():
    seed = torch.tensor([2092917972], dtype=torch.uint64, device="npu")
    position = torch.tensor([88], dtype=torch.int64, device="npu")
    columns = torch.arange(16, device="npu")
    # Independent 32-bit Murmur3 reference; columns 1,2,5,6,7,10,11,12,15
    # set the high bit after the first multiply in the column mixing step.
    expected = [
        [
            3847850963,
            2821684749,
            3379976169,
            551063260,
            3868596079,
            3051706844,
            234660108,
            1276890786,
            2134884091,
            1601909887,
            3088546729,
            907784942,
            2462296977,
            2633501825,
            3345631188,
            2252777896,
        ]
    ]
    assert murmur_hash32(seed, position, columns).cpu().tolist() == expected
