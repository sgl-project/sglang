import pytest
import torch
from sgl_kernel import apply_token_bitmask_inplace_cuda


def test_apply_token_bitmask_inplace_kernel():
    neginf = float("-inf")
    bool_mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
    logits = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=torch.float32
    )
    expected = torch.where(bool_mask, logits, neginf)

    logits_gpu = logits.to("cuda")
    bitmask = torch.tensor([0b1010101010], dtype=torch.int32).to("cuda")
    apply_token_bitmask_inplace_cuda(logits_gpu, bitmask)
    torch.cuda.synchronize()
    torch.testing.assert_close(logits_gpu, expected.to("cuda"))


if __name__ == "__main__":
    test_apply_token_bitmask_inplace_kernel()
    pytest.main([__file__])
