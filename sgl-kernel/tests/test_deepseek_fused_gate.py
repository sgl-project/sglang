import torch
import pytest
from tqdm import tqdm
from sgl_kernel import deepseekv3_fused_gate


@pytest.mark.parametrize("seq_length", range(1, 20000))
def test_deepseekV3GateModule(seq_length):
    torch.manual_seed(seq_length)

    tensor = torch.rand((seq_length, 256)).to(torch.bfloat16).cuda()
    scores = tensor.clone()
    bias = torch.rand(256).to(torch.bfloat16).cuda()

    # Your module under test
    output, indices_my = deepseekv3_fused_gate(tensor, bias, seq_length)

    ###### Reference Implementation ######
    scores = scores.sigmoid()
    original_scores = scores
    scores = scores + bias
    n_groups = 8

    scores = scores.view(scores.size(0), n_groups, -1)
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    indices = group_scores.topk(4, dim=-1)[1]

    mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
    scores = (scores * mask.unsqueeze(-1)).flatten(1)
    indices = torch.topk(scores, 8, dim=-1)[1]
    weights = original_scores.gather(1, indices)

    weights /= weights.sum(dim=-1, keepdim=True)
    weights *= 2.5
    output_ref = weights.type_as(scores)

    # Assertions
    output_check = torch.allclose(output_ref.sort()[0], output.sort()[0], rtol=1e-04, atol=1e-05)
    idx_check = torch.allclose(indices.sort()[0], indices_my.sort()[0], rtol=1e-04, atol=1e-05)

    assert output_check, f"Output mismatch at seq_length {seq_length}"
    assert idx_check, f"Indices mismatch at seq_length {seq_length}"


if __name__ == "__main__":
    pytest.main([__file__])