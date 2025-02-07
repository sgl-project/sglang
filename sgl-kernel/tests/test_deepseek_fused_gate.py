import torch
import pytest
from tqdm import tqdm
from sgl_kernel import deepseekv3_fused_gate
from sglang.srt.layers.moe.topk import biased_grouped_topk


@pytest.mark.parametrize("seq_length", range(1, 20000))
def test_deepseekV3GateModule(seq_length):
    torch.manual_seed(seq_length)

    tensor = torch.rand((seq_length, 256)).to(torch.bfloat16).cuda()
    scores = tensor.clone()
    bias = torch.rand(256).to(torch.bfloat16).cuda()

    output, indices = deepseekv3_fused_gate(tensor, bias)

    ref_output, ref_indices = biased_grouped_topk(
        scores, scores, bias, topk=8, renormalize=True, num_expert_group=8, topk_group=4
    )

    # correctness check
    output_check = torch.allclose(ref_output.sort()[0].to(torch.float32), output.sort()[0].to(torch.float32), rtol=1e-04, atol=1e-05)
    idx_check = torch.allclose(ref_indices.sort()[0].to(torch.int32), indices.sort()[0].to(torch.int32), rtol=1e-04, atol=1e-05)

    assert output_check, f"Output mismatch at seq_length {seq_length}"
    assert idx_check, f"Indices mismatch at seq_length {seq_length}"


if __name__ == "__main__":
    pytest.main([__file__])