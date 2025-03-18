import pytest
import torch
from sgl_kernel import moe_fused_gate
from tqdm import tqdm

from sglang.srt.layers.moe.topk import biased_grouped_topk


@pytest.mark.parametrize(
    "seq_length",
    list(range(1, 100))
    + [
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
    ],
)
def test_moe_fused_gate(seq_length):
    torch.manual_seed(seq_length)
    dtype = torch.bfloat16
    num_experts, num_expert_group, topk_group, topk = 256, 8, 4, 8
    tensor = torch.rand((seq_length, num_experts)).to(dtype).cuda()
    scores = tensor.clone()
    bias = torch.rand(num_experts).to(dtype).cuda()

    output, indices = moe_fused_gate(
        tensor,
        bias,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        topk=topk,
    )
    ref_output, ref_indices = biased_grouped_topk(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        compiled=False,
    )
    # correctness check
    idx_check = torch.allclose(
        ref_indices.sort()[0].to(torch.int32),
        indices.sort()[0].to(torch.int32),
        rtol=1e-04,
        atol=1e-05,
    )
    output_check = torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-04,
        atol=1e-05,
    )

    assert idx_check, f"Indices mismatch at seq_length {seq_length}"
    assert output_check, f"Output mismatch at seq_length {seq_length}"


if __name__ == "__main__":
    pytest.main([__file__])
