import torch
import triton

from sglang.srt.layers.moe.fused_moe_triton.fused_valu_moe import diverse_experts


# @torch.compile(options={"triton.cudagraphs": True}, fullgraph=True)
def torch_diverse_experts(topk_ids, num_experts, valu_tokens, align_size):
    m, topk = topk_ids.shape
    EM = m * topk

    # 1. do bincount
    bin_counts = torch.bincount(topk_ids.view(-1))

    # 2. sort
    sorted_value, sorted_ids = torch.sort(bin_counts, stable=True)

    # 3. find the critical position
    cumsum = torch.cumsum(sorted_value, dim=0)
    critical_pos = (cumsum > valu_tokens).nonzero()[0][0]

    # 4. valu pre-process # codespell:ignore
    valu_expert_filt = sorted_ids[:critical_pos]
    valu_expert_ids = torch.tensor([], device=topk_ids.device, dtype=topk_ids.dtype)
    valu_sorted_token_ids = torch.tensor(
        [], device=topk_ids.device, dtype=topk_ids.dtype
    )
    for expert_id in valu_expert_filt:
        sorted_token_ids = torch.eq(topk_ids.view(-1), expert_id).nonzero().flatten()
        expert_ids = torch.zeros_like(sorted_token_ids) + expert_id
        valu_sorted_token_ids = torch.cat((valu_sorted_token_ids, sorted_token_ids))
        valu_expert_ids = torch.cat((valu_expert_ids, expert_ids))
    valu_valid_tokens = torch.tensor(
        [len(valu_sorted_token_ids)], device=topk_ids.device, dtype=topk_ids.dtype
    )

    # 5. tfu pre-process
    tfu_expert_filt = sorted_ids[critical_pos:]
    tfu_expert_ids = torch.tensor([], device=topk_ids.device, dtype=topk_ids.dtype)
    tfu_sorted_token_ids = torch.tensor(
        [], device=topk_ids.device, dtype=topk_ids.dtype
    )
    for expert_id in tfu_expert_filt:
        sorted_token_ids = torch.eq(topk_ids.view(-1), expert_id).nonzero().flatten()
        padding_len = triton.cdiv(len(sorted_token_ids), align_size) * align_size - len(
            sorted_token_ids
        )
        if padding_len > 0:
            padding_val = (
                torch.zeros(
                    (padding_len,), device=topk_ids.device, dtype=topk_ids.dtype
                )
                + EM
            )
            sorted_token_ids = torch.cat((sorted_token_ids, padding_val))
        expert_ids = (
            torch.zeros(
                (len(sorted_token_ids) // align_size,),
                device=topk_ids.device,
                dtype=topk_ids.dtype,
            )
            + expert_id
        )
        tfu_sorted_token_ids = torch.cat((tfu_sorted_token_ids, sorted_token_ids))
        tfu_expert_ids = torch.cat((tfu_expert_ids, expert_ids))
    num_tokens_post_padded = torch.tensor(
        [len(tfu_sorted_token_ids)], device=topk_ids.device, dtype=topk_ids.dtype
    )

    return (
        valu_expert_ids,
        valu_sorted_token_ids,
        valu_valid_tokens,
        tfu_expert_ids,
        tfu_sorted_token_ids,
        num_tokens_post_padded,
    )


def test_diverse_experts(
    num_tokens,
    num_experts,
    topk,
    valu_tokens,
    align_size,
    hidden_size=1024,
    dtype=torch.float16,
):
    score = torch.randn((num_tokens, num_experts), device="cuda", dtype=dtype)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    with torch.cuda.nvtx.range(f"triton diverse_experts"):
        results = diverse_experts(
            topk_ids,
            num_experts,
            valu_tokens,
            align_size,
        )
    with torch.cuda.nvtx.range(f"torch diverse_experts"):
        # compiled_model = torch.compile(torch_diverse_experts, backend="inductor", mode="max-autotune")
        # reference = compiled_model(topk_ids, num_experts, valu_tokens, align_size, )
        reference = torch_diverse_experts(
            topk_ids,
            num_experts,
            valu_tokens,
            align_size,
        )

    results = list(results)
    valu_valid_tokens = results[2].item()
    results[0] = results[0][:valu_valid_tokens]
    results[1] = results[1][:valu_valid_tokens]
    num_tokens_post_padded = results[5].item()
    results[3] = results[3][: num_tokens_post_padded // align_size]
    results[4] = results[4][:num_tokens_post_padded]

    for i, (res, ref) in enumerate(zip(results, reference)):
        if res.shape != ref.shape:
            print(
                f"shape fail at {i}th Tensor, res shape = {res.shape}, ref shape = {ref.shape}"
            )
            return
        validate = torch.allclose(res, ref)
        if not validate:
            print(f"validate fail at {i}th Tensor, res={res}, ref={ref}")
            return
    print(f"validation OK!!!")


if __name__ == "__main__":
    torch.manual_seed(2025)
    test_diverse_experts(200, 160, 6, 150, 16)
