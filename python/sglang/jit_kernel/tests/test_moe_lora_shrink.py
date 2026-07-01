import sys

import pytest
import torch

from sglang.jit_kernel.moe_lora_shrink import moe_lora_shrink
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")


def build_routing(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    """Pure-python moe_align: group flattened (token, top-k) slots by expert into
    blocks of ``block_m``, padding short blocks with the ``num_valid`` sentinel.

    Mirrors moe_align_block_size for max_loras == 1 (virtual expert == topk id).
    Returns int32 (sorted_token_ids, expert_ids, num_tokens_post_padded[1]).
    """
    flat = topk_ids.reshape(-1).cpu().tolist()
    num_valid = len(flat)
    by_expert = {e: [] for e in range(num_experts)}
    for slot, e in enumerate(flat):
        by_expert[e].append(slot)

    sorted_token_ids = []
    expert_ids = []
    for e in range(num_experts):
        slots = by_expert[e]
        for start in range(0, len(slots), block_m):
            chunk = slots[start : start + block_m]
            chunk = chunk + [num_valid] * (block_m - len(chunk))
            sorted_token_ids.extend(chunk)
            expert_ids.append(e)

    npp = len(sorted_token_ids)
    dev = topk_ids.device
    return (
        torch.tensor(sorted_token_ids, dtype=torch.int32, device=dev),
        torch.tensor(expert_ids, dtype=torch.int32, device=dev),
        torch.tensor([npp], dtype=torch.int32, device=dev),
    )


def ref_shrink(hidden_states, lora_a, topk_ids, top_k):
    bs = topk_ids.shape[0]
    rank = lora_a.shape[1]
    a = lora_a.float()
    out = torch.zeros(
        bs * top_k, rank, device=hidden_states.device, dtype=torch.float32
    )
    flat = topk_ids.reshape(-1)
    for t in range(bs * top_k):
        e = int(flat[t].item())
        out[t] = hidden_states[t // top_k].float() @ a[e].t()
    return out


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])  # WMMA: no fp32
@pytest.mark.parametrize("bs", [1, 16, 64])
@pytest.mark.parametrize("num_experts", [8, 64])
@pytest.mark.parametrize("top_k", [1, 8])
@pytest.mark.parametrize("rank", [16, 32, 64])  # one or more 16-wide WMMA N-tiles
@pytest.mark.parametrize("hidden", [512, 2048])  # multiple of BLOCK_K=64
@pytest.mark.parametrize("block_m", [16])  # one WMMA M-tile per block
def test_moe_lora_shrink(dtype, bs, num_experts, top_k, rank, hidden, block_m):
    if top_k > num_experts:
        pytest.skip("top_k must be <= num_experts")
    device = "cuda"
    torch.manual_seed(0)

    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    hidden_states = torch.randn(bs, hidden, device=device, dtype=dtype) * 0.1
    lora_a = torch.randn(num_experts, rank, hidden, device=device, dtype=dtype) * 0.1

    sorted_token_ids, expert_ids, npp = build_routing(topk_ids, num_experts, block_m)

    output = torch.empty(bs * top_k, rank, device=device, dtype=dtype)
    moe_lora_shrink(
        output,
        hidden_states,
        lora_a,
        sorted_token_ids,
        expert_ids,
        npp,
        top_k,
        block_m,
    )

    ref = ref_shrink(hidden_states, lora_a, topk_ids, top_k)
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (2e-2, 2e-2)
    torch.testing.assert_close(output.float(), ref, rtol=rtol, atol=atol)


def test_moe_lora_shrink_unsupported_rank():
    device = "cuda"
    dtype = torch.bfloat16
    bs, num_experts, top_k, rank, hidden, block_m = 8, 8, 4, 24, 512, 16
    torch.manual_seed(0)

    topk_ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(bs)]
    ).to(torch.int32)
    hidden_states = torch.randn(bs, hidden, device=device, dtype=dtype) * 0.1
    lora_a = torch.randn(num_experts, rank, hidden, device=device, dtype=dtype) * 0.1
    sorted_token_ids, expert_ids, npp = build_routing(topk_ids, num_experts, block_m)

    output = torch.empty(bs * top_k, rank, device=device, dtype=dtype)
    with pytest.raises(RuntimeError, match="rank"):
        moe_lora_shrink(
            output,
            hidden_states,
            lora_a,
            sorted_token_ids,
            expert_ids,
            npp,
            top_k,
            block_m,
        )


def test_moe_lora_shrink_dtype_mismatch():
    device = "cuda"
    hidden_states = torch.randn(4, 64, device=device, dtype=torch.float16)
    lora_a = torch.randn(8, 16, 64, device=device, dtype=torch.bfloat16)
    output = torch.empty(4, 16, device=device, dtype=torch.float16)
    routing = build_routing(torch.zeros(4, 1, dtype=torch.int32, device=device), 8, 16)
    with pytest.raises(RuntimeError, match="dtype"):
        moe_lora_shrink(output, hidden_states, lora_a, *routing, 1, 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
