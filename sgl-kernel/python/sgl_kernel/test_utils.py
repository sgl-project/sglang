import torch


def create_per_token_group_quant_test_data(num_tokens, hidden_dim, flags):
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if flags["fuse_silu_and_mul"]:
        effective_hidden_dim = hidden_dim * 2
    else:
        effective_hidden_dim = hidden_dim
    del hidden_dim

    if flags["masked_layout"]:
        num_local_experts = 6
        num_max_dispatch_tokens_per_rank = 768
        num_ranks = 48
        num_global_experts = 288
        assert num_local_experts * num_ranks == num_global_experts

        # mimic DeepEP low_latency_dispatch output
        x = torch.randn(num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, effective_hidden_dim, device=device, dtype=dtype)

        masked_data_generation_mode = flags.get("masked_data_generation_mode", "default")
        if masked_data_generation_mode == "default":
            masked_m = torch.tensor(_split_evenly(num_tokens, num_local_experts), device='cuda', dtype=torch.int)
        elif masked_data_generation_mode == "imbalanced":
            masked_m = TODO
        else:
            raise NotImplementedError

        return x, masked_m
    else:
        return torch.randn(num_tokens, effective_hidden_dim, device=device, dtype=dtype), None

def _split_evenly(num: int, arr_len: int) -> list[int]:
    base = num // arr_len
    remainder = num % arr_len
    ans = [base + 1 if i < remainder else base for i in range(arr_len)]
    assert sum(ans) == num
    return ans
