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
        x = torch.randn(
            num_local_experts,
            num_max_dispatch_tokens_per_rank * num_ranks,
            effective_hidden_dim,
            device=device,
            dtype=dtype,
        )

        masked_data_generation_mode = flags.get(
            "masked_data_generation_mode", "default"
        )
        if masked_data_generation_mode == "default":
            masked_m = _compute_balanced_split(num_tokens, num_local_experts)
        elif masked_data_generation_mode == "imbalanced":
            masked_m = _compute_imbalanced_split(num_tokens, num_local_experts)
        else:
            raise NotImplementedError

        masked_m = torch.tensor(masked_m, device="cuda", dtype=torch.int)

        return x, masked_m
    else:
        return (
            torch.randn(num_tokens, effective_hidden_dim, device=device, dtype=dtype),
            None,
        )


def _compute_balanced_split(total: int, arr_len: int) -> list[int]:
    base = total // arr_len
    remainder = total % arr_len
    ans = [base + 1 if i < remainder else base for i in range(arr_len)]
    assert sum(ans) == total
    return ans

def _compute_imbalanced_split(total: int, arr_len: int) -> list[int]:
    return TODO
