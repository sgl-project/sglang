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

        x = torch.randn(num_local_experts, TODO, effective_hidden_dim, device=device, dtype=dtype)

        masked_data_generation_mode = flags.get("masked_data_generation_mode", "default")
        if masked_data_generation_mode == "default":
            masked_m = TODO
        elif masked_data_generation_mode == "imbalanced":
            masked_m = TODO
        else:
            raise NotImplementedError

        return x, masked_m
    else:
        return torch.randn(num_tokens, effective_hidden_dim, device=device, dtype=dtype), None
