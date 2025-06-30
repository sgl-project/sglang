import torch

def create_per_token_group_quant_test_data(
    num_tokens, hidden_dim, group_size, flags
):
    device = torch.device("cuda")

    if TODO:
        return TODO
    else:
        return torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.bfloat16)
