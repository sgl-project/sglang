import torch
# Set a seed for reproducibility
torch.manual_seed(42)

# Create a global tensor variable for testing
# 61 rows, each row has a random permutation of integers from 0 to 255
EP_PERMUTE_TENSOR = torch.stack([
    torch.randperm(256)
    for _ in range(61)
], dim=0)