# python3 ./sglang_rebalance/python/sglang/srt/epmoe_permute_tensor.py
import torch
# Set a seed for reproducibility
torch.manual_seed(42)

# Create a global tensor variable for testing
# 61 rows, each row has a random permutation of integers from 0 to 255
EP_PERMUTE_TENSOR = torch.stack([
    torch.randperm(256)
    for _ in range(61)
], dim=0)

EP_BACK_MAPPING_TENSOR = torch.zeros((61, 256), dtype=torch.long)
for layer_idx in range(61):
    for expert_idx, permuted_expert_id in enumerate(EP_PERMUTE_TENSOR[layer_idx]):
        EP_BACK_MAPPING_TENSOR[layer_idx, permuted_expert_id] = expert_idx

# # Save the tensors to a text file
# with open("ep_permute_tensors.txt", "w") as f:
#     f.write("EP_PERMUTE_TENSOR:\n")
#     # Save the full tensor without truncation
#     torch.set_printoptions(threshold=float('inf'))
#     f.write(str(EP_PERMUTE_TENSOR))
#     f.write("\n\nEP_BACK_MAPPING_TENSOR:\n")
#     f.write(str(EP_BACK_MAPPING_TENSOR))
#     # Reset print options to default
#     torch.set_printoptions(threshold=1000)