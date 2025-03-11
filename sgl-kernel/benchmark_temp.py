
import torch
from sgl_kernel import topk_softmax

num_tokens = 1024
num_experts = 64
topk = 4

gating_output = torch.randn(num_tokens, num_experts, device='cuda', dtype=torch.float32)
topk_weights = torch.zeros(num_tokens, topk, device='cuda', dtype=torch.float32)
topk_indices = torch.zeros(num_tokens, topk, device='cuda', dtype=torch.int32)
token_expert_indices = torch.zeros(num_tokens, topk, device='cuda', dtype=torch.int32)

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
end.record()

torch.cuda.synchronize()
