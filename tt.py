import torch
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double() + 1, y.double() + 1
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()
x=torch.tensor([[-99., -99., -99.,  -99., -99,   0.,   0.,   0.]])
topk_weights=torch.tensor([[0.3153, 0.5592, 1.1223, 1.4370, 1.8091, 0.0235, 0.4934, 0.5309]], dtype=torch.float32)
topk_idx=torch.tensor([[-1, -1, -1, -1, -1, 57, -1, -1]])
combined_x=torch.tensor([[-2.2656, -2.2656, -2.2656,  -2.2656, -2.2656, 0.0000,  0.0000,  0.0000]],)


diff = calc_diff(x * topk_weights.masked_fill(topk_idx == -1, 0).sum(dim=1).view(-1, 1), combined_x)
assert torch.isnan(combined_x).sum().item() == 0
assert diff < 1e-5, f'Error: {diff=}'