import statistics
import time

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from transformer_engine.common import recipe

# Set dimensions.
in_features = 768
out_features = 3072
hidden_size = 2048
num_iterations = 100

# Initialize TE model and inputs.
te_model = te.Linear(in_features, out_features, bias=True).cuda()
# Initialize Torch model and inputs
torch_model = nn.Linear(in_features, out_features, bias=True).cuda()

with torch.no_grad():
    torch_model.weight.copy_(te_model.weight)
    torch_model.bias.copy_(te_model.bias)

inp = torch.randn(hidden_size, in_features, device="cuda")

# Create an FP8 recipe. Note: All input args are optional.
fp8_recipe = recipe.DelayedScaling(margin=0, fp8_format=recipe.Format.E4M3)

# warmup GPUs
print("Warm up...")
for _ in range(10):
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        _ = te_model(inp)
    _ = torch_model(inp)

# test torch performance
torch_times = []
torch.cuda.synchronize()
print("testing torch performance")
for _ in range(num_iterations):
    start = time.perf_counter()
    out = torch_model(inp)
    torch.cuda.synchronize()
    torch_times.append(time.perf_counter() - start)

# test te performace
te_times = []
torch.cuda.synchronize()
print("tesing te performance")
for _ in range(num_iterations):
    start = time.perf_counter()
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = te_model(inp)
    torch.cuda.synchronize()
    te_times.append(time.perf_counter() - start)

te_mean = statistics.mean(te_times) * 1000  # convert to ms
te_std = statistics.stdev(te_times) * 1000
torch_mean = statistics.mean(torch_times) * 1000
torch_std = statistics.stdev(torch_times) * 1000


print("\nResults (in milliseconds):")
print(f"TransformerEngine: {te_mean:.3f} ± {te_std:.3f} ms")
print(f"PyTorch: {torch_mean:.3f} ± {torch_std:.3f} ms")
print(f"Speedup: {torch_mean/te_mean:.2f}x")

# loss = out.sum()
# loss.backward()
