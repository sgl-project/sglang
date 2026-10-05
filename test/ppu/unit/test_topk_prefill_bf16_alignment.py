import torch

from sglang.jit_kernel.dsv4.topk import top_k_per_row_prefill_bf16

B, K, topk = 4, 40000, 2048  # K > 32768 → 走 stream kernel
torch.manual_seed(0)
scores = torch.randn(B, K, dtype=torch.bfloat16, device="cuda")
row_starts = torch.tensor(
    [1, 3, 5, 7], dtype=torch.int32, device="cuda"
)  # 奇数 → 修复前必崩
lens = torch.full((B,), K - 8, dtype=torch.int32, device="cuda")
row_ends = row_starts + lens
page_table = torch.arange(K, dtype=torch.int32, device="cuda").repeat(B, 1).contiguous()
out = torch.full((B, topk), -1, dtype=torch.int32, device="cuda")

top_k_per_row_prefill_bf16(scores, row_starts, row_ends, page_table, out, 1, None)
torch.cuda.synchronize()

for i in range(B):
    s, e = int(row_starts[i]), int(row_ends[i])
    got = scores[i, s:e].float()[out[i].long()].sort(descending=True).values
    ref = scores[i, s:e].float().topk(topk).values.sort(descending=True).values
    assert torch.equal(got, ref), f"row {i} mismatch"
print("PASS")
