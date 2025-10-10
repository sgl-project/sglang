import torch
from sglang.srt.layers.attention.infllmv2_backend import InfLLM2Runner, InfLLM2Config
from sglang.srt.mem_cache.infllmv2_memory import InfLLM2KVViews, KVViewsSidecar, ContextMemoryManager

@torch.no_grad()
def test_prefill_then_decode_uniform():
    torch.manual_seed(0)
    dev = 'cuda'
    B, Sq, Hq, Hk, Dh = 2, 8, 8, 2, 64
    Sk1, Sk2 = 256, 320

    q = torch.randn(B, Sq, Hq, Dh, device=dev, dtype=torch.float16)
    k = torch.randn(B, Sk1, Hk, Dh, device=dev, dtype=torch.float16)
    v = torch.randn_like(k)

    kv = InfLLM2KVViews(); sidecar = KVViewsSidecar(kv)
    cfg = InfLLM2Config(enable=True, topk=2, block_size=64, incremental=True, sw_span=128, sink_len=64)
    runner = InfLLM2Runner(kv, sidecar, cfg, mem_mgr=mem)
    state = {}

    out1 = runner.forward(q, k, v, state, layer_id=0)
    assert out1.shape == q.shape

    # decode：统一增长 64
    k2 = torch.cat([k, torch.randn(B, Sk2 - Sk1, Hk, Dh, device=dev, dtype=torch.float16)], dim=1)
    v2 = torch.cat([v, torch.randn_like(k[:, :Sk2 - Sk1])], dim=1)
    out2 = runner.forward(q, k2, v2, state, layer_id=0)
    assert out2.shape == q.shape
    print('ok', out1.mean().item(), out2.mean().item())

@torch.no_grad()
def test_speculative_preview_then_commit():
    torch.manual_seed(1)
    dev = 'cuda'
    B, Sq, Hq, Hk, Dh = 1, 4, 8, 2, 64
    Sk = 192
    q = torch.randn(B, Sq, Hq, Dh, device=dev, dtype=torch.float16)
    k = torch.randn(B, Sk, Hk, Dh, device=dev, dtype=torch.float16)
    v = torch.randn_like(k)

    kv = InfLLM2KVViews(); sidecar = KVViewsSidecar(kv)
    cfg = InfLLM2Config(enable=True, topk=2, block_size=64, incremental=True)
    runner = InfLLM2Runner(kv, sidecar, cfg, mem_mgr=mem)
    state = mem.get_state("req2")

    # 1) 预填充：构建到 Sk
    _ = runner.forward(q[:, :1], k, v, state, layer_id=0)

    # 2) Speculative verify：仅可见 Sk（不提交）
    out_preview = runner.forward(q, k, v, state, layer_id=0, kv_visible_sc=Sk, no_commit=True)
    assert out_preview.shape == q.shape

    # 3) 假设接受了 2 个 token，提交 2 个新 KV
    k_commit = torch.cat([k, torch.randn(B, 2, Hk, Dh, device=dev, dtype=torch.float16)], dim=1)
    v_commit = torch.cat([v, torch.randn(B, 2, Hk, Dh, device=dev, dtype=torch.float16)], dim=1)
    out_commit = runner.forward(q[:, :2], k_commit, v_commit, state, layer_id=0)
    assert out_commit.shape == (B, 2, Hq, Dh)
    print('spec-ok', out_preview.mean().item(), out_commit.mean().item())

if __name__ == '__main__':
    test_prefill_then_decode_uniform()
    test_speculative_preview_then_commit()