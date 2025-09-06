import torch as _torch
from sglang.srt.layers.attention.infllmv2_backend import InfLLM2Runner as _Runner, InfLLM2Config as _Cfg
from sglang.srt.mem_cache.infllmv2_memory import InfLLM2KVViews as _KVV, KVViewsSidecar as _SC, ContextMemoryManager as _Mem

@_torch.no_grad()
def test_manager_incremental():
    _torch.manual_seed(42)
    dev = 'cuda'
    B, Sq, Hq, Hk, Dh = 1, 2, 8, 2, 64
    Sk1, Sk2 = 128, 160
    q = _torch.randn(B, Sq, Hq, Dh, device=dev, dtype=_torch.float16)
    k1 = _torch.randn(B, Sk1, Hk, Dh, device=dev, dtype=_torch.float16)
    v1 = _torch.randn_like(k1)
    k2 = _torch.cat([k1, _torch.randn(B, Sk2 - Sk1, Hk, Dh, device=dev, dtype=_torch.float16)], dim=1)
    v2 = _torch.cat([v1, _torch.randn(B, Sk2 - Sk1, Hk, Dh, device=dev, dtype=_torch.float16)], dim=1)

    mem = _Mem(ttl_seconds=60)
    kv = _KVV(); sc = _SC(kv); cfg = _Cfg(enable=True, topk=2, block_size=64, incremental=True)
    runner = _Runner(kv, sc, cfg, mem_mgr=mem)

    rid = 'req-mgr-1'
    # 第一次：构建并缓存
    out_a = runner.forward(q, k1, v1, state=None, layer_id=0, request_id=rid)
    # 第二次：增量（应复用 sidecar 并只追加）
    out_b = runner.forward(q, k2, v2, state=None, layer_id=0, request_id=rid)

    assert out_a.shape == out_b.shape == q.shape
    print('manager-ok', float(out_a.mean()), float(out_b.mean()))