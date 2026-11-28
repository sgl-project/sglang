import torch

from sglang.srt.mem_cache import memory_pool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_musa_uses_common_store_cache(monkeypatch):
    calls = []

    def fake_store_cache(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(memory_pool, "_is_cuda", False)
    monkeypatch.setattr(memory_pool, "_is_hip", False)
    monkeypatch.setattr(memory_pool, "_is_musa", True)
    monkeypatch.setattr(memory_pool, "can_use_store_cache", lambda *_: True)
    monkeypatch.setattr(memory_pool, "store_cache", fake_store_cache)

    k = torch.empty((4, 4), dtype=torch.bfloat16)
    v = torch.empty((4, 2), dtype=torch.bfloat16)
    k_cache = torch.empty((16, 4), dtype=torch.bfloat16)
    v_cache = torch.empty((16, 2), dtype=torch.bfloat16)
    indices = torch.tensor([1, 2, 3, 4], dtype=torch.int32)

    memory_pool._set_kv_buffer_impl(
        k,
        v,
        k_cache,
        v_cache,
        indices,
        row_dim=4,
        v_row_dim=2,
        store_dtype=torch.bfloat16,
        device_module=torch,
        size_limit=16,
    )

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].shape == (4, 4)
    assert args[1].shape == (4, 2)
    assert args[2].shape == (16, 4)
    assert args[3].shape == (16, 2)
    assert args[4] is indices
    assert kwargs == {
        "row_bytes": 8,
        "v_row_bytes": 4,
        "size_limit": 16,
    }
