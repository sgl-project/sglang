from types import SimpleNamespace

import pytest
import torch

from sglang.srt.disaggregation.fake.kv_reset import zero_fake_mha_kv_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def pool(value=7):
    return SimpleNamespace(
        k_buffer=[torch.full((4, 2), value, dtype=torch.bfloat16)],
        v_buffer=[torch.full((4, 2), value, dtype=torch.bfloat16)],
    )


def test_target_full_swa_and_draft_reset_in_place(monkeypatch):
    full, swa, draft = pool(), pool(), pool()
    target = SimpleNamespace(full_kv_pool=full, swa_kv_pool=swa)
    tensors = [t for p in [full, swa, draft] for t in p.k_buffer + p.v_buffer]
    pointers = [t.data_ptr() for t in tensors]
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda *_: pytest.fail("CPU reset synchronized CUDA")
    )
    result = zero_fake_mha_kv_cache(target, draft)
    assert result == {"buffers": 6, "bytes": 96}
    assert pointers == [t.data_ptr() for t in tensors]
    assert all(torch.count_nonzero(t) == 0 for t in tensors)


def test_shared_storage_slices_are_both_cleared_and_aliases_not_double_counted():
    storage = torch.ones(8)
    p = SimpleNamespace(k_buffer=[storage[:4]], v_buffer=[storage[4:]])
    result = zero_fake_mha_kv_cache(p, p)
    assert result == {"buffers": 2, "bytes": 32}
    assert torch.count_nonzero(storage) == 0


def test_unsupported_pool_is_rejected_before_any_write():
    valid = pool()
    with pytest.raises(TypeError):
        zero_fake_mha_kv_cache(valid, SimpleNamespace())
    assert torch.all(valid.k_buffer[0] == 7)


def test_noncontiguous_and_missing_buffers_are_rejected():
    bad = SimpleNamespace(k_buffer=[torch.ones(8)[::2]], v_buffer=[])
    with pytest.raises(TypeError):
        zero_fake_mha_kv_cache(bad)
    with pytest.raises(ValueError):
        zero_fake_mha_kv_cache(None)
