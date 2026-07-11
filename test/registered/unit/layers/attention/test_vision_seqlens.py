import torch

from sglang.srt.layers.attention.vision import SingletonCache, resolve_max_seqlen
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_resolve_max_seqlen_accepts_raw_tensor_without_attribute_cache():
    cu_seqlens = torch.tensor([0, 2, 7], dtype=torch.int32)

    assert resolve_max_seqlen(cu_seqlens, cu_seqlens) == 5


def test_resolve_max_seqlen_caches_on_singleton_carrier():
    source = SingletonCache()
    cu_seqlens = torch.tensor([0, 2, 7], dtype=torch.int32)

    assert resolve_max_seqlen(source, cu_seqlens) == 5
    assert source._max_seqlen == 5
