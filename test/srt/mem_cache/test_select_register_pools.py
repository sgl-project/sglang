# ponytail: fake pools via spec-mocks, no real KV alloc; DSA must be checked
# before MLA since DSATokenToKVPool subclasses MLATokenToKVPool.
from unittest.mock import MagicMock

import pytest

from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import (
    _select_register_pools,
)


def test_mla_registers_kv_buffer_as_k_empty_v():
    p = MagicMock(spec=MLATokenToKVPool)
    p.kv_buffer = ["l0", "l1"]
    assert _select_register_pools(p) == (["l0", "l1"], [])


def test_dsa_raises_notimplemented():
    # DSA ⊂ MLA: must hit the DSA branch, not the MLA one.
    p = MagicMock(spec=DSATokenToKVPool)
    with pytest.raises(NotImplementedError):
        _select_register_pools(p)


def test_mha_registers_both_buffers():
    p = MagicMock(spec=MHATokenToKVPool)
    p.k_buffer = ["k0"]
    p.v_buffer = ["v0"]
    assert _select_register_pools(p) == (["k0"], ["v0"])
