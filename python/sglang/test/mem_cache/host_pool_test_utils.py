from __future__ import annotations

import sys
from enum import Enum
from types import ModuleType
from typing import Protocol, runtime_checkable


def _noop(*args: object, **kwargs: object) -> None:
    _ = args, kwargs


class _StubPageFirstDirectMhaFormat(str, Enum):
    SPLIT_KV_PLANES = "split_kv_planes"
    KV_CONTIGUOUS_PAGE_SLOT = "kv_contiguous_page_slot"


@runtime_checkable
class _StubPageFirstDirectMhaFormatProvider(Protocol):
    @property
    def page_first_direct_mha_format(self) -> _StubPageFirstDirectMhaFormat: ...


def _stub_resolve_page_first_direct_mha_format(
    allocator: object,
) -> _StubPageFirstDirectMhaFormat:
    if isinstance(allocator, _StubPageFirstDirectMhaFormatProvider):
        return allocator.page_first_direct_mha_format
    return _StubPageFirstDirectMhaFormat.SPLIT_KV_PLANES


def install_memory_pool_host_stub(
    *,
    host_kv_cache_cls: type[object] = object,
    host_tensor_allocator_cls: type[object] = object,
) -> None:
    memory_pool_host_module = ModuleType("sglang.srt.mem_cache.memory_pool_host")
    memory_pool_host_module.HostKVCache = host_kv_cache_cls
    memory_pool_host_module.HostTensorAllocator = host_tensor_allocator_cls
    memory_pool_host_module.PageFirstDirectMhaFormat = _StubPageFirstDirectMhaFormat
    memory_pool_host_module.PageFirstDirectMhaFormatProvider = (
        _StubPageFirstDirectMhaFormatProvider
    )
    memory_pool_host_module.resolve_page_first_direct_mha_format = (
        _stub_resolve_page_first_direct_mha_format
    )
    sys.modules.setdefault(
        "sglang.srt.mem_cache.memory_pool_host",
        memory_pool_host_module,
    )


def install_memory_pool_host_layout_import_stubs() -> None:
    jit_hicache_module = ModuleType("sglang.jit_kernel.hicache")
    jit_hicache_module.can_use_hicache_jit_kernel = lambda *args, **kwargs: False
    jit_hicache_module.transfer_hicache_all_layer = _noop
    jit_hicache_module.transfer_hicache_all_layer_mla = _noop
    jit_hicache_module.transfer_hicache_one_layer = _noop
    jit_hicache_module.transfer_hicache_one_layer_mla = _noop
    sys.modules.setdefault("sglang.jit_kernel.hicache", jit_hicache_module)

    memory_pool_module = ModuleType("sglang.srt.mem_cache.memory_pool")
    memory_pool_module.DSATokenToKVPool = object
    memory_pool_module.KVCache = object
    memory_pool_module.MambaPool = object
    memory_pool_module.MHATokenToKVPool = object
    memory_pool_module.MLATokenToKVPool = object
    sys.modules.setdefault("sglang.srt.mem_cache.memory_pool", memory_pool_module)

    utils_module = ModuleType("sglang.srt.utils")
    utils_module.is_cuda = lambda: False
    utils_module.is_hip = lambda: False
    utils_module.is_mps = lambda: False
    utils_module.is_npu = lambda: False
    utils_module.is_xpu = lambda: False
    sys.modules.setdefault("sglang.srt.utils", utils_module)

    sgl_kernel_module = ModuleType("sgl_kernel.kvcacheio")
    for name in [
        "transfer_kv_all_layer",
        "transfer_kv_all_layer_direct_lf_pf",
        "transfer_kv_all_layer_lf_pf",
        "transfer_kv_all_layer_lf_ph",
        "transfer_kv_all_layer_mla",
        "transfer_kv_all_layer_mla_lf_pf",
        "transfer_kv_direct",
        "transfer_kv_per_layer",
        "transfer_kv_per_layer_direct_pf_lf",
        "transfer_kv_per_layer_mla",
        "transfer_kv_per_layer_mla_pf_lf",
        "transfer_kv_per_layer_pf_lf",
        "transfer_kv_per_layer_ph_lf",
    ]:
        setattr(sgl_kernel_module, name, _noop)
    sys.modules.setdefault("sgl_kernel.kvcacheio", sgl_kernel_module)
