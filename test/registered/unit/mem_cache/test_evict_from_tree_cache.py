import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _module(name: str, **attrs):
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


class _FakeSWAAllocator:
    def __init__(self, availability):
        self.availability = availability

    def full_available_size(self):
        return self.availability["full"]

    def swa_available_size(self):
        return self.availability["swa"]


@dataclasses.dataclass
class _EvictParams:
    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0


def _load_evict_from_tree_cache():
    module_name = "_common_evict_under_test"
    stubs = {
        "sglang.kernels.ops.memory.common": _module(
            "sglang.kernels.ops.memory.common",
            _get_last_loc_safe_kernel=lambda *args, **kwargs: None,
            get_last_loc_kernel=lambda *args, **kwargs: None,
        ),
        "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks": _module(
            "sglang.srt.hardware_backend.npu.dsv4.dsv4_common_hooks",
            maybe_evict_dsv4_state_on_swa=lambda *args, **kwargs: None,
        ),
        "sglang.srt.mem_cache.allocator.swa": _module(
            "sglang.srt.mem_cache.allocator.swa",
            SWATokenToKVPoolAllocator=_FakeSWAAllocator,
        ),
        "sglang.srt.mem_cache.base_prefix_cache": _module(
            "sglang.srt.mem_cache.base_prefix_cache",
            BasePrefixCache=object,
            EvictParams=_EvictParams,
        ),
        "sglang.srt.mem_cache.memory_pool": _module(
            "sglang.srt.mem_cache.memory_pool",
            HybridReqToTokenPool=object,
            ReqToTokenPool=object,
        ),
        "sglang.srt.runtime_context": _module(
            "sglang.srt.runtime_context",
            get_server_args=lambda: SimpleNamespace(),
        ),
        "sglang.srt.utils.common": _module(
            "sglang.srt.utils.common",
            ceil_align=lambda value, alignment: value,
        ),
    }
    module_path = (
        Path(__file__).resolve().parents[4] / "python/sglang/srt/mem_cache/common.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    original_modules = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module.evict_from_tree_cache


evict_from_tree_cache = _load_evict_from_tree_cache()


def _make_evict_fixture(*, full_available: int, swa_available: int):
    availability = {"full": full_available, "swa": swa_available}
    allocator = _FakeSWAAllocator(availability)

    def evict(params: _EvictParams):
        availability["full"] += params.num_tokens
        availability["swa"] += params.swa_num_tokens

    tree_cache = SimpleNamespace(
        is_chunk_cache=lambda: False,
        token_to_kv_pool_allocator=allocator,
        evict=MagicMock(side_effect=evict),
    )
    return tree_cache, availability


def test_tail_only_demand_does_not_evict_available_swa():
    tree_cache, availability = _make_evict_fixture(full_available=80, swa_available=64)

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict.assert_called_once_with(
        _EvictParams(num_tokens=48, swa_num_tokens=0)
    )
    assert availability == {"full": 128, "swa": 64}


def test_tail_only_demand_evicts_only_swa_deficit():
    tree_cache, availability = _make_evict_fixture(full_available=128, swa_available=32)

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict.assert_called_once_with(
        _EvictParams(num_tokens=0, swa_num_tokens=32)
    )
    assert availability == {"full": 128, "swa": 64}


def test_tail_only_demand_with_enough_capacity_does_not_evict():
    tree_cache, _ = _make_evict_fixture(full_available=128, swa_available=64)

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict.assert_not_called()


def test_default_swa_demand_matches_full_demand():
    tree_cache, availability = _make_evict_fixture(full_available=128, swa_available=64)

    evict_from_tree_cache(tree_cache, 128)

    tree_cache.evict.assert_called_once_with(
        _EvictParams(num_tokens=0, swa_num_tokens=64)
    )
    assert availability == {"full": 128, "swa": 128}


def test_evict_stops_when_cache_cannot_make_progress():
    tree_cache, _ = _make_evict_fixture(full_available=0, swa_available=0)
    tree_cache.evict.side_effect = None

    evict_from_tree_cache(tree_cache, 128, swa_num_tokens=64)

    tree_cache.evict.assert_called_once_with(
        _EvictParams(num_tokens=128, swa_num_tokens=64)
    )
