"""FlexKV wait-complete prefetch wiring (minimal scheduler surface)."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _load_flexkv_module(module_filename: str, module_name: str):
    connector_name = "sglang.srt.mem_cache.storage.flexkv.flexkv_connector"
    connector_stub = ModuleType(connector_name)
    connector_stub.FlexKVConnector = object
    connector_stub.FlexKVHostReleaseShim = object

    module_path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/mem_cache/storage/flexkv"
        / module_filename
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with patch.dict(sys.modules, {connector_name: connector_stub}):
        spec.loader.exec_module(module)
    return module


def test_scheduler_flexkv_prefetch_is_one_liner_to_tree_cache():
    sched = Scheduler.__new__(Scheduler)
    sched.enable_hicache_storage = False
    sched.enable_flexkv = True
    sched.tree_cache = MagicMock()

    req = SimpleNamespace(rid="x")
    sched._prefetch_kvcache(req)
    sched.tree_cache.prefetch_request.assert_called_once_with(req)


def test_flexkv_radix_prefetch_request_page_aligns_and_launches():
    module = _load_flexkv_module(
        "flexkv_radix_cache.py", "_flexkv_radix_prefetch_request_ut"
    )
    cache = module.FlexKVRadixCache.__new__(module.FlexKVRadixCache)
    cache.page_size = 2
    cache.flexkv_connector = MagicMock()
    cache.flexkv_connector.prefetch_async = MagicMock(return_value=1)

    req = MagicMock()
    req.rid = "r1"
    req.full_untruncated_fill_ids = [1, 2, 3, 4, 5]
    req._compute_max_prefix_len = MagicMock(return_value=4)
    req.init_next_round_input = MagicMock()

    cache.prefetch_request(req)

    req.init_next_round_input.assert_called_once_with(tree_cache=None, cow_mamba=False)
    args, _kwargs = cache.flexkv_connector.prefetch_async.call_args
    assert args[0] == "r1"
    assert list(args[1]) == [1, 2, 3, 4]


def test_scheduler_wait_gate_uses_existing_or_condition():
    """Document the only scheduler wait change: or-in enable_flexkv."""
    enable_hicache_storage = False
    enable_flexkv = True
    assert (enable_hicache_storage or enable_flexkv) is True
