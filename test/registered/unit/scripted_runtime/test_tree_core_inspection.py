import os
import subprocess
import sys
from array import array
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.tree_core_registry import _TREE_CORE_REGISTRY
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scripted_runtime.context.lock_ref_exhauster import (
    ScriptedLockRefExhauster,
)
from sglang.test.scripted_runtime.context.radix import (
    get_all_node_hit_counts,
    get_all_node_lock_refs,
    get_node_lock_ref,
)
from sglang.test.scripted_runtime.tree_core_inspection import (
    install_tree_core_inspectors,
)

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _cache(backend, enable_session=False):
    with (
        mock.patch.dict(_TREE_CORE_REGISTRY),
        envs.SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND.override(backend),
    ):
        install_tree_core_inspectors()
        return UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
                page_size=1,
                tree_components=(ComponentType.FULL,),
                enable_session_radix_cache=enable_session,
            )
        )


def _insert(core, token_count):
    step = core.begin_insert(
        InsertParams(
            key=RadixKey(array("q", range(token_count))),
            value=torch.arange(token_count, dtype=torch.int64),
        )
    )
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    return step.result.last_device_node


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_scripted_inspection_tracks_hits_locks_and_stale_handles(backend):
    cache = _cache(backend)
    assert cache._tree_core_backend == backend
    context = SimpleNamespace(scheduler=SimpleNamespace(tree_cache=cache))
    prefix = _insert(cache.tree_core, 44)
    decode = _insert(cache.tree_core, 45)

    assert get_all_node_hit_counts(context) == {prefix: 2, decode: 1}
    assert get_all_node_lock_refs(context) == {prefix: 0, decode: 0}
    exhauster = ScriptedLockRefExhauster(context.scheduler)
    exhauster.exhaust(leave_refs=0)
    assert all(value > 0 for value in get_all_node_lock_refs(context).values())
    assert get_node_lock_ref(cache, decode) > 0
    exhauster.release()
    assert get_all_node_lock_refs(context) == {prefix: 0, decode: 0}

    cache.tree_core.reset()
    assert get_all_node_hit_counts(context) == {}
    assert get_node_lock_ref(cache, decode) == 0
    assert get_node_lock_ref(cache, None) == 0


def test_scripted_inspection_preserves_session_fallback():
    cache = _cache("rust", enable_session=True)
    assert cache._tree_core_backend == "python"
    assert type(cache.tree_core).__name__ == "UnifiedTreeCoreInspector"


@pytest.mark.parametrize("scripted", [False, True])
def test_inspectors_install_only_in_scripted_subprocesses(scripted):
    process_env = dict(os.environ)
    process_env["SGLANG_TEST_SCRIPTED_RUNTIME"] = str(scripted)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sglang.test.scripted_runtime; "
                "from sglang.srt.mem_cache.unified_cache.tree_core_registry "
                "import get_tree_core_factory; "
                "print(get_tree_core_factory('rust').__module__)"
            ),
        ],
        env=process_env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    expected_module = (
        "sglang.test.scripted_runtime.tree_core_inspection"
        if scripted
        else "sglang.srt.mem_cache.unified_cache.tree_core_registry"
    )
    assert result.stdout.strip() == expected_module


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
