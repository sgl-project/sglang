"""KV cache sharding for PD prefill on Intel XPU over the NIXL backend.

Manual, not CI: the XPU CI image does not install NIXL. Run where NIXL is
installed; the test skips when it is not:

    python3 test/manual/kv_transfer/test_disaggregation_kv_cache_sharding_nixl_xpu.py

Runs the scenarios of
test/registered/e2e/disaggregation/test_disaggregation_kv_cache_sharding_xpu.py
over NIXL, without staging (NIXL refuses staging together with sharding). Needs
4 XPUs.
"""

import importlib.util
import pathlib
import unittest

_SHARDING_TEST = (
    pathlib.Path(__file__).resolve().parents[2]
    / "registered/e2e/disaggregation/test_disaggregation_kv_cache_sharding_xpu.py"
)
# Loaded as a module, not imported by name, so unittest does not also collect
# its Mooncake classes here.
_spec = importlib.util.spec_from_file_location("_kv_sharding_xpu", _SHARDING_TEST)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

try:
    import nixl._api  # noqa: F401

    _NIXL_AVAILABLE = True
except ImportError:
    _NIXL_AVAILABLE = False

# With the memtype cache on, UCX classifies XPU memory as host and NIXL fails to
# register the KV cache (NIXL_ERR_BACKEND).
_NIXL_XPU_ENV = {"UCX_TLS": "ze_copy,ze_ipc,ib,tcp", "UCX_MEMTYPE_CACHE": "n"}


class _NixlMixin:
    kv_transfer_backend = "nixl"
    extra_prefill_env = {
        "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
        **_NIXL_XPU_ENV,
    }
    extra_decode_env = _NIXL_XPU_ENV


@unittest.skipUnless(_NIXL_AVAILABLE, "nixl not installed")
@unittest.skipUnless(_base._XPU_DEVICE_COUNT >= 4, _base._SKIP_REASON)
class TestDisaggregationKVCacheShardingNixlXpu(
    _NixlMixin, _base.TestDisaggregationKVCacheShardingXpu
):
    """MLA latent KV striped over the two prefill attention-TP ranks."""


@unittest.skipUnless(_NIXL_AVAILABLE, "nixl not installed")
@unittest.skipUnless(_base._XPU_DEVICE_COUNT >= 4, _base._SKIP_REASON)
class TestDisaggregationKVCacheShardingGqaCpNixlXpu(
    _NixlMixin, _base.TestDisaggregationKVCacheShardingGqaCpXpu
):
    """GQA KV striped over two zigzag prefill CP ranks."""


if __name__ == "__main__":
    unittest.main()
