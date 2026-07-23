import pytest
from test_unified_radix_cache_unittest import (
    _CONFIGS,
    UnifiedRadixCacheSuite,
)

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LoadBackCase(UnifiedRadixCacheSuite, CustomTestCase):
    __test__ = False


@pytest.mark.parametrize(
    "config",
    [
        config
        for config in _CONFIGS
        if config.has_swa
        and not config.has_mamba
        and config.sliding_window_size > config.page_size
    ],
)
def test_hicache_swa_splits_oversized_host_tombstone(config):
    case = _LoadBackCase(
        methodName="test_hicache_swa_splits_oversized_host_tombstone_before_load_back"
    )
    case.cfg = config
    case.setUp()
    try:
        case.test_hicache_swa_splits_oversized_host_tombstone_before_load_back()
    finally:
        case.tearDown()
