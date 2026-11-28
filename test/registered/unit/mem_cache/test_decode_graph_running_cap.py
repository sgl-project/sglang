"""Default max_running_requests stays within the decode CUDA graph range (#33483).

CPU-only.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_configurator as kvc
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _patched(*, max_bs=None, bs=None, backend=Backend.FULL, disagg="null"):
    decode = SimpleNamespace(max_bs=max_bs, bs=bs, backend=backend)
    exec_cfg = SimpleNamespace(
        graph=SimpleNamespace(cuda_graph_config=SimpleNamespace(decode=decode))
    )
    return (
        patch.object(kvc, "get_exec", return_value=exec_cfg),
        patch.object(
            kvc,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=disagg),
        ),
    )


class TestDecodeGraphRunningCap(CustomTestCase):
    def _cap(self, n, **kw):
        p_exec, p_disagg = _patched(**kw)
        with p_exec, p_disagg:
            return kvc._cap_to_decode_graph_range(n)

    def test_kv_derived_limit_above_graph_range_is_capped(self):
        self.assertEqual(self._cap(4096, max_bs=256), 256)
        self.assertEqual(self._cap(4096, max_bs=48), 48)

    def test_explicit_bs_list_uses_its_largest_size(self):
        self.assertEqual(self._cap(4096, bs=[1, 8, 160, 64]), 160)

    def test_limit_within_graph_range_is_unchanged(self):
        self.assertEqual(self._cap(100, max_bs=256), 100)
        self.assertEqual(self._cap(256, max_bs=256), 256)

    def test_disabled_decode_graphs_leave_limit_unchanged(self):
        self.assertEqual(self._cap(4096, max_bs=256, backend=Backend.DISABLED), 4096)

    def test_prefill_disaggregation_worker_is_unchanged(self):
        self.assertEqual(self._cap(4096, max_bs=256, disagg="prefill"), 4096)


if __name__ == "__main__":
    unittest.main()
