# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""`--enable-unified-memory` refuses `--enable-two-batch-overlap`.

BUG REGRESSION. The combination launches and captures fine, then dies in the
forward path on the first captured decode replay. Nothing else rejects the
pair, so without this gate a running server crashes mid-serving.

    python -m pytest test/registered/unit/server_args/test_unified_tbo_gate.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_handler(*, unified, tbo):
    """Run just `handle_unified_memory_pool` over a minimal stand-in."""
    sa = ServerArgs.__new__(ServerArgs)
    for name, value in {
        "enable_unified_memory": unified,
        "enable_two_batch_overlap": tbo,
        "disaggregation_mode": "null",
        "speculative_algorithm": None,
        "speculative_eagle_topk": None,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "dcp_size": 1,
        "cuda_graph_config": SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED),
            decode=SimpleNamespace(backend=Backend.FULL),
        ),
        "cuda_graph_backend_prefill": Backend.DISABLED,
    }.items():
        object.__setattr__(sa, name, value)
    handle_unified_memory_pool(sa)


class TestUnifiedTboGate(unittest.TestCase):
    def test_tbo_with_unified_memory_is_refused(self):
        with self.assertRaises(AssertionError) as ctx:
            _run_handler(unified=True, tbo=True)
        self.assertIn("two-batch-overlap", str(ctx.exception))

    def test_gate_fires_only_on_the_pair(self):
        """An inverted condition here would reject every unified launch."""
        _run_handler(unified=True, tbo=False)
        _run_handler(unified=False, tbo=True)


if __name__ == "__main__":
    unittest.main()
