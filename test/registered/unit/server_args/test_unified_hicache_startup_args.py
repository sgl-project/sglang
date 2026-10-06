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
"""`--enable-unified-memory --enable-hierarchical-cache` passes the argument layer
with `--pp-size > 1` and with `SGLANG_DISABLE_LAZY_COMPACTION` set.

The argument layer used to reject both combinations. This checks only that it
no longer does; it says nothing about pipeline-parallel serving. The lazy
compaction checks inside the allocator are a separate layer and stay as they
were.

    python -m pytest test/registered/unit/server_args/test_unified_hicache_startup_args.py -v
"""

import unittest
from types import SimpleNamespace

import msgspec

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.environ import envs
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_handler(**overrides):
    """Run just `handle_unified_memory_pool` over a minimal stand-in."""
    sa = ServerArgs(model_path="dummy")
    fields = {
        "enable_unified_memory": True,
        "enable_hierarchical_cache": True,
        "pp_size": 1,
        "disaggregation_mode": "null",
        "speculative_algorithm": None,
        "speculative_eagle_topk": None,
        "enable_two_batch_overlap": False,
        "enable_lmcache": False,
        "dcp_size": 1,
        "cuda_graph_config": SimpleNamespace(
            prefill=SimpleNamespace(backend=Backend.DISABLED),
            decode=SimpleNamespace(backend=Backend.FULL),
        ),
        "cuda_graph_backend_prefill": Backend.DISABLED,
    }
    fields.update(overrides)
    for name, value in fields.items():
        msgspec.Struct.__setattr__(sa, name, value)
    handle_unified_memory_pool(sa)


class TestUnifiedHiCacheStartupArgs(unittest.TestCase):
    def test_pipeline_parallel_passes_the_argument_layer(self):
        _run_handler(pp_size=2)

    def test_disabled_lazy_compaction_passes_the_argument_layer(self):
        with envs.SGLANG_DISABLE_LAZY_COMPACTION.override(True):
            _run_handler()


if __name__ == "__main__":
    unittest.main()
