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
"""`--enable-unified-memory` disables PREFILL cuda-graph capture.

BUG REGRESSION. Only decode capture is wired: the prefill graph runner builds
its ForwardBatch directly, so it never runs the unified pool's write-loc
rebind (rebind_write_loc) and the captured batch holds VIRTUAL ids -- the
captured store would silently write wrong slots.

The old gate only rejected `TC_PIECEWISE`, but the generic prefill default is
`BREAKABLE` -- so the DEFAULT unified invocation was broken; it only ever
worked when `--disable-piecewise-cuda-graph` (a deprecated alias for
`--cuda-graph-backend-prefill=disabled`) happened to be passed.

Pinned: the default is auto-disabled with a warning (unified boots out of the
box), an EXPLICIT prefill backend still raises (never silently override a
user's stated intent), and decode capture is untouched either way.

    python -m pytest test/registered/unit/server_args/test_unified_prefill_cuda_graph_gate.py -v
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_handler(*, prefill_backend, explicit):
    """Run just `handle_unified_memory_pool` over a minimal stand-in."""
    sa = ServerArgs.__new__(ServerArgs)
    cg = SimpleNamespace(
        prefill=SimpleNamespace(backend=prefill_backend),
        decode=SimpleNamespace(backend=Backend.FULL),
    )
    for name, value in {
        "enable_unified_memory": True,
        "disaggregation_mode": "null",
        "speculative_algorithm": None,
        "speculative_eagle_topk": None,
        "enable_hierarchical_cache": False,
        "enable_lmcache": False,
        "dcp_size": 1,
        "cuda_graph_config": cg,
        "cuda_graph_backend_prefill": prefill_backend if explicit else None,
    }.items():
        object.__setattr__(sa, name, value)
    handle_unified_memory_pool(sa)
    return cg


class TestUnifiedPrefillCudaGraphGate(unittest.TestCase):
    def test_default_prefill_capture_is_auto_disabled(self):
        """The generic default (BREAKABLE) must be turned off, not crash the
        server 30 seconds later inside graph capture."""
        for backend in (Backend.BREAKABLE, Backend.FULL, Backend.TC_PIECEWISE):
            cg = _run_handler(prefill_backend=backend, explicit=False)
            self.assertEqual(cg.prefill.backend, Backend.DISABLED)
            # Decode capture is the wired path and must survive untouched.
            self.assertEqual(cg.decode.backend, Backend.FULL)

    def test_explicit_prefill_backend_is_refused(self):
        """A user who explicitly asked for prefill graphs gets a clear error,
        not a silent override of their stated intent."""
        for backend in (Backend.BREAKABLE, Backend.FULL, Backend.TC_PIECEWISE):
            with self.assertRaises(ValueError) as ctx:
                _run_handler(prefill_backend=backend, explicit=True)
            self.assertIn("prefill capture is not wired", str(ctx.exception))

    def test_already_disabled_is_a_no_op(self):
        cg = _run_handler(prefill_backend=Backend.DISABLED, explicit=True)
        self.assertEqual(cg.prefill.backend, Backend.DISABLED)
        self.assertEqual(cg.decode.backend, Backend.FULL)


if __name__ == "__main__":
    unittest.main()
