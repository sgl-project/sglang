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
"""`--enable-unified-memory` and PREFILL cuda-graph capture.

Capture is wired: the captured batch reads `out_cache_loc` out of the registry
slot, refilled before each replay from the already-rebound kernel-facing loc,
and the read tables are refilled out-of-graph from the live v2p. So BREAKABLE
(the CUDA default) and TC_PIECEWISE must be left alone -- an earlier gate
disabled every prefill backend outright, which cost every unified run its
prefill graph.

The FULL backend is the exception, and for a reason that is not about unified
memory: its metadata path (`_init_full_cg_prefill_metadata`) is implemented
only by the fa3/fa4 family. Anything else lands in the decode-shaped
`_apply_cuda_graph_metadata`, which has no EXTEND branch.

Pinned here: FULL survives on fa3/fa4, FULL is disabled with a warning on any
other backend (Inkling declares FULL as a MODEL default, so refusing to boot
would fail on a flag the user never typed), and decode capture is never
touched.

    python -m pytest test/registered/unit/server_args/test_unified_prefill_cuda_graph_gate.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.kv_cache_hook import handle_unified_memory_pool
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_handler(*, prefill_backend, attention_backends):
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
        "cuda_graph_backend_prefill": prefill_backend,
    }.items():
        object.__setattr__(sa, name, value)
    with patch(
        "sglang.srt.arg_groups.kv_cache_hook.attention_backends_of",
        return_value=attention_backends,
    ):
        handle_unified_memory_pool(sa)
    return cg


class TestUnifiedPrefillCudaGraphGate(unittest.TestCase):
    def test_non_full_prefill_backends_are_left_enabled(self):
        """BUG REGRESSION. Unified used to disable prefill capture outright, so
        the default BREAKABLE graph silently never ran."""
        for backend in (Backend.BREAKABLE, Backend.TC_PIECEWISE):
            for attn in (("fa4", "fa4"), ("triton", "triton")):
                with self.subTest(prefill=backend, attn=attn):
                    cg = _run_handler(prefill_backend=backend, attention_backends=attn)
                    self.assertEqual(cg.prefill.backend, backend)
                    self.assertEqual(cg.decode.backend, Backend.FULL)

    def test_full_prefill_survives_on_the_fa_family(self):
        for attn in (("fa3", "fa3"), ("fa4", "fa4")):
            with self.subTest(attn=attn):
                cg = _run_handler(prefill_backend=Backend.FULL, attention_backends=attn)
                self.assertEqual(cg.prefill.backend, Backend.FULL)

    def test_full_prefill_is_disabled_on_other_backends(self):
        """Warn and fall back rather than raise: Inkling declares FULL as a
        model default, indistinguishable at this point from a user flag."""
        for attn in (("triton", "triton"), ("flashinfer", "flashinfer")):
            with self.subTest(attn=attn):
                cg = _run_handler(prefill_backend=Backend.FULL, attention_backends=attn)
                self.assertEqual(cg.prefill.backend, Backend.DISABLED)
                self.assertEqual(cg.decode.backend, Backend.FULL)

    def test_already_disabled_is_a_no_op(self):
        cg = _run_handler(
            prefill_backend=Backend.DISABLED, attention_backends=("triton", "triton")
        )
        self.assertEqual(cg.prefill.backend, Backend.DISABLED)
        self.assertEqual(cg.decode.backend, Backend.FULL)


if __name__ == "__main__":
    unittest.main()
