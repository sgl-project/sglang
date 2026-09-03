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
"""`--enable-symm-mem` is dropped under DCP on NCCL older than 2.30.7.

NCCL's symmetric collectives lost the consumer-side
`cudaGridDependencySynchronize()` barrier between 2.28 and 2.30.7 (NCCL commit
2948185), so a symmetric collective launched with programmatic dependent launch
can read its input before the producing kernel reaches coherency. With decode
context parallelism and CUDA graphs on, that silently corrupts the small decode
buckets: measured on 8xB200 (DeepSeek-V2-Lite, tp4 = dcp4, NCCL 2.29.7), GSM8K
scores 0.100 at `--parallel 1` and 0.450 at `--parallel 2` while `--parallel 32`
stays at 0.650 -- batched evals cannot see it. Preloading NCCL 2.30.7 restores
0.600/0.625, and so does dropping symmetric memory.

    python -m pytest test/registered/unit/server_args/test_dcp_symm_mem_nccl_guard.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import _dcp_symm_mem_nccl_guard
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _run_guard(*, enable_symm_mem, dcp_size, disable_cuda_graph, has_fix):
    view = SimpleNamespace(
        enable_symm_mem=enable_symm_mem,
        dcp_size=dcp_size,
        disable_cuda_graph=disable_cuda_graph,
    )
    with patch(
        "sglang.srt.distributed.device_communicators.pynccl_wrapper.nccl_has_symmetric_pdl_fix",
        return_value=has_fix,
    ):
        return _dcp_symm_mem_nccl_guard(view)


class TestDcpSymmMemNcclGuard(unittest.TestCase):
    def test_drops_symm_mem_on_old_nccl(self):
        self.assertEqual(
            _run_guard(
                enable_symm_mem=True,
                dcp_size=4,
                disable_cuda_graph=False,
                has_fix=False,
            ),
            {"enable_symm_mem": False},
        )

    def test_keeps_symm_mem_on_fixed_nccl(self):
        self.assertEqual(
            _run_guard(
                enable_symm_mem=True,
                dcp_size=4,
                disable_cuda_graph=False,
                has_fix=True,
            ),
            {},
        )

    def test_keeps_symm_mem_without_dcp(self):
        self.assertEqual(
            _run_guard(
                enable_symm_mem=True,
                dcp_size=1,
                disable_cuda_graph=False,
                has_fix=False,
            ),
            {},
        )

    def test_keeps_symm_mem_without_cuda_graph(self):
        self.assertEqual(
            _run_guard(
                enable_symm_mem=True,
                dcp_size=4,
                disable_cuda_graph=True,
                has_fix=False,
            ),
            {},
        )


if __name__ == "__main__":
    unittest.main()
