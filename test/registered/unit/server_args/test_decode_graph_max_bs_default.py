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
"""Default decode CUDA graph `max_bs` follows `max_running_requests`.

The GPU-capacity default (up to 512) can dwarf a small running batch, in
which case the server captures — and reserves memory for — batch sizes it
can never replay. The default is clamped to the running limit; an explicit
`--cuda-graph-max-bs-decode` keeps its existing semantics.

    python -m pytest test/registered/unit/server_args/test_decode_graph_max_bs_default.py -v
"""

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups.memory_hook import handle_gpu_memory_settings
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.model_executor.cuda_graph_config import CudaGraphConfig, PhaseConfig
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

# Lands in the B200/MI300 branch (capacity default 512).
_BIG_GPU_MB = 200 * 1024


def _handled_decode_config(*, max_running_requests, decode_max_bs):
    args = ServerArgs(
        model_path="dummy",
        max_running_requests=max_running_requests,
        mem_fraction_static=0.8,
        cuda_graph_config=CudaGraphConfig(
            decode=PhaseConfig(max_bs=decode_max_bs),
            prefill=PhaseConfig(),
        ),
    )
    with (
        patch(
            "sglang.srt.arg_groups.memory_hook.use_mla_backend",
            return_value=False,
        ),
        patch(
            # `handle_gpu_memory_settings` computes `gpu_mem` itself
            # (`get_device_memory_capacity(cfg.device)`), imported at module
            # scope into `memory_hook` -- patch the name where it is looked
            # up, not its origin module.
            "sglang.srt.arg_groups.memory_hook.get_device_memory_capacity",
            return_value=_BIG_GPU_MB,
        ),
    ):
        handle_gpu_memory_settings(args)
    return resolution_result(args, "cuda_graph_config").decode


class TestDecodeGraphMaxBsDefault(CustomTestCase):
    def test_default_clamped_to_running_limit(self):
        decode = _handled_decode_config(
            max_running_requests=1, decode_max_bs=None
        )
        self.assertEqual(decode.max_bs, 1)
        self.assertEqual(decode.bs, [1])

    def test_default_kept_without_running_limit(self):
        decode = _handled_decode_config(
            max_running_requests=None, decode_max_bs=None
        )
        self.assertEqual(decode.max_bs, 512)
        self.assertEqual(decode.bs[-1], 512)

    def test_explicit_max_bs_untouched(self):
        decode = _handled_decode_config(
            max_running_requests=1, decode_max_bs=64
        )
        self.assertEqual(decode.max_bs, 64)


if __name__ == "__main__":
    unittest.main()
