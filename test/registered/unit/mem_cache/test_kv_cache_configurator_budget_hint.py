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
"""The --mem-fraction-static that the KV budget error suggests has to start.

The budget also subtracts the multimodal reservation and the DFLASH draft KV
ring, which do not shrink as the fraction grows. A hint that left them out
suggested a fraction that failed the same check: Qwen3-ASR-1.7B started
second on an 80 GB A100 was told 0.079, and 0.08 still failed.

    python -m pytest test/registered/unit/mem_cache/test_kv_cache_configurator_budget_hint.py -v
"""

import re
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, stage="base-a", runner_config="cpu")

SUGGESTION = re.compile(r"Raise --mem-fraction-static above (\d+\.\d+)")


def profile(mem_fraction_static, *, available, pre, mm_reservation, draft_ring):
    configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
    configurator.device = "cpu"
    configurator.gpu_id = 0
    configurator.mambaish_config = None
    configurator.model_config = SimpleNamespace(is_multimodal=True)
    schedule = SimpleNamespace(
        mem_fraction_static=mem_fraction_static, prefill_max_requests=None
    )
    with (
        patch.object(kcc, "get_available_gpu_memory", return_value=available),
        patch.object(
            kcc,
            "get_parallel",
            return_value=SimpleNamespace(
                launch_world_size=1, world_group=SimpleNamespace(cpu_group=None)
            ),
        ),
        patch.object(kcc, "get_schedule", return_value=schedule),
        patch.object(
            kcc, "get_disagg", return_value=SimpleNamespace(disaggregation_mode="null")
        ),
        patch.object(kcc, "get_exec", return_value=MagicMock()),
        patch.object(
            kcc, "get_mm", return_value=SimpleNamespace(mm_feature_transport=None)
        ),
        patch.object(kcc, "mm_runtime_reservation_gb", return_value=mm_reservation),
        patch.object(
            kcc.KVCacheConfigurator,
            "_dflash_draft_kv_ring_bytes",
            new=lambda self: int(draft_ring * (1 << 30)),
        ),
    ):
        return configurator._profile_available_bytes(pre)


class TestBudgetHint(CustomTestCase):
    def assert_hint_is_the_minimum(self, mem_fraction_static, **budget):
        with self.assertRaises(ValueError) as raised:
            profile(mem_fraction_static, **budget)
        suggested = float(SUGGESTION.search(str(raised.exception)).group(1))
        self.assertGreater(profile(suggested, **budget), 0)
        with self.assertRaises(ValueError):
            profile(round(suggested - 0.001, 3), **budget)

    def test_multimodal_reservation(self):
        # Qwen3-ASR-1.7B after Qwen3-4B on an 80 GB A100: 45.80 GB free after
        # loading, 0.10 GB multimodal reservation, and about 49.73 GB free
        # before loading, which is what the 0.0790 it was told implies.
        self.assert_hint_is_the_minimum(
            0.08, available=45.80, pre=49.73, mm_reservation=0.10, draft_ring=0.0
        )

    def test_draft_kv_ring(self):
        self.assert_hint_is_the_minimum(
            0.10, available=40.0, pre=45.0, mm_reservation=0.0, draft_ring=0.5
        )


if __name__ == "__main__":
    unittest.main()
