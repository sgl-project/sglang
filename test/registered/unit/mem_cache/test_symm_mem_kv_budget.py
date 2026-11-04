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
"""CPU regressions for taking the symmetric-memory prealloc out of the KV budget.

``prealloc_symmetric_memory_pool`` runs after the KV pool is sized, so the
prealloc must not be left to fit in ``--mem-fraction-static`` leftover slack.
See issue #44106.

 python -m pytest test/registered/unit/mem_cache/test_symm_mem_kv_budget.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestSymmMemRuntimeReservation(CustomTestCase):
    def test_disabled_is_zero(self):
        self.assertEqual(
            kcc.symm_mem_runtime_reservation_gb(enable_symm_mem=False), 0.0
        )

    def test_nonpositive_prealloc_is_zero(self):
        for prealloc_gb in (-1, 0):
            with patch.object(
                kcc.envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE,
                "get",
                return_value=prealloc_gb,
            ):
                self.assertEqual(
                    kcc.symm_mem_runtime_reservation_gb(enable_symm_mem=True),
                    0.0,
                    prealloc_gb,
                )

    def test_default_four_gib_is_reserved(self):
        with patch.object(
            kcc.envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE, "get", return_value=4
        ):
            self.assertEqual(
                kcc.symm_mem_runtime_reservation_gb(enable_symm_mem=True), 4.0
            )

    def test_custom_prealloc_is_reserved(self):
        with patch.object(
            kcc.envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE, "get", return_value=1
        ):
            self.assertEqual(
                kcc.symm_mem_runtime_reservation_gb(enable_symm_mem=True), 1.0
            )


class TestSymmMemKvBudget(CustomTestCase):
    """The 4 GiB default pool is subtracted even when mem_fraction_static is set."""

    def _profile(self, *, enable_symm_mem: bool, prealloc_gb: int) -> int:
        cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        cfg.device = "cuda"
        cfg.gpu_id = 0
        cfg.mambaish_config = None
        cfg.post_capture_kv_active = False
        cfg.model_config = SimpleNamespace(is_multimodal=False)

        schedule = SimpleNamespace(
            mem_fraction_static=0.90,
            prefill_max_requests=None,
            max_mamba_cache_size=None,
        )
        exec_ctx = SimpleNamespace(
            comm=SimpleNamespace(enable_symm_mem=enable_symm_mem),
            graph=SimpleNamespace(
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(
                        full_prefill_max_req=None,
                        backend=Backend.DISABLED,
                    )
                )
            ),
        )
        parallel = SimpleNamespace(
            launch_world_size=1,
            world_group=SimpleNamespace(cpu_group=None),
        )
        disagg = SimpleNamespace(disaggregation_mode="null")
        mm = SimpleNamespace(mm_feature_transport=None)

        with (
            patch.object(kcc, "get_available_gpu_memory", return_value=40.0),
            patch.object(kcc, "get_schedule", return_value=schedule),
            patch.object(kcc, "get_exec", return_value=exec_ctx),
            patch.object(kcc, "get_parallel", return_value=parallel),
            patch.object(kcc, "get_disagg", return_value=disagg),
            patch.object(kcc, "get_mm", return_value=mm),
            patch.object(
                kcc.envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE,
                "get",
                return_value=prealloc_gb,
            ),
        ):
            # pre_model_load_memory=100, mem_fraction=0.90 → slack=10 GB
            # available=40 GB → 30 GB without the prealloc.
            return cfg._profile_available_bytes(pre_model_load_memory=100)

    def test_explicit_mem_fraction_still_loses_the_prealloc(self):
        without = self._profile(enable_symm_mem=False, prealloc_gb=4)
        with_pool = self._profile(enable_symm_mem=True, prealloc_gb=4)
        self.assertEqual(without - with_pool, 4 << 30)

    def test_disabled_prealloc_does_not_shrink_the_budget(self):
        without = self._profile(enable_symm_mem=False, prealloc_gb=4)
        disabled = self._profile(enable_symm_mem=True, prealloc_gb=0)
        self.assertEqual(without, disabled)


if __name__ == "__main__":
    unittest.main()
