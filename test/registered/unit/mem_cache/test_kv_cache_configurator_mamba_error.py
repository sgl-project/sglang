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
"""_profile_available_bytes reports the real culprit when memory runs out.

Regression test for sgl-project/sglang#47830: serving a hybrid
linear-attention model with native MTP through EAGLE failed at startup
with "Raise --mem-fraction-static above 0.693" while the flag was already
0.9. The Mamba state cache reservation, not the weights, had exhausted
the budget, so the message pointed at a lever the user had already
pulled past the suggested value.

    python -m pytest test/registered/unit/mem_cache/test_kv_cache_configurator_mamba_error.py -v
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache import kv_cache_configurator as kcc  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_configurator(*, mambaish):
    cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
    cfg.device = "cpu"
    cfg.gpu_id = 0
    cfg.mambaish_config = SimpleNamespace() if mambaish else None
    cfg.post_capture_kv_active = False
    cfg.model_config = SimpleNamespace(is_multimodal=False)
    return cfg


class TestMambaExhaustionError(CustomTestCase):
    def _profile(self, cfg, *, available_gb, mamba_rest, pre_model_load=31.57):
        schedule = SimpleNamespace(
            mem_fraction_static=0.9,
            prefill_max_requests=None,
            max_mamba_cache_size=None,
            max_running_requests=64,
        )
        with (
            patch.object(
                kcc, "get_available_gpu_memory", return_value=available_gb
            ),
            patch.object(
                kcc,
                "get_parallel",
                return_value=SimpleNamespace(
                    launch_world_size=1,
                    world_group=SimpleNamespace(cpu_group=None),
                ),
            ),
            patch.object(kcc, "get_schedule", return_value=schedule),
            patch.object(
                kcc,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="decode"),
            ),
            patch.object(kcc, "get_exec", return_value=MagicMock()),
            patch.object(
                kcc,
                "get_mm",
                return_value=SimpleNamespace(mm_feature_transport=None),
            ),
            patch.object(kcc, "mm_runtime_reservation_gb", return_value=0.0),
            patch.object(
                kcc.KVCacheConfigurator,
                "_handle_max_mamba_cache",
                new=lambda self, rest: mamba_rest,
            ),
        ):
            return cfg._profile_available_bytes(pre_model_load)

    def test_mamba_reservation_exhaustion_names_mamba_options(self):
        # Reporter's numbers: 9.70 GB available, 3.16 GB static slack, so
        # 6.54 GB remain before an 8.00 GB Mamba reservation.
        cfg = _make_configurator(mambaish=True)
        with self.assertRaises(ValueError) as ctx:
            self._profile(cfg, available_gb=9.70, mamba_rest=9.70 - 3.16 - 8.00)
        message = str(ctx.exception)
        self.assertIn("Mamba state cache", message)
        self.assertIn("--max-running-requests", message)
        self.assertIn("8.00 GB", message)
        self.assertIn("6.54 GB", message)
        self.assertNotIn("Raise --mem-fraction-static above", message)

    def test_zero_rest_after_mamba_reports_mamba_options(self):
        cfg = _make_configurator(mambaish=True)
        with self.assertRaises(ValueError) as ctx:
            self._profile(cfg, available_gb=9.70, mamba_rest=0.0)
        self.assertIn("Mamba state cache", str(ctx.exception))

    def test_weights_only_exhaustion_keeps_mem_fraction_advice(self):
        cfg = _make_configurator(mambaish=False)
        with self.assertRaises(ValueError) as ctx:
            self._profile(cfg, available_gb=2.00, mamba_rest=0.0)
        message = str(ctx.exception)
        self.assertIn("Loaded weights", message)
        self.assertIn("Raise --mem-fraction-static above", message)

    def test_mamba_present_but_weights_exhausted_keeps_old_advice(self):
        # The budget is already negative before the Mamba reservation, so
        # the weights-only advice still applies.
        cfg = _make_configurator(mambaish=True)
        with self.assertRaises(ValueError) as ctx:
            self._profile(cfg, available_gb=2.00, mamba_rest=-10.0)
        message = str(ctx.exception)
        self.assertIn("Loaded weights", message)
        self.assertIn("Raise --mem-fraction-static above", message)


if __name__ == "__main__":
    unittest.main()
