"""Unit tests for srt/hardware_backend/npu/mc2_memory.py (Ascend DeepEP MC2
scratch guard); CPU-only, every device query is patched."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu import mc2_memory
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_PARALLEL = SimpleNamespace(
    launch_world_size=1, world_group=SimpleNamespace(cpu_group=None)
)
_SCHEDULE = SimpleNamespace(mem_fraction_static=0.85)
_DEEPEP = SimpleNamespace(is_deepep=lambda: True)
_NOT_DEEPEP = SimpleNamespace(is_deepep=lambda: False)

# Measured on a dual-node A2 16-rank EP DeepEP deployment (HCCL_BUFFSIZE=1536,
# decode graph bs=8): free MB before decode capture at --mem-fraction-static
# 0.85 (capture aborted) and at 0.80 (capture succeeded).
_FREE_MB_ABORTED = 9207
_FREE_MB_CAPTURED = 12288
_MEASURED_HEADROOM_MB = 8192


class TestHcclBuffsizeResolution(CustomTestCase):
    def test_resolution_order(self):
        # Shared with the HCCL process-group options, so this order is the
        # buffer size the communicator is actually built with.
        cases = [
            ({"DEEPEP_HCCL_BUFFSIZE": "1536", "HCCL_BUFFSIZE": "400"}, 1536),
            ({"DEEPEP_HCCL_BUFFSIZE": None, "HCCL_BUFFSIZE": "400"}, 400),
            ({"DEEPEP_HCCL_BUFFSIZE": "", "HCCL_BUFFSIZE": "400"}, 400),
            ({"DEEPEP_HCCL_BUFFSIZE": "", "HCCL_BUFFSIZE": ""}, 200),
            ({"DEEPEP_HCCL_BUFFSIZE": None, "HCCL_BUFFSIZE": None}, 200),
        ]
        for env, expected in cases:
            with self.subTest(env=env), temp_set_env(**env):
                self.assertEqual(mc2_memory.resolve_deepep_hccl_buffsize_mb(), expected)

    def test_estimate_matches_measured_point(self):
        # Measured scratch at a 1536 MB buffer: 2 x 1536 + 1 MB, padded by the margin.
        self.assertEqual(
            mc2_memory.estimate_mc2_scratch_mb(hccl_buffsize_mb=1536, margin_mb=64),
            3136,
        )


class TestMc2Preflight(CustomTestCase):
    def _run_preflight(self, *, free_mb, applies=True, is_draft_worker=False):
        with (
            temp_set_env(DEEPEP_HCCL_BUFFSIZE="1536", HCCL_BUFFSIZE=None),
            patch.object(mc2_memory, "_mc2_scratch_applies", return_value=applies),
            patch.object(mc2_memory, "get_parallel", return_value=_PARALLEL),
            patch.object(mc2_memory, "get_schedule", return_value=_SCHEDULE),
            patch.object(
                mc2_memory, "get_available_gpu_memory", return_value=free_mb / 1024
            ) as probe,
        ):
            mc2_memory.preflight_check_mc2_scratch(
                device="npu", gpu_id=0, is_draft_worker=is_draft_worker
            )
        return probe

    def test_measured_boundary_with_capture_headroom(self):
        with (
            envs.SGLANG_NPU_MC2_PREFLIGHT_CHECK.override(True),
            envs.SGLANG_NPU_MC2_CAPTURE_HEADROOM_MB.override(_MEASURED_HEADROOM_MB),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self._run_preflight(free_mb=_FREE_MB_ABORTED)
            self._run_preflight(free_mb=_FREE_MB_CAPTURED).assert_called_once()
        message = str(ctx.exception)
        for expected in (
            "9207 MB",
            "11328 MB",
            "MC2 scratch 3136 MB",
            "1536 MB [DEEPEP_HCCL_BUFFSIZE]",
            "SGLANG_NPU_MC2_SCRATCH_MARGIN_MB",
            "capture headroom 8192 MB",
            "SGLANG_NPU_MC2_CAPTURE_HEADROOM_MB",
            "--mem-fraction-static (currently 0.85)",
            "set SGLANG_NPU_RESERVE_MC2_SCRATCH=1",
        ):
            self.assertIn(expected, message)

    def test_default_headroom_is_scratch_only(self):
        with envs.SGLANG_NPU_MC2_PREFLIGHT_CHECK.override(True):
            self._run_preflight(free_mb=_FREE_MB_ABORTED).assert_called_once()
            with self.assertRaises(RuntimeError) as ctx:
                self._run_preflight(free_mb=3135)
        self.assertIn("3136 MB the capture", str(ctx.exception))
        self.assertIn("capture headroom 0 MB", str(ctx.exception))

    def test_message_reports_active_reservation(self):
        with (
            envs.SGLANG_NPU_MC2_PREFLIGHT_CHECK.override(True),
            envs.SGLANG_NPU_RESERVE_MC2_SCRATCH.override(True),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                self._run_preflight(free_mb=0)
        self.assertIn("SGLANG_NPU_RESERVE_MC2_SCRATCH is on", str(ctx.exception))

    def test_noop_when_default_disabled_predicate_false_or_draft(self):
        for env_on, applies, is_draft_worker in [
            (None, True, False),
            (False, True, False),
            (True, False, False),
            (True, True, True),
        ]:
            with (
                self.subTest(
                    env_on=env_on, applies=applies, is_draft_worker=is_draft_worker
                ),
                ExitStack() as stack,
            ):
                if env_on is not None:
                    stack.enter_context(
                        envs.SGLANG_NPU_MC2_PREFLIGHT_CHECK.override(env_on)
                    )
                probe = self._run_preflight(
                    free_mb=0, applies=applies, is_draft_worker=is_draft_worker
                )
                probe.assert_not_called()


class TestMc2Reservation(CustomTestCase):
    def test_reservation_is_opt_in_scratch_only_and_target_only(self):
        with (
            temp_set_env(DEEPEP_HCCL_BUFFSIZE="1536", HCCL_BUFFSIZE=None),
            patch.object(mc2_memory, "_mc2_scratch_applies", return_value=True),
            envs.SGLANG_NPU_MC2_CAPTURE_HEADROOM_MB.override(_MEASURED_HEADROOM_MB),
        ):
            self.assertEqual(
                mc2_memory.mc2_scratch_reservation_gb(is_draft_worker=False), 0.0
            )
            with envs.SGLANG_NPU_RESERVE_MC2_SCRATCH.override(True):
                # The capture headroom is checked, never reserved.
                self.assertEqual(
                    mc2_memory.mc2_scratch_reservation_gb(is_draft_worker=False),
                    3136 / 1024,
                )
                self.assertEqual(
                    mc2_memory.mc2_scratch_reservation_gb(is_draft_worker=True), 0.0
                )


class TestMc2Predicate(CustomTestCase):
    def _patches(self, *, soc_name, backend=_DEEPEP):
        stack = ExitStack()
        stack.enter_context(patch.object(mc2_memory, "_is_npu", True))
        stack.enter_context(
            patch.object(mc2_memory, "get_moe_a2a_backend", return_value=backend)
        )
        stack.enter_context(
            patch.object(mc2_memory, "_get_npu_soc_name", return_value=soc_name)
        )
        return stack

    def test_applies_on_a2_deepep_only(self):
        for soc_name, backend, expected in [
            ("Ascend910B3", _DEEPEP, True),
            ("Ascend910B4_1", _DEEPEP, True),
            ("Ascend910_9392", _DEEPEP, False),
            ("Ascend910B3", _NOT_DEEPEP, False),
        ]:
            with self.subTest(soc=soc_name, deepep=backend.is_deepep()):
                with self._patches(soc_name=soc_name, backend=backend):
                    self.assertEqual(mc2_memory._mc2_scratch_applies(), expected)

    def test_a3_makes_check_and_reservation_noops(self):
        with (
            self._patches(soc_name="Ascend910_9392"),
            envs.SGLANG_NPU_MC2_PREFLIGHT_CHECK.override(True),
            envs.SGLANG_NPU_RESERVE_MC2_SCRATCH.override(True),
            patch.object(mc2_memory, "get_available_gpu_memory") as probe,
        ):
            mc2_memory.preflight_check_mc2_scratch(
                device="npu", gpu_id=0, is_draft_worker=False
            )
            self.assertEqual(
                mc2_memory.mc2_scratch_reservation_gb(is_draft_worker=False), 0.0
            )
        probe.assert_not_called()

    def test_off_npu_skips_soc_query(self):
        with (
            patch.object(mc2_memory, "_is_npu", False),
            patch.object(mc2_memory, "_get_npu_soc_name") as soc,
        ):
            self.assertFalse(mc2_memory._mc2_scratch_applies())
        soc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
